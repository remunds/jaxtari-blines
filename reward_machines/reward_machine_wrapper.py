import functools
from typing import Any, Optional
import jax
import jax.numpy as jnp
from flax import struct
from jaxatari.wrappers import FlattenObservationWrapper, JaxatariWrapper, NormalizeObservationWrapper, ObjectCentricState
from agents.double_dqn.types import TimeStep
import jaxatari.spaces as spaces
import numpy as np

from agents.double_dqn.types import TimeStep
from reward_machines.options import OptionSpec
from reward_machines.reward_machine import RewardMachine

@struct.dataclass
class RewardMachineState:
    env_state: ObjectCentricState
    u: jnp.ndarray
    prev_obs: jnp.ndarray

class RewardMachineWrapper(JaxatariWrapper):
    def __init__(
        self,
        env: FlattenObservationWrapper,
        reward_machine: RewardMachine,
        use_crm: bool = False,
        use_shaping: bool = False,
        gamma: float = 0.99,
        options: Optional[OptionSpec] = None,
        ):
        super().__init__(env)
        self.rm = reward_machine
        self.use_crm = use_crm
        self.states = jnp.arange(self.rm.num_states)
        self.use_shaping = use_shaping
        self.gamma = gamma
        self.options = options
 
        base = self._env.observation_space()
        self._observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=((base.shape[0]) + self.rm.num_states,),
            dtype=jnp.float32,
        )
 

    # Apppend RM state on-hot encoded to the back of the flattend observations
    @functools.partial(jax.jit, static_argnums=(0,))
    def _augment_obs(self, obs, state):
        onehot = jax.nn.one_hot(state, self.rm.num_states)
        return jnp.concatenate([obs, onehot], axis=-1)

    def observation_space(self) -> spaces.Box:
        return self._observation_space

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        state = RewardMachineState(
            env_state=env_state,
            u=jnp.array(self.rm.init_state, dtype=jnp.int32),
            prev_obs=obs
        )
        aug_obs = self._augment_obs(obs, state.u)
        return aug_obs, state
    

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_crm_experience(self, u, action, prev_obs, obs, true_props, shaping, env_done):
        # Events are computed once by the caller and shared across all RM states.
        next_u, rm_reward, _fired_idx, rm_done = self.rm.step_from_props(u, true_props)
        prev_aug_obs = self._augment_obs(prev_obs, u)
        aug_obs = self._augment_obs(obs, next_u)
        return TimeStep(
            obs=prev_aug_obs,
            action=action,
            reward=rm_reward + shaping,
            next_obs=aug_obs,
            done=jnp.logical_or(env_done, rm_done),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _option_signals(self, u, true_props):
        clause = self.rm.clause_matches(true_props)  # (T,) bool
        option_rewards = jnp.max(
            self.options.scatter * clause[:, None].astype(jnp.float32), axis=0
        )  # (num_options,)
        # Only subgoal edges terminate an option. Including penalty edges here
        # would end every option after one step (at_surface_idle fires on
        # nearly every frame) and collapse HRM into flat DQN.
        out_of_u = self.options.is_option_edge & (self.rm.from_states == u)
        return option_rewards, jnp.any(clause & out_of_u)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        obs, env_state, _env_reward, terminated, truncated, info = self._env.step(
            state.env_state, action
        )
        done = terminated | truncated
        episode_over = jnp.logical_or(info["env_done"], truncated)
        true_props = self.rm.get_events(obs)
        true_props = jnp.where(episode_over, jnp.zeros_like(true_props), true_props)
 
        # Potential-based shaping with Phi(terminal) = 0
        if self.use_shaping:
            next_phi = jnp.where(done, 0.0, self.rm.potential(obs))
            shaping = self.gamma * next_phi - self.rm.potential(state.prev_obs)
        else:
            shaping = jnp.zeros(())

        next_u, rm_reward, fired_idx, rm_done = self.rm.step_from_props(state.u, true_props)
 
        rm_states = self.states if self.use_crm else jnp.atleast_1d(state.u)
        experiences = jax.vmap(
            self._get_crm_experience, in_axes=(0, None, None, None, None, None, None)
        )(rm_states, action, state.prev_obs, obs, true_props, shaping, done)
 
        next_u = jnp.where(episode_over, self.rm.init_state, next_u)
        aug_obs = self._augment_obs(obs, next_u)
        done = jnp.logical_or(terminated, rm_done)
        new_state = RewardMachineState(env_state=env_state, u=next_u, prev_obs=obs)
 
        info["rm_fired_idx"] = fired_idx
        info["rm_reward"] = rm_reward
        info["shaping"] = shaping
        info["experiences"] = experiences
 
        if self.options is not None:
            option_rewards, option_terminate = self._option_signals(state.u, true_props)
            info["option_rewards"] = option_rewards
            info["option_terminate"] = option_terminate
        return aug_obs, new_state, rm_reward, done, truncated, info
