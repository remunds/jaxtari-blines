import functools
from typing import Any
import jax
import jax.numpy as jnp
from flax import struct
from jaxatari.wrappers import JaxatariWrapper, ObjectCentricState
from agents.dqn.types import TimeStep
from reward_machines.reward_machine import RewardMachine
import jaxatari.spaces as spaces
import numpy as np

@struct.dataclass
class RewardMachineState:
    env_state: ObjectCentricState
    u: jnp.ndarray
    prev_obs: jnp.ndarray

class RewardMachineWrapper(JaxatariWrapper):
    def __init__(self, env, reward_machine: RewardMachine):
        super().__init__(env)
        self.rm = reward_machine
        self.states = jnp.arange(self.rm.num_states)
        base = self._env.observation_space()
        base_low  = np.broadcast_to(base.low,  base.shape).flatten()
        base_high = np.broadcast_to(base.high, base.shape).flatten()
        new_low  = np.concatenate([base_low,  np.zeros(self.rm.num_states)])
        new_high = np.concatenate([base_high, np.ones(self.rm.num_states)])
        self._observation_space = spaces.Box(
            low=new_low,
            high=new_high,
            shape=(base.shape[0] + self.rm.num_states,),
            dtype=jnp.float32,
        )

    # Apppend RM state on-hot encoded to the back of the flattend observations
    @functools.partial(jax.jit, static_argnums=(0,))
    def _augment_obs(self, obs, state):
        onehot = jax.nn.one_hot(state, self.rm.num_states)
        return jnp.concatenate([obs, onehot], axis=-1)

    # Adapt observation space (Add rm state to obs)
    def observation_space(self) -> spaces.Box:
        """Returns a Box space for the flattened observation."""
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
    def _get_crm_experience(self, state, action, prev_obs, obs):
        next_u, rm_reward, rm_done = self.rm.step(state, obs)
        prev_aug_obs = self._augment_obs(prev_obs, state)
        aug_obs = self._augment_obs(obs, next_u)
        return TimeStep(obs=prev_aug_obs, action=action, reward=rm_reward, next_obs=aug_obs, done=rm_done)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        obs, env_state, _env_reward, terminated, truncated, info = self._env.step(state.env_state, action)
        next_u, rm_reward, rm_done = self.rm.step(state.u, obs)
        crm_experiences = jax.vmap(self._get_crm_experience, in_axes=(0, None, None, None))(self.states, action, state.prev_obs, obs)


        episode_over = jnp.logical_or(info["env_done"], truncated)
        next_u = jnp.where(episode_over, self.rm.init_state, next_u)

        aug_obs = self._augment_obs(obs, next_u)
        done = jnp.logical_or(terminated, rm_done)
        new_state = RewardMachineState(env_state=env_state, u=next_u, prev_obs=obs)

        # Add rm reward to env_reward
        # total_reward = rm_reward + _env_reward

        # Rm should fully decide on the reward
        total_reward = rm_reward

        info["rm_reward"] = rm_reward
        info["crm_experiences"] = crm_experiences
        return aug_obs, new_state, total_reward, done, truncated, info
