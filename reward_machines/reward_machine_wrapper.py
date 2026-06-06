import functools
from typing import Any
import jax
import jax.numpy as jnp
from flax import struct
from jaxatari.wrappers import JaxatariWrapper
from reward_machines.reward_machine import RewardMachine

@struct.dataclass
class RewardMachineState:
    env_state: Any
    u: jnp.ndarray

class RewardMachineWrapper(JaxatariWrapper):
    def __init__(self, env, reward_machine: RewardMachine):
        super().__init__(env)
        self.rm = reward_machine

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        state = RewardMachineState(
            env_state=env_state,
            u=jnp.array(self.rm.init_state, dtype=jnp.int32)
        )
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        obs, env_state, _env_reward, terminated, truncated, info = self._env.step(state.env_state, action)
        next_u, rm_reward, rm_done = self.rm.step(state.u, obs)

        episode_over = jnp.logical_or(info["env_done"], truncated)
        next_u = jnp.where(episode_over, self.rm.init_state, next_u)

        done = jnp.logical_or(terminated, rm_done)
        new_state = RewardMachineState(env_state=env_state, u=next_u)
        return obs, new_state, rm_reward, done, truncated, info
