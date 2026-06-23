from typing import Callable

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxatari.environment import JaxEnvironment
from jaxatari.wrappers import JaxatariWrapper


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    QNetwork: nn.Module,
    pixel_based: bool = True,
    epsilon: float = 0.05,
    seed: int = 1,
):
    env: JaxEnvironment | JaxatariWrapper = make_env(env_id, pixel_based=pixel_based, eval=True)()
    key = jax.random.PRNGKey(seed)

    @jax.jit
    def wrapped_reset(key):
        next_obs, state = env.reset(key)
        if pixel_based:
            next_obs = next_obs.squeeze(-1)
        return next_obs[None, ...], state

    @jax.jit
    def wrapped_step(state, action):
        next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        if pixel_based:
            next_obs = next_obs.squeeze(-1)
        return next_obs[None, ...], next_state, reward, done, info

    key, reset_key, net_key = jax.random.split(key, 3)
    next_obs, handle = wrapped_reset(reset_key)

    network = QNetwork(action_dim=env.action_space().n)
    dummy_obs = jnp.zeros_like(next_obs)
    q_params = network.init(net_key, dummy_obs)

    with open(model_path, "rb") as f:
        (_, q_params) = flax.serialization.from_bytes((None, q_params), f.read())

    @jax.jit
    def get_action(q_params, obs, key):
        q_values = network.apply(q_params, obs)
        greedy_action = jnp.argmax(q_values, axis=1)
        key, subkey = jax.random.split(key)
        random_action = jax.random.randint(subkey, greedy_action.shape, 0, env.action_space().n)
        explore = jax.random.uniform(key, greedy_action.shape) < epsilon
        action = jnp.where(explore, random_action, greedy_action)
        return action, key

    def step_fn(carry, _):
        next_obs, env_state, keys = carry
        actions, keys = jax.vmap(get_action, in_axes=(None, 0, 0))(q_params, next_obs, keys)
        next_obs, env_state, reward, done, info = jax.vmap(wrapped_step)(env_state, actions)
        first_state = jax.tree.map(lambda x: x[0], env_state)
        return (next_obs, env_state, keys), (first_state, done, reward)

    reset_keys = jax.random.split(key, eval_episodes)
    next_obs, env_states = jax.vmap(wrapped_reset)(reset_keys)

    _, (first_states_history, dones, rewards) = jax.lax.scan(
        step_fn, (next_obs, env_states, reset_keys), None, length=10_000
    )

    first_done = jnp.argmax(dones, axis=0)
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
    masked_rewards = rewards * (1 - mask_after_first_done)
    episodic_returns = jnp.sum(masked_rewards, axis=0)

    env_states_until_done = jax.tree.map(
        lambda x: x[:first_done[0] + 1],
        first_states_history.atari_state.atari_state.env_state,
    )

    return episodic_returns, env_states_until_done
