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
    Model: nn.Module,
    epsilon: float = 0.05,
    seed: int = 1,
    k_tau_samples: int = 32,
):

    env: JaxEnvironment | JaxatariWrapper = make_env(env_id)()
    _Network = Model
    key = jax.random.PRNGKey(seed)

    @jax.jit
    def wrapped_reset(key):
        """wrappes the reset function of the environment to correct the observation shape"""
        next_obs, state = env.reset(key)
        return next_obs.squeeze()[None, ...], state

    @jax.jit
    def wrapped_step(state, action):
        """wrappes the step function of the environment to correct the observation shape"""
        next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze()[None, ...], next_state, reward, done, info

    key, sample_key, init_key, step_key = jax.random.split(key, 4)
    network = _Network(action_dim=env.action_space().n)

    dummy_obs = env.observation_space().sample(sample_key).squeeze()[None, ...]
    dummy_tau = jnp.zeros((1, k_tau_samples))
    q_params = network.init(init_key, dummy_obs, dummy_tau)


    with open(model_path, "rb") as f:
        (args, q_params) = flax.serialization.from_bytes((None, q_params), f.read())

    @jax.jit
    def get_action(q_params: flax.core.FrozenDict, next_obs: jnp.ndarray, key: jax.random.PRNGKey):
        key, tau_key, act_key, explore_key = jax.random.split(key, 4)
        tau = jax.random.uniform(tau_key, (next_obs.shape[0], k_tau_samples))
        q_values = jnp.mean(network.apply(q_params, next_obs, tau), axis=1)
        greedy_action = jnp.argmax(q_values, axis=1)

        random_action = jax.random.randint(act_key, greedy_action.shape, 0, env.action_space().n)
        explore = jax.random.uniform(explore_key, greedy_action.shape) < epsilon
        action = jnp.where(explore, random_action, greedy_action)

        return action, key

    def step_fn(carry, _):
        next_obs, env_state, keys = carry

        actions, keys = jax.vmap(get_action, in_axes=(None, 0, 0))(q_params, next_obs, keys)
        next_obs, env_state, reward, done, info = jax.vmap(wrapped_step)(env_state, actions)

        first_states = jax.tree.map(lambda x: x[0], env_state)

        return (next_obs, env_state, keys), (first_states, done, reward, actions)

    reset_keys = jax.random.split(key, eval_episodes)
    next_obs, env_states = jax.vmap(wrapped_reset)(reset_keys)

    step_keys = jax.random.split(step_key, eval_episodes)

    carry = (next_obs, env_states, step_keys)
    all_first_states = []
    all_dones = []
    all_rewards = []
    done_ever = jnp.zeros(eval_episodes, dtype=jnp.bool_)

    @jax.jit
    def scanned_step(carry):
        carry, (first_states_chunk, dones_chunk, rewards_chunk, actions_chunk) = jax.lax.scan(
            step_fn, carry, None, length=1000
        )
        return carry, (first_states_chunk, dones_chunk, rewards_chunk, actions_chunk)

    while not jnp.all(done_ever):
        carry, (first_states_chunk, dones_chunk, rewards_chunk, actions_chunk) = scanned_step(carry)
        all_first_states.append(first_states_chunk)
        all_dones.append(dones_chunk)
        all_rewards.append(rewards_chunk)
        done_ever = done_ever | jnp.any(dones_chunk, axis=0)

    first_states_history = jax.tree.map(
        lambda *xs: jnp.concatenate(xs, axis=0), *all_first_states
    )
    dones = jnp.concatenate(all_dones, axis=0)
    rewards = jnp.concatenate(all_rewards, axis=0)

    first_done = jnp.argmax(dones, axis=0)
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)

    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
    masked_rewards = rewards * (1 - mask_after_first_done)
    episodic_returns = jnp.sum(masked_rewards, axis=0)

    env_states_until_done = jax.tree.map(
        lambda x: x[:first_done[0] + 1],
        first_states_history.atari_state.atari_state.env_state
    )

    return episodic_returns, env_states_until_done
