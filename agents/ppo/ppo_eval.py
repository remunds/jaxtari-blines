from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxtari.environment import JaxEnvironment
from jaxtari.wrappers import JaxtariWrapper


def evaluate(
    params,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    Model: tuple,
    seed: int = 1,
):
    """Run `eval_episodes` episodes in parallel with the greedy policy.

    The greedy action is the argmax of the actor's logits. `params` are the PPO
    AgentParams (network_params, actor_params, critic_params). Returns the per-episode returns and the env states of the first episode
    up to its end (used for video logging).
    """
    env: JaxEnvironment | JaxtariWrapper = make_env(env_id)()
    Network, Actor = Model
    network = Network()
    actor = Actor(action_dim=env.action_space().n)

    @jax.jit
    def wrapped_reset(key):
        """Reset the env and fix the observation shape to (1, *obs_shape)."""
        obs, state = env.reset(key)
        return obs.squeeze()[None, ...], state

    @jax.jit
    def wrapped_step(state, action):
        """Step the env and fix the observation shape to (1, *obs_shape)."""
        obs, state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return obs.squeeze()[None, ...], state, reward, done, info

    @jax.jit
    def greedy_action(obs):
        hidden = network.apply(params.network_params, obs)
        return jnp.argmax(actor.apply(params.actor_params, hidden), axis=1)

    def step_fn(carry, _):
        obs, env_state = carry
        actions = jax.vmap(greedy_action)(obs)
        obs, env_state, reward, done, _ = jax.vmap(wrapped_step)(env_state, actions)
        first_state = jax.tree.map(lambda x: x[0], env_state)
        return (obs, env_state), (first_state, done, reward)

    @jax.jit
    def scanned_step(carry):
        return jax.lax.scan(step_fn, carry, None, length=1000)

    obs, env_states = jax.vmap(wrapped_reset)(jax.random.split(jax.random.PRNGKey(seed), eval_episodes))
    carry = (obs, env_states)

    first_states_chunks, done_chunks, reward_chunks = [], [], []
    done_ever = jnp.zeros(eval_episodes, dtype=jnp.bool_)
    while not jnp.all(done_ever):
        carry, (first_states, dones, rewards) = scanned_step(carry)
        first_states_chunks.append(first_states)
        done_chunks.append(dones)
        reward_chunks.append(rewards)
        done_ever = done_ever | jnp.any(dones, axis=0)

    first_states_history = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *first_states_chunks)
    dones = jnp.concatenate(done_chunks, axis=0)
    rewards = jnp.concatenate(reward_chunks, axis=0)

    # only count rewards up to and including each episode's first done
    first_done = jnp.argmax(dones, axis=0)
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
    episodic_returns = jnp.sum(rewards * (1 - mask_after_first_done), axis=0)
    print(
        f"Evaluated {eval_episodes} episodes, mean return: {episodic_returns.mean():.2f}, "
        f"std return: {episodic_returns.std():.2f}"
    )

    # the state path depends on the wrapper stack (AtariWrapper -> PixelObsWrapper)
    env_states_until_done = jax.tree.map(
        lambda x: x[: int(first_done[0]) + 1],
        first_states_history.atari_state.atari_state.env_state,
    )
    return episodic_returns, env_states_until_done
