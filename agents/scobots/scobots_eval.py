"""Evaluation for the SCoBots agent, adapted from agents/ppo/ppo_eval.py.

The eval env is built with `eval=True`: no reward clipping, no episodic life and
no concept reward, so the returns are the game's own score whatever REWARD_MODE
the agent was trained with.
"""

from typing import Callable

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


def load_model(model_path: str) -> tuple[dict, dict, dict]:
    """``(config, actor_params, critic_params)`` of a checkpoint written by scobots.py."""
    with open(model_path, "rb") as f:
        blob = flax.serialization.msgpack_restore(f.read())
    # to_bytes([config, [actor, critic]]) stores every sequence as {"0": ..., "1": ...}
    config, params = blob["0"], blob["1"]
    config["HIDDEN_DIMS"] = [config["HIDDEN_DIMS"][k] for k in sorted(config["HIDDEN_DIMS"], key=int)]
    return config, params["0"], params["1"]


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    Model: type[nn.Module],
    seed=1,
    max_steps=27_000,  # 27k * 4 (frame skip) = 108k frames, the usual cap
):
    env = make_env(env_id)()
    key = jax.random.key(seed)
    config, actor_params, _critic_params = load_model(model_path)
    actor = Model(tuple(config["HIDDEN_DIMS"]), env.action_space().n, 0.01)

    @jax.jit
    def get_action(obs, key):
        logits = actor.apply(actor_params, obs)
        # sample action: Gumbel-max trick, as in training
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        return jnp.argmax(logits - jnp.log(-jnp.log(u))), key

    def step_fn(carry, _):
        obs, env_state, keys = carry
        actions, keys = jax.vmap(get_action)(obs, keys)
        obs, env_state, reward, terminated, truncated, _ = jax.vmap(env.step)(env_state, actions)
        first_states = jax.tree.map(lambda x: x[0], env_state)
        return (obs, env_state, keys), (first_states, jnp.logical_or(terminated, truncated), reward)

    # evaluate eval_episodes concurrently
    keys = jax.random.split(key, eval_episodes)
    obs, env_states = jax.vmap(env.reset)(keys)
    _, (first_states, dones, rewards) = jax.lax.scan(
        step_fn, (obs, env_states, keys), None, length=max_steps)

    # count each env's reward up to and including its first episode end
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)))
    episodic_returns = jnp.sum(rewards * (1 - mask_after_first_done), axis=0)

    # first episode's game states, for video capture
    first_done = jnp.argmax(dones, axis=0)
    env_states_until_done = jax.tree.map(
        lambda x: x[: first_done[0] + 1], first_states.atari_state.atari_state.env_state)
    return episodic_returns, env_states_until_done
