from functools import partial
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxatari.environment import JaxEnvironment
from jaxatari.wrappers import JaxatariWrapper

_NEG_INF = -1e9


def evaluate_hrm(
    option_params,
    meta_params,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    option_net: nn.Module,
    meta_net: nn.Module,
    options,
    num_rm_states: int,
    seed: int = 1,
    option_max_steps: int = 200,
):
    env: JaxEnvironment | JaxatariWrapper = make_env(env_id)()
    key = jax.random.key(seed)

    raw_dim = env.observation_space().shape[0] - num_rm_states
    availability = options.availability

    @jax.jit
    def wrapped_reset(key):
        next_obs, state = env.reset(key)
        return next_obs.squeeze()[None, ...], state

    @jax.jit
    def wrapped_step(state, action):
        next_obs, next_state, reward, terminated, truncated, info = env.step(
            state, action.squeeze()
        )
        done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze()[None, ...], next_state, reward, done, info

    def u_of(obs):
        return jnp.argmax(obs[..., -num_rm_states:], axis=-1)

    @jax.jit
    def get_option(params, obs):
        q_vals = meta_net.apply(params, obs)[0]  # (num_options,)
        mask = availability[u_of(obs[0])]
        return jnp.argmax(jnp.where(mask, q_vals, _NEG_INF))

    @jax.jit
    def get_action(params, obs, option):
        q_vals = option_net.apply(params, obs[..., :raw_dim])[0]  # (num_options, A)
        return jnp.argmax(q_vals[option], axis=-1)

    def step_fn(carry, _):
        next_obs, env_state, option, active, length = carry

        # Commit a new option only where none is running.
        proposed = jax.vmap(get_option, in_axes=(None, 0))(meta_params, next_obs)
        option = jnp.where(active, option, proposed)
        length = jnp.where(active, length, 0)

        actions = jax.vmap(get_action, in_axes=(None, 0, 0))(
            option_params, next_obs, option
        )
        next_obs, env_state, reward, done, infos = jax.vmap(wrapped_step)(
            env_state, jnp.array(actions)
        )

        length = length + 1
        terminate = infos["option_terminate"] | done | (length >= option_max_steps)

        first_states = jax.tree.map(lambda x: x[0], env_state)
        reward = infos["env_reward"]
        carry = (next_obs, env_state, option, ~terminate, length)
        return carry, (first_states, done, reward, option)

    reset_keys = jax.random.split(key, eval_episodes)
    next_obs, env_states = jax.vmap(wrapped_reset)(reset_keys)
    carry = (
        next_obs,
        env_states,
        jnp.zeros(eval_episodes, jnp.int32),
        jnp.zeros(eval_episodes, bool),  # forces a meta choice on step 0
        jnp.zeros(eval_episodes, jnp.int32),
    )

    all_first_states, all_dones, all_rewards, all_options = [], [], [], []
    done_ever = jnp.zeros(eval_episodes, dtype=jnp.bool_)

    @partial(jax.jit, donate_argnums=(0,))
    def scanned_step(carry):
        return jax.lax.scan(step_fn, carry, None, length=1000)

    while not jnp.all(done_ever):
        carry, (first_chunk, dones_chunk, rewards_chunk, options_chunk) = scanned_step(carry)
        all_first_states.append(first_chunk)
        all_dones.append(dones_chunk)
        all_rewards.append(rewards_chunk)
        all_options.append(options_chunk)
        done_ever = done_ever | jnp.any(dones_chunk, axis=0)

    first_states_history = jax.tree.map(
        lambda *xs: jnp.concatenate(xs, axis=0), *all_first_states
    )
    dones = jnp.concatenate(all_dones, axis=0)
    rewards = jnp.concatenate(all_rewards, axis=0)
    options_history = jnp.concatenate(all_options, axis=0)

    # mask everything after the first done per episode, then sum the reward
    first_done = jnp.argmax(dones, axis=0)
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    mask_after_first_done = jnp.pad(
        has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0
    )
    masked_rewards = rewards * (1 - mask_after_first_done)
    episodic_returns = jnp.sum(masked_rewards, axis=0)

    # How the meta-controller spent the first episode. With one option per RM
    # state this is fully determined by the RM and tells you nothing
    valid = 1 - mask_after_first_done[:, 0]
    share = [
        float(jnp.sum((options_history[:, 0] == i) * valid) / jnp.maximum(jnp.sum(valid), 1))
        for i in range(options.num_options)
    ]
    print(
        f"Evaluated {eval_episodes} episodes, "
        f"mean return: {episodic_returns.mean():.2f}, "
        f"std return: {episodic_returns.std():.2f}"
    )
    print("  option usage: " + ", ".join(
        f"{n}={s:.0%}" for n, s in zip(options.names, share)
    ))

    state = first_states_history.atari_state.env_state.atari_state.env_state
    env_states_until_done = jax.tree.map(lambda x: x[: first_done[0] + 1], state)
    return episodic_returns, env_states_until_done
