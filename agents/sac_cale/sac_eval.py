from typing import Callable, Tuple, Any
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
    Model: tuple,
    seed: int = 1,
) -> Tuple[jnp.ndarray, Any]:
    """
    Evaluate a trained SAC agent deterministically.
    """
    env: JaxEnvironment | JaxatariWrapper = make_env(env_id)()
    _Encoder, _Actor, _SoftQNetwork = Model
    key = jax.random.PRNGKey(seed)

    @jax.jit
    def wrapped_reset(key):
        next_obs, state = env.reset(key)
        return next_obs, state

    @jax.jit
    def wrapped_step(state, action):
        next_obs, next_state, reward, terminated, truncated, info = env.step(state, action)
        done = jnp.logical_or(terminated, truncated)
        return next_obs, next_state, reward, done, info

    action_space = env.action_space()
    low = jnp.array(action_space.low)
    high = jnp.array(action_space.high)
    action_scale = (high - low) / 2.0
    action_bias = (high + low) / 2.0

    key, encoder_key, actor_key, qf1_key, qf2_key = jax.random.split(key, 5)

    shared_encoder = _Encoder()
    actor = _Actor(action_dim=action_space.shape[0])
    qf1 = _SoftQNetwork()
    qf2 = _SoftQNetwork()

    sample_obs = env.observation_space().sample(jax.random.PRNGKey(0))
    sample_obs = sample_obs[None, ...]

    sample_obs = sample_obs.astype(jnp.float32)

    # Initialize parameter structure for checkpoint loading
    encoder_params = shared_encoder.init(encoder_key, sample_obs)
    hidden = shared_encoder.apply(encoder_params, sample_obs)
    actor_params = actor.init(actor_key, hidden)
    dummy_action = jnp.zeros((1, action_space.shape[0]), dtype=jnp.float32)
    qf1_params = qf1.init(qf1_key, hidden, dummy_action)
    qf2_params = qf2.init(qf2_key, hidden, dummy_action)

    # Load checkpoint
    with open(model_path, "rb") as f:
        (args, (encoder_params, actor_params, qf1_params, qf2_params)) = flax.serialization.from_bytes(
            (None, (encoder_params, actor_params, qf1_params, qf2_params)), f.read()
        )

    @jax.jit
    def get_action(encoder_params, actor_params, obs):
        """Deterministic action evaluation (uses mean)."""
        obs_batch = obs[None, ...]  # Add single batch dimension: (1, ...)
        hidden = shared_encoder.apply(encoder_params, obs_batch)
        mean, _ = actor.apply(actor_params, hidden)
        action_tanh = jnp.tanh(mean.squeeze(0))
        action_env = action_bias + action_scale * action_tanh
        return action_env

    def step_fn(carry, _):
        next_obs, env_state = carry
        actions = jax.vmap(get_action, in_axes=(None, None, 0))(
            encoder_params, actor_params, next_obs
        )
        next_obs, env_state, reward, done, infos = jax.vmap(wrapped_step)(env_state, actions)
        first_states = jax.tree.map(lambda x: x[0], env_state)
        return (next_obs, env_state), (first_states, done, reward, actions)

    reset_keys = jax.random.split(key, eval_episodes)
    next_obs, env_states = jax.vmap(wrapped_reset)(reset_keys)
    _, (first_states, dones, rewards, actions) = jax.lax.scan(
        step_fn, (next_obs, env_states), None, length=27_000
    )

    # Mask rewards after episode termination
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
    masked_rewards = rewards * (1 - mask_after_first_done)
    episodic_returns = jnp.sum(masked_rewards, axis=0)
    print(f"Evaluated {eval_episodes} episodes, mean return: {episodic_returns.mean():.2f}, std return: {episodic_returns.std():.2f}")

    # Robust slice of environment states for video logging (Episode 0).
    first_done_idx = jnp.argmax(dones[:, 0])
    has_done = jnp.any(dones[:, 0])
    slice_end = int(jax.device_get(jnp.where(has_done, first_done_idx + 1, dones.shape[0])))
    env_states_until_done = jax.tree.map(lambda x: x[:slice_end], first_states)

    raw_env_states_until_done = env_states_until_done.atari_state.atari_state.env_state

    return episodic_returns, raw_env_states_until_done
