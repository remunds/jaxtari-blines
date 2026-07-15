# Discrete adaptation of DDPG for JAXAtari.
#
# Reference for the continuous original:
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action_jax.py
#
# There is no discrete/Atari DDPG reference implementation, so this is an
# ADAPTATION, not a port. Design choices (documented for review):
#   - Actor outputs logits over the discrete action set. Acting samples via
#     Gumbel-softmax / categorical with temperature GUMBEL_TAU (deterministic
#     argmax at evaluation).
#   - Critic is DQN-style: Q(s, .) for all actions in one forward pass.
#   - Actor loss is the exact expected Q under the policy:
#     L_actor = -E_s[ sum_a pi(a|s) Q(s,a) ]  (lower-variance analytic form of
#     the Gumbel-softmax relaxation; gradients flow through pi).
#   - Critic target uses the target actor's argmax action and the target
#     critic: y = r + gamma * (1-d) * Q_target(s', argmax target_actor(s')).
#   - Soft target updates (polyak TAU) for both actor and critic.
#
# Training loop, logging, eval and mods handling follow the repo-wide
# outer-scan pattern (see agents/dqn_progress/dqn.py on rafet's fork and
# agents/c51/c51.py here).
import os
import random
import time
from functools import partial

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flashbax as fbx
import wandb
from flax.training.train_state import TrainState
import jaxatari
from jaxatari.wrappers import (
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    AtariWrapper,
    LogWrapper,
    FlattenObservationWrapper,
)

try:
    import tqdx as _tqdx
except ImportError:
    _tqdx = None


def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False):
    def thunk():
        active_mods = mods
        if not eval and isinstance(active_mods, (list, tuple)) and len(active_mods) > 1:
            active_mods = []

        if isinstance(active_mods, (list, tuple)) and len(active_mods) == 0:
            mods_arg = None
        else:
            mods_arg = active_mods

        env = jaxatari.make(env_id, mods=mods_arg)

        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=not eval,
            first_fire=True,
            noop_max=30,
            full_action_space=False,
        )
        if pixel_based:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(84, 84),
                grayscale=True,
                use_native_downscaling=native_downscaling,
                smooth_image=False,
                frame_stack_size=4,
                frame_skip=4,
                max_pooling=True,
                clip_reward=not eval,
            )
        else:
            env = FlattenObservationWrapper(
                NormalizeObservationWrapper(
                    ObjectCentricWrapper(
                        env,
                        frame_stack_size=4,
                        frame_skip=4,
                        clip_reward=not eval,
                    )
                )
            )
        env = LogWrapper(env)
        return env
    return thunk


class PixelTorso(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        return x


class OCTorso(nn.Module):
    hidden_size: int = 512

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        return x


class Actor(nn.Module):
    """Outputs logits over discrete actions (deterministic policy = argmax)."""
    action_dim: int
    pixel_based: bool

    @nn.compact
    def __call__(self, x):
        x = PixelTorso()(x) if self.pixel_based else OCTorso()(x)
        return nn.Dense(self.action_dim)(x)


class Critic(nn.Module):
    """DQN-style critic: Q(s, .) for all discrete actions."""
    action_dim: int
    pixel_based: bool

    @nn.compact
    def __call__(self, x):
        x = PixelTorso()(x) if self.pixel_based else OCTorso()(x)
        return nn.Dense(self.action_dim)(x)


class DDPGTrainState(TrainState):
    target_params: flax.core.FrozenDict


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


def build_eval_fn(env, actor_apply, eval_episodes, max_steps, action_dim):
    """Full eval: per-episode returns + first-env state history (for videos)."""

    def wrapped_reset(key):
        next_obs, state = env.reset(key)
        return next_obs.squeeze()[None, ...], state

    def wrapped_step(state, action):
        next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze()[None, ...], next_state, reward, done, info

    def get_action(params, obs, key, epsilon):
        logits = actor_apply(params, obs)
        greedy_action = jnp.argmax(logits, axis=1)
        key, subkey = jax.random.split(key)
        random_action = jax.random.randint(subkey, greedy_action.shape, 0, action_dim)
        explore = jax.random.uniform(key, greedy_action.shape) < epsilon
        action = jnp.where(explore, random_action, greedy_action)
        return action, key

    def step_fn(carry, _):
        obs, env_state, keys, params, epsilon = carry
        actions, keys = jax.vmap(get_action, in_axes=(None, 0, 0, None))(params, obs, keys, epsilon)
        next_obs, next_env_state, reward, done, info = jax.vmap(wrapped_step)(env_state, actions)
        first_state = jax.tree.map(lambda x: x[0], next_env_state)
        return (next_obs, next_env_state, keys, params, epsilon), (first_state, done, reward)

    @jax.jit
    def eval_fn(params, reset_keys, epsilon):
        obs, env_state = jax.vmap(wrapped_reset)(reset_keys)
        _, (first_states_history, dones, rewards) = jax.lax.scan(
            step_fn, (obs, env_state, reset_keys, params, epsilon), None, length=max_steps
        )
        has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
        mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
        masked_rewards = rewards * (1 - mask_after_first_done)
        episodic_returns = jnp.sum(masked_rewards, axis=0)
        first_done = jnp.argmax(dones, axis=0)
        return episodic_returns, first_states_history, first_done

    return eval_fn


def build_eval_return_fn(env, actor_apply, action_dim, max_steps):
    """Lightweight in-scan eval: no state history / videos, returns mean episodic return."""

    def wrapped_reset(key):
        obs, state = env.reset(key)
        return obs.squeeze()[None, ...], state

    def wrapped_step(state, action):
        obs, state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return obs.squeeze()[None, ...], state, reward, done

    def get_action(params, obs, key, epsilon):
        logits = actor_apply(params, obs)
        greedy = jnp.argmax(logits, axis=1)
        key, subkey = jax.random.split(key)
        rand = jax.random.randint(subkey, greedy.shape, 0, action_dim)
        explore = jax.random.uniform(key, greedy.shape) < epsilon
        return jnp.where(explore, rand, greedy), key

    def step_fn(carry, _):
        obs, state, keys, params, epsilon = carry
        actions, keys = jax.vmap(get_action, in_axes=(None, 0, 0, None))(params, obs, keys, epsilon)
        obs, state, reward, done = jax.vmap(wrapped_step)(state, actions)
        return (obs, state, keys, params, epsilon), (done, reward)  # no state history

    def eval_return_fn(params, reset_keys, epsilon):
        obs, state = jax.vmap(wrapped_reset)(reset_keys)
        _, (dones, rewards) = jax.lax.scan(
            step_fn, (obs, state, reset_keys, params, epsilon), None, length=max_steps
        )
        has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
        mask = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
        masked = rewards * (1 - mask)
        return jnp.mean(jnp.sum(masked, axis=0))

    return eval_return_fn


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    if _tqdx is None:
        raise ImportError(
            "DDPG scanned training needs tqdx: uv add 'tqdx @ git+https://github.com/huterguier/tqdx'"
        )

    if isinstance(config.get("TRAIN_MODS"), list):
        config["TRAIN_MODS"] = tuple(config["TRAIN_MODS"])
    if isinstance(config.get("EVAL_MODS"), list):
        config["EVAL_MODS"] = tuple(config["EVAL_MODS"])

    pixel_based = config.get("PIXEL_BASED", True)
    num_envs = config.get("NUM_ENVS", 1)
    run_name = config.get("RUN_NAME", f"{config['ENV_ID']}_{config['EXP_NAME']}_{'oc' if not pixel_based else 'pixel'}_{config['SEED']}")

    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
    )

    # do not modify the seeding
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    train_mods = list(config.get("TRAIN_MODS", []))
    env = make_env(
        config.get("ENV_ID"),
        train_mods,
        pixel_based,
        config.get("NATIVE_DOWNSCALING", True),
        False,
    )()

    action_dim = env.action_space().n
    obs_shape = env.observation_space().shape
    if pixel_based:
        obs_shape = obs_shape[:-1]

    @jax.jit
    def vmap_reset(rng):
        obs, state = jax.vmap(env.reset)(rng)
        return obs.reshape(rng.shape[0], *obs_shape), state

    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        next_done = jnp.logical_or(terminated, truncated)
        return next_obs.reshape(action.shape[0], *obs_shape), state, reward, next_done, info

    key, actor_key, critic_key = jax.random.split(key, 3)
    actor = Actor(action_dim=action_dim, pixel_based=pixel_based)
    critic = Critic(action_dim=action_dim, pixel_based=pixel_based)

    dummy_obs = jnp.zeros((1, *obs_shape))
    actor_params = actor.init(actor_key, dummy_obs)
    critic_params = critic.init(critic_key, dummy_obs)

    actor_state = DDPGTrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        target_params=jax.tree.map(jnp.copy, actor_params),
        tx=optax.adam(learning_rate=config.get("ACTOR_LR", 3e-4)),
    )
    critic_state = DDPGTrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        target_params=jax.tree.map(jnp.copy, critic_params),
        tx=optax.adam(learning_rate=config.get("CRITIC_LR", 3e-4)),
    )

    obs_dtype = jnp.uint8 if pixel_based else jnp.float16
    replay_buffer = fbx.make_item_buffer(
        max_length=config.get("BUFFER_SIZE", 200000),
        min_length=config.get("LEARNING_STARTS", 10000),
        sample_batch_size=config.get("BATCH_SIZE", 32),
        add_batches=True,
    )
    example_transition = {
        "obs": jnp.zeros(obs_shape, dtype=obs_dtype),
        "action": jnp.zeros((), dtype=jnp.int32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "done": jnp.zeros((), dtype=jnp.bool_),
        "next_obs": jnp.zeros(obs_shape, dtype=obs_dtype),
    }
    buffer_state = replay_buffer.init(example_transition)

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
    )

    eval_mods_list = list(config.get("EVAL_MODS", [])) or list(config.get("TRAIN_MODS", []))
    eval_configs = [([], "default")]
    for mod in eval_mods_list:
        mods_cfg = list(mod) if isinstance(mod, (list, tuple)) else [mod]
        mod_label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods_cfg)
        eval_configs.append((mods_cfg, mod_label))

    eval_episodes = 10
    eval_max_steps = 10000

    eval_fns = {}
    for mods_cfg, mod_label in eval_configs:
        eval_env = make_env(
            config["ENV_ID"],
            mods=mods_cfg,
            pixel_based=pixel_based,
            native_downscaling=config.get("NATIVE_DOWNSCALING", True),
            eval=True,
        )()
        eval_fns[mod_label] = build_eval_fn(
            env=eval_env,
            actor_apply=actor.apply,
            eval_episodes=eval_episodes,
            max_steps=eval_max_steps,
            action_dim=action_dim,
        )

    # in-scan eval runs on the training env config (default game if TRAIN_MODS empty)
    train_label = "default" if not train_mods else "_".join(str(m) for m in train_mods)
    inscan_eval_env = make_env(
        config["ENV_ID"],
        mods=train_mods,
        pixel_based=pixel_based,
        native_downscaling=config.get("NATIVE_DOWNSCALING", True),
        eval=True,
    )()
    inscan_eval_fn = build_eval_return_fn(inscan_eval_env, actor.apply, action_dim, eval_max_steps)

    eval_reset_keys = jax.random.split(jax.random.PRNGKey(config["SEED"]), eval_episodes)

    gamma = config.get("GAMMA", 0.99)
    batch_size = config.get("BATCH_SIZE", 32)
    polyak_tau = config.get("TAU", 0.005)
    gumbel_tau = config.get("GUMBEL_TAU", 1.0)
    updates_per_step = max(1, num_envs // config.get("TRAIN_FREQUENCY", 4))

    def update(actor_state, critic_state, b_obs, b_act, b_nobs, b_rew, b_don):
        # --- critic update ---
        next_logits = actor.apply(actor_state.target_params, b_nobs)
        next_actions = jnp.argmax(next_logits, axis=-1)  # deterministic target policy
        next_q_all = critic.apply(critic_state.target_params, b_nobs)
        next_q = next_q_all[jnp.arange(batch_size), next_actions]
        y = b_rew + gamma * (1.0 - b_don) * next_q
        y = jax.lax.stop_gradient(y)

        def critic_loss_fn(params):
            q_all = critic.apply(params, b_obs)
            q_pred = q_all[jnp.arange(batch_size), b_act]
            return jnp.mean((q_pred - y) ** 2)

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        # --- actor update: maximize expected Q under the policy ---
        def actor_loss_fn(params):
            logits = actor.apply(params, b_obs)
            probs = jax.nn.softmax(logits, axis=-1)
            q_all = jax.lax.stop_gradient(critic.apply(critic_state.params, b_obs))
            return -jnp.mean(jnp.sum(probs * q_all, axis=-1))

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        # --- soft target updates ---
        critic_state = critic_state.replace(
            target_params=optax.incremental_update(critic_state.params, critic_state.target_params, polyak_tau)
        )
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(actor_state.params, actor_state.target_params, polyak_tau)
        )
        return actor_state, critic_state, critic_loss, actor_loss

    def step_once(carry, _):
        actor_state, critic_state, buffer_state, env_state, obs, rng, global_step, ep_stats = carry

        # Gumbel-softmax exploration: sample from categorical(logits / tau)
        rng, action_rng = jax.random.split(rng)
        logits = actor.apply(actor_state.params, obs)
        actions = jax.random.categorical(action_rng, logits / gumbel_tau, axis=-1)

        next_obs, next_env_state, rewards, next_done, infos = vmap_step(env_state, actions)

        transition = {
            "obs": obs.astype(obs_dtype),
            "action": actions.astype(jnp.int32),
            "reward": rewards.astype(jnp.float32),
            "done": next_done.astype(jnp.bool_),
            "next_obs": next_obs.astype(obs_dtype),
        }
        buffer_state = replay_buffer.add(buffer_state, transition)

        new_returns = ep_stats.episode_returns + rewards
        new_lengths = ep_stats.episode_lengths + 1
        ep_stats = ep_stats.replace(
            episode_returns=new_returns * (1 - next_done),
            episode_lengths=new_lengths * (1 - next_done),
            returned_episode_returns=jnp.where(next_done, new_returns, ep_stats.returned_episode_returns),
            returned_episode_lengths=jnp.where(next_done, new_lengths, ep_stats.returned_episode_lengths),
        )

        def do_update(update_carry, _):
            a_state, c_state, u_key = update_carry
            u_key, sample_key = jax.random.split(u_key)
            batch = replay_buffer.sample(buffer_state, sample_key).experience
            a_state, c_state, critic_loss, actor_loss = update(
                a_state, c_state,
                batch["obs"].astype(jnp.float32),
                batch["action"],
                batch["next_obs"].astype(jnp.float32),
                batch["reward"],
                batch["done"].astype(jnp.float32),
            )
            return (a_state, c_state, u_key), (critic_loss, actor_loss)

        should_train = (global_step % config.get("TRAIN_FREQUENCY", 4)) < num_envs
        can_train = jnp.logical_and(replay_buffer.can_sample(buffer_state), should_train)

        (actor_state, critic_state, rng), (critic_losses, actor_losses) = jax.lax.cond(
            can_train,
            lambda c: jax.lax.scan(do_update, c, None, length=updates_per_step),
            lambda c: (c, (jnp.zeros(updates_per_step), jnp.zeros(updates_per_step))),
            (actor_state, critic_state, rng),
        )

        global_step += num_envs
        new_carry = (actor_state, critic_state, buffer_state, next_env_state, next_obs, rng, global_step, ep_stats)
        return new_carry, (jnp.mean(critic_losses), jnp.mean(actor_losses))

    def save_and_eval(step_count, actor_state):
        if config.get("SAVE_PATH", "./models") is not None:
            model_path = f'{config.get("SAVE_PATH", "./models")}/{run_name}/{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes((None, actor_state.params)))
            print(f"model saved to {model_path}")

        metrics = {}
        for mods_cfg, mod_label in eval_configs:
            episodic_returns, first_states_history, first_done = eval_fns[mod_label](
                actor_state.params, eval_reset_keys, 0.05
            )
            avg_eval_return = float(jnp.mean(episodic_returns))
            return_key = f"eval/episodic_return_{mod_label}"
            metrics[return_key] = avg_eval_return
            print(f"final eval ({mod_label}): average return = {avg_eval_return}")
            wandb.log({return_key: avg_eval_return}, step=step_count)

            if config.get("CAPTURE_VIDEO", False):
                clean_renderer = jaxatari.make(config["ENV_ID"], mods=mods_cfg).renderer
                env_states_until_done = jax.tree.map(
                    lambda x: x[: first_done[0] + 1],
                    first_states_history.atari_state.atari_state.env_state,
                )
                frames = jax.vmap(clean_renderer.render)(env_states_until_done)
                frames = jnp.transpose(frames, (0, 3, 1, 2))
                video = wandb.Video(np.array(frames), fps=30, format="mp4")
                wandb.log({f"eval/video_{mod_label}": video}, step=step_count)
                print(f"video (eval) logged with {frames.shape} frames ({mod_label}).")
        return metrics

    if config.get("NUM_STEPS"):
        CHUNK_SIZE = config["NUM_STEPS"] // num_envs
    else:
        CHUNK_SIZE = config.get("CHUNK_SIZE", 1000)
    total_iterations = config.get("TOTAL_TIMESTEPS", 10000000) // (num_envs * CHUNK_SIZE)
    eval_every = config.get("EVAL_EVERY", 10)
    eval_during_train = config.get("EVAL_DURING_TRAIN", True)

    steps_per_chunk = num_envs * CHUNK_SIZE
    _timing = {"start": None, "start_step": 0, "last": None}

    def log_cb(m):
        now = time.time()
        step = int(m["charts/global_step"])
        d = {k: float(v) for k, v in m.items()}
        d["charts/global_step"] = step
        if _timing["start"] is None:
            _timing["start"] = now
            _timing["start_step"] = step
            _timing["last"] = now
        else:
            dt = now - _timing["last"]
            elapsed = now - _timing["start"]
            d["charts/SPS_update"] = int(steps_per_chunk / dt) if dt > 0 else 0
            d["charts/SPS"] = int((step - _timing["start_step"]) / elapsed) if elapsed > 0 else 0
            _timing["last"] = now
        wandb.log(d, step=step)

    def outer_step(carry, i):
        base_carry, last_eval = carry
        base_carry, (critic_losses, actor_losses) = jax.lax.scan(step_once, base_carry, None, length=CHUNK_SIZE)
        actor_state, critic_state, buffer_state, env_state, obs, rng, global_step, ep_stats = base_carry

        if eval_during_train:
            eval_return = jax.lax.cond(
                (i % eval_every) == 0,
                lambda p: inscan_eval_fn(p, eval_reset_keys, 0.05),
                lambda p: last_eval,
                actor_state.params,
            )
        else:
            eval_return = last_eval

        avg_critic_loss = jnp.sum(critic_losses) / jnp.maximum(jnp.sum(critic_losses != 0), 1)
        avg_actor_loss = jnp.sum(actor_losses) / jnp.maximum(jnp.sum(actor_losses != 0), 1)
        metrics = {
            "charts/global_step": global_step,
            "charts/avg_episodic_return": ep_stats.returned_episode_returns.mean(),
            "charts/avg_episodic_length": ep_stats.returned_episode_lengths.mean().astype(jnp.float32),
            "losses/critic_loss": avg_critic_loss,
            "losses/actor_loss": avg_actor_loss,
        }
        if eval_during_train:
            metrics[f"eval/episodic_return_{train_label}"] = eval_return
        jax.debug.callback(log_cb, metrics)

        return (base_carry, eval_return), None

    key, reset_key = jax.random.split(key)
    obs, env_state = vmap_reset(jax.random.split(reset_key, num_envs))
    global_step = jnp.array(0, dtype=jnp.int32)
    base_carry = (actor_state, critic_state, buffer_state, env_state, obs, key, global_step, episode_stats)

    @partial(jax.jit, donate_argnums=(0,))
    def train(base_carry):
        if eval_during_train:
            init_eval = inscan_eval_fn(base_carry[0].params, eval_reset_keys, 0.05)
        else:
            init_eval = jnp.float32(0.0)
        carry, _ = _tqdx.scan(outer_step, (base_carry, init_eval), jnp.arange(1, total_iterations + 1))
        return carry

    print(f"[ddpg_scan] compiling one scan of {total_iterations} chunks x {CHUNK_SIZE * num_envs} steps...")
    start_time = time.time()
    (base_carry, _last_eval) = jax.block_until_ready(train(base_carry))
    wall = time.time() - start_time

    actor_state = base_carry[0]
    total_steps = int(base_carry[6])
    print(f"[ddpg_scan] {total_steps} steps in {wall:.1f}s incl. compile -> {int(total_steps / wall)} SPS (compile-inclusive)")

    eval_metrics = save_and_eval(total_steps, actor_state)
    wandb.finish()
    return eval_metrics
