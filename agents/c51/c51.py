# Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_atari_jax.py
import os
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


class QNetworkPixel(nn.Module):
    action_dim: int
    n_atoms: int

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
        x = nn.Dense(self.action_dim * self.n_atoms)(x)
        x = x.reshape((x.shape[0], self.action_dim, self.n_atoms))
        return jax.nn.softmax(x, axis=-1)


class QNetworkOC(nn.Module):
    action_dim: int
    n_atoms: int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim * self.n_atoms)(x)
        x = x.reshape((x.shape[0], self.action_dim, self.n_atoms))
        return jax.nn.softmax(x, axis=-1)


class C51TrainState(TrainState):
    target_params: flax.core.FrozenDict
    atoms: jnp.ndarray


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


def build_eval_fn(env, apply_fn, v_min, v_max, n_atoms, eval_episodes, max_steps, action_dim):
    atoms = jnp.linspace(v_min, v_max, n_atoms)

    def wrapped_reset(key):
        next_obs, state = env.reset(key)
        return next_obs.squeeze()[None, ...], state

    def wrapped_step(state, action):
        next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze()[None, ...], next_state, reward, done, info

    def get_action(params, obs, key, epsilon):
        pmfs = apply_fn(params, obs)
        q_values = (pmfs * atoms[None, None, :]).sum(-1)
        greedy_action = jnp.argmax(q_values, axis=1)
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


def build_eval_return_fn(env, apply_fn, atoms, action_dim, max_steps):
    """Lightweight in-scan eval: no state history / videos, returns mean episodic return."""

    def wrapped_reset(key):
        obs, state = env.reset(key)
        return obs.squeeze()[None, ...], state

    def wrapped_step(state, action):
        obs, state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return obs.squeeze()[None, ...], state, reward, done

    def get_action(params, obs, key, epsilon):
        pmfs = apply_fn(params, obs)
        q_values = (pmfs * atoms[None, None, :]).sum(-1)
        greedy = jnp.argmax(q_values, axis=1)
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
    import random
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    if _tqdx is None:
        raise ImportError(
            "C51 scanned training needs tqdx: uv add 'tqdx @ git+https://github.com/huterguier/tqdx'"
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

    n_atoms = config.get("N_ATOMS", 51)
    v_min = config.get("V_MIN", -10.0)
    v_max = config.get("V_MAX", 10.0)
    atoms = jnp.linspace(v_min, v_max, n_atoms)
    delta_z = (v_max - v_min) / (n_atoms - 1)

    key, q_key = jax.random.split(key, 2)
    network = (
        QNetworkPixel(action_dim=action_dim, n_atoms=n_atoms)
        if pixel_based
        else QNetworkOC(action_dim=action_dim, n_atoms=n_atoms)
    )

    dummy_obs = jnp.zeros((1, *obs_shape))
    q_params = network.init(q_key, dummy_obs)

    tx = optax.adam(
        learning_rate=config.get("LEARNING_RATE"),
        eps=0.01 / config.get("BATCH_SIZE", 32),
    )

    agent_state = C51TrainState.create(
        apply_fn=network.apply,
        params=q_params,
        target_params=jax.tree.map(jnp.copy, q_params),
        atoms=atoms,
        tx=tx,
    )

    obs_dtype = jnp.uint8 if pixel_based else jnp.float16
    replay_buffer = fbx.make_item_buffer(
        max_length=config.get("BUFFER_SIZE", 100000),
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
            apply_fn=network.apply,
            v_min=v_min,
            v_max=v_max,
            n_atoms=n_atoms,
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
    inscan_eval_fn = build_eval_return_fn(inscan_eval_env, network.apply, atoms, action_dim, eval_max_steps)

    eval_reset_keys = jax.random.split(jax.random.PRNGKey(config["SEED"]), eval_episodes)

    gamma = config.get("GAMMA", 0.99)
    batch_size = config.get("BATCH_SIZE", 32)
    updates_per_step = max(1, num_envs // config.get("TRAIN_FREQUENCY", 4))

    def update(agent_state, b_obs, b_act, b_nobs, b_rew, b_don):
        next_pmfs = network.apply(agent_state.target_params, b_nobs)
        next_vals = (next_pmfs * agent_state.atoms[None, None, :]).sum(-1)
        next_act = jnp.argmax(next_vals, axis=-1)
        next_pmfs = next_pmfs[jnp.arange(batch_size), next_act]

        next_atoms = b_rew[:, None] + gamma * agent_state.atoms[None, :] * (1.0 - b_don[:, None])
        tz = jnp.clip(next_atoms, v_min, v_max)
        b = (tz - v_min) / delta_z
        l = jnp.clip(jnp.floor(b).astype(jnp.int32), 0, n_atoms - 1)
        u = jnp.clip(jnp.ceil(b).astype(jnp.int32), 0, n_atoms - 1)
        d_l = (u.astype(jnp.float32) + (l == u).astype(jnp.float32) - b) * next_pmfs
        d_u = (b - l.astype(jnp.float32)) * next_pmfs

        def project_one(l_i, u_i, dl_i, du_i):
            return jnp.zeros(n_atoms).at[l_i].add(dl_i).at[u_i].add(du_i)

        target_pmfs = jax.vmap(project_one)(l, u, d_l, d_u)
        target_pmfs = jax.lax.stop_gradient(target_pmfs)

        def loss_fn(params):
            pmfs = network.apply(params, b_obs)
            q_pred = pmfs[jnp.arange(batch_size), b_act]
            q_pred_clipped = jnp.clip(q_pred, 1e-5, 1 - 1e-5)
            loss = -(target_pmfs * jnp.log(q_pred_clipped)).sum(-1).mean()
            return loss, q_pred

        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(agent_state.params)
        new_state = agent_state.apply_gradients(grads=grads)
        return new_state, loss

    def step_once(carry, _):
        state, buffer_state, env_state, obs, rng, global_step, ep_stats = carry

        rng, action_rng, explore_rng = jax.random.split(rng, 3)
        epsilon = jnp.interp(
            global_step,
            jnp.array([0, config.get("EXPLORATION_FRACTION", 0.10) * config.get("TOTAL_TIMESTEPS", 10000000)]),
            jnp.array([config.get("START_E", 1.0), config.get("END_E", 0.01)]),
        )

        pmfs = state.apply_fn(state.params, obs)
        q_values = (pmfs * state.atoms[None, None, :]).sum(-1)
        greedy_actions = q_values.argmax(axis=-1)
        random_actions = jax.random.randint(action_rng, (num_envs,), 0, action_dim)
        explore_mask = jax.random.uniform(explore_rng, (num_envs,)) < epsilon
        actions = jnp.where(explore_mask, random_actions, greedy_actions)

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
            u_state, u_key = update_carry
            u_key, sample_key = jax.random.split(u_key)
            batch = replay_buffer.sample(buffer_state, sample_key).experience
            new_u_state, loss = update(
                u_state,
                batch["obs"].astype(jnp.float32),
                batch["action"],
                batch["next_obs"].astype(jnp.float32),
                batch["reward"],
                batch["done"].astype(jnp.float32),
            )
            return (new_u_state, u_key), loss

        should_train = (global_step % config.get("TRAIN_FREQUENCY", 4)) < num_envs
        can_train = jnp.logical_and(replay_buffer.can_sample(buffer_state), should_train)

        (state, rng), losses = jax.lax.cond(
            can_train,
            lambda c: jax.lax.scan(do_update, c, None, length=updates_per_step),
            lambda c: (c, jnp.zeros(updates_per_step)),
            (state, rng),
        )
        avg_loss = jnp.mean(losses)

        update_target_flag = jnp.logical_and(
            can_train,
            (global_step % config.get("TARGET_NETWORK_FREQUENCY", 10000)) < num_envs,
        )
        new_target_params = jax.lax.cond(
            update_target_flag,
            lambda _: optax.incremental_update(state.params, state.target_params, 1.0),
            lambda _: state.target_params,
            None,
        )
        state = state.replace(target_params=new_target_params)

        global_step += num_envs
        return (state, buffer_state, next_env_state, next_obs, rng, global_step, ep_stats), (avg_loss, epsilon)

    def save_and_eval(step_count, agent_state):
        if config.get("SAVE_PATH", "./models") is not None:
            model_path = f'{config.get("SAVE_PATH", "./models")}/{run_name}/{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes((None, agent_state.params)))
            print(f"model saved to {model_path}")

        metrics = {}
        for mods_cfg, mod_label in eval_configs:
            episodic_returns, first_states_history, first_done = eval_fns[mod_label](
                agent_state.params, eval_reset_keys, 0.05
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
        base_carry, (losses, epsilons) = jax.lax.scan(step_once, base_carry, None, length=CHUNK_SIZE)
        state, buffer_state, env_state, obs, rng, global_step, ep_stats = base_carry

        if eval_during_train:
            eval_return = jax.lax.cond(
                (i % eval_every) == 0,
                lambda p: inscan_eval_fn(p, eval_reset_keys, 0.05),
                lambda p: last_eval,
                state.params,
            )
        else:
            eval_return = last_eval

        avg_loss = jnp.sum(losses) / jnp.maximum(jnp.sum(losses != 0), 1)
        metrics = {
            "charts/global_step": global_step,
            "charts/avg_episodic_return": ep_stats.returned_episode_returns.mean(),
            "charts/avg_episodic_length": ep_stats.returned_episode_lengths.mean().astype(jnp.float32),
            "charts/epsilon": epsilons[-1],
            "losses/td_loss": avg_loss,
        }
        if eval_during_train:
            metrics[f"eval/episodic_return_{train_label}"] = eval_return
        jax.debug.callback(log_cb, metrics)

        return (base_carry, eval_return), None

    key, reset_key = jax.random.split(key)
    obs, env_state = vmap_reset(jax.random.split(reset_key, num_envs))
    global_step = jnp.array(0, dtype=jnp.int32)
    base_carry = (agent_state, buffer_state, env_state, obs, key, global_step, episode_stats)

    @partial(jax.jit, donate_argnums=(0,))
    def train(base_carry):
        if eval_during_train:
            init_eval = inscan_eval_fn(base_carry[0].params, eval_reset_keys, 0.05)
        else:
            init_eval = jnp.float32(0.0)
        carry, _ = _tqdx.scan(outer_step, (base_carry, init_eval), jnp.arange(1, total_iterations + 1))
        return carry

    print(f"[c51_scan] compiling one scan of {total_iterations} chunks x {CHUNK_SIZE * num_envs} steps...")
    start_time = time.time()
    (base_carry, _last_eval) = jax.block_until_ready(train(base_carry))
    wall = time.time() - start_time

    agent_state = base_carry[0]
    total_steps = int(base_carry[5])
    print(f"[c51_scan] {total_steps} steps in {wall:.1f}s incl. compile -> {int(total_steps / wall)} SPS (compile-inclusive)")

    eval_metrics = save_and_eval(total_steps, agent_state)
    wandb.finish()
    return eval_metrics
