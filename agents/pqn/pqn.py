import os
import time
from functools import partial

import flax
import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

import jaxatari
from jaxatari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
)
import wandb

try:
    import tqdx as _tqdx
except ImportError:
    _tqdx = None


@flax.struct.dataclass
class Storage:
    obs:     jnp.array
    actions: jnp.array
    rewards: jnp.array
    dones:   jnp.array
    values:  jnp.array
    returns: jnp.array


class QNetworkPixel(nn.Module):
    """CNN Q-network for pixel observations (4×84×84×1 input)."""
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.squeeze(-1)
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


class QNetworkOC(nn.Module):
    """MLP Q-network for object-centric observations (flat vector input)."""
    action_dim:  int
    hidden_size: int = 256

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


def make_env(env_id: str, mods=[], pixel_based: bool = True, eval: bool = False):
    """Returns a thunk (callable) that constructs the environment."""
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
        )
        if pixel_based:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(84, 84),
                grayscale=True,
                frame_stack_size=4,
                frame_skip=4,
                max_pooling=True,
                clip_reward=not eval,
            )
        else:
            env = ObjectCentricWrapper(env, frame_stack_size=4, frame_skip=4, clip_reward=not eval)
            env = NormalizeObservationWrapper(env)
            env = FlattenObservationWrapper(env)
        return LogWrapper(env)
    return thunk


def build_eval_fn(env, apply_fn, action_dim, max_steps):
    """Full eval: returns per-episode returns plus first-env state history (for videos)."""

    def wrapped_reset(key):
        next_obs, state = env.reset(key)
        return next_obs[None, ...], state

    def wrapped_step(state, action):
        next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return next_obs[None, ...], next_state, reward, done, info

    def get_action(params, obs, key, epsilon):
        q_values = apply_fn(params, obs)
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


def build_eval_return_fn(env, apply_fn, action_dim, max_steps):
    """Lightweight in-scan eval: no state history / videos, returns mean episodic return."""

    def wrapped_reset(key):
        obs, state = env.reset(key)
        return obs[None, ...], state

    def wrapped_step(state, action):
        obs, state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return obs[None, ...], state, reward, done

    def get_action(params, obs, key, epsilon):
        q_values = apply_fn(params, obs)
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


def single_run(config: dict) -> dict:
    config = {k.upper(): v for k, v in config.items() if k.lower() != "alg"}

    if _tqdx is None:
        raise ImportError(
            "PQN scanned training needs tqdx: uv add 'tqdx @ git+https://github.com/huterguier/tqdx'"
        )

    if isinstance(config.get("TRAIN_MODS"), list):
        config["TRAIN_MODS"] = tuple(config["TRAIN_MODS"])
    if isinstance(config.get("EVAL_MODS"), list):
        config["EVAL_MODS"] = tuple(config["EVAL_MODS"])

    pixel_based         = bool(config.get("PIXEL_BASED", True))
    game                = str(config.get("ENV_ID", "pong")).lower()
    seed                = int(config.get("SEED", 42))
    exp_name            = str(config.get("EXP_NAME", "pqn"))

    total_timesteps     = int(config.get("TOTAL_TIMESTEPS", 10_000_000))
    num_envs            = int(config.get("NUM_ENVS", 8))
    num_steps           = int(config.get("NUM_STEPS", 128))
    learning_rate       = float(config.get("LEARNING_RATE", 2.5e-4))
    anneal_lr           = bool(config.get("ANNEAL_LR", True))
    gamma               = float(config.get("GAMMA", 0.99))
    q_lambda_val        = float(config.get("Q_LAMBDA", 0.65))
    num_minibatches     = int(config.get("NUM_MINIBATCHES", 4))
    update_epochs       = int(config.get("UPDATE_EPOCHS", 4))
    max_grad_norm       = float(config.get("MAX_GRAD_NORM", 10.0))
    start_e             = float(config.get("START_E", 1.0))
    end_e               = float(config.get("END_E", 0.01))
    exploration_fraction = float(config.get("EXPLORATION_FRACTION", 0.10))
    save_path           = config.get("SAVE_PATH", "./models")

    batch_size        = num_envs * num_steps
    minibatch_size    = batch_size // num_minibatches
    num_iterations    = total_timesteps // batch_size
    exploration_steps = float(exploration_fraction * total_timesteps)

    run_name = config.get("RUN_NAME") or f"{game}_{exp_name}_{'pixel' if pixel_based else 'oc'}_{seed}"

    wandb.init(
        project=config.get("PROJECT", "jaxatari-pqn"),
        entity=config.get("ENTITY", None) or None,
        config=config,
        name=run_name,
        save_code=True,
    )

    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    key, q_key = jax.random.split(key)

    train_mods = list(config.get("TRAIN_MODS", []))
    env = make_env(game, mods=train_mods, pixel_based=pixel_based, eval=False)()

    key, probe_key = jax.random.split(key)
    _obs, _ = env.reset(probe_key)
    obs_shape = _obs.shape
    n_actions = env.action_space().n

    print(f"[PQN] run_name    : {run_name}")
    print(f"      obs_shape   : {obs_shape}")
    print(f"      n_actions   : {n_actions}")
    print(f"      num_envs    : {num_envs}")
    print(f"      batch_size  : {batch_size}")
    print(f"      minibatch   : {minibatch_size}")
    print(f"      iterations  : {num_iterations}")

    q_network = QNetworkPixel(n_actions) if pixel_based else QNetworkOC(n_actions)

    total_grad_steps = num_iterations * update_epochs * num_minibatches
    lr = (optax.linear_schedule(learning_rate, 0.0, total_grad_steps)
          if anneal_lr else learning_rate)
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.radam(lr))

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, jnp.zeros((1, *obs_shape))),
        tx=tx,
    )

    vmap_reset = jax.jit(jax.vmap(env.reset))
    vmap_step  = jax.jit(jax.vmap(env.step))

    key, *env_keys = jax.random.split(key, num_envs + 1)
    next_obs, env_states = vmap_reset(jnp.array(env_keys))
    next_done = jnp.zeros(num_envs, dtype=jnp.float32)

    # ---- eval environments (default + mods), mirroring the DQN pattern ----
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
        eval_env = make_env(game, mods=mods_cfg, pixel_based=pixel_based, eval=True)()
        eval_fns[mod_label] = build_eval_fn(
            env=eval_env,
            apply_fn=q_network.apply,
            action_dim=n_actions,
            max_steps=eval_max_steps,
        )

    # in-scan eval runs on the training env config (default game if TRAIN_MODS empty)
    train_label = "default" if not train_mods else "_".join(str(m) for m in train_mods)
    inscan_eval_env = make_env(game, mods=train_mods, pixel_based=pixel_based, eval=True)()
    inscan_eval_fn = build_eval_return_fn(inscan_eval_env, q_network.apply, n_actions, eval_max_steps)

    eval_reset_keys = jax.random.split(jax.random.PRNGKey(seed), eval_episodes)

    def step_once(carry, _):
        q_params, env_states, last_obs, last_done, key, global_step = carry

        epsilon = jnp.maximum(
            end_e,
            start_e + (end_e - start_e) * global_step.astype(jnp.float32) / exploration_steps,
        )

        q_vals      = q_network.apply(q_params, last_obs)
        max_actions = jnp.argmax(q_vals, axis=-1)
        max_vals    = q_vals[jnp.arange(num_envs), max_actions]

        key, act_key, exp_key = jax.random.split(key, 3)
        rnd     = jax.random.randint(act_key, (num_envs,), 0, n_actions)
        explore = jax.random.uniform(exp_key, (num_envs,)) < epsilon
        actions = jnp.where(explore, rnd, max_actions)

        next_obs, new_states, rewards, terminated, truncated, infos = vmap_step(env_states, actions)
        done = jnp.logical_or(terminated, truncated).astype(jnp.float32)

        storage = Storage(
            obs=last_obs, actions=actions, rewards=rewards,
            dones=last_done, values=max_vals,
            returns=jnp.zeros_like(rewards),
        )
        new_carry = (q_params, new_states, next_obs, done, key, global_step + num_envs)
        return new_carry, (storage, infos)

    @jax.jit
    def rollout(q_params, env_states, last_obs, last_done, key, global_step):
        init_carry = (q_params, env_states, last_obs, last_done, key, global_step)
        final_carry, (storage, infos) = jax.lax.scan(
            step_once, init_carry, None, length=num_steps
        )
        return final_carry, storage, infos

    def compute_q_lambda_once(carry, inp):
        next_return = carry
        reward, next_val, next_done = inp
        ret = reward + gamma * (q_lambda_val * next_return + (1.0 - q_lambda_val) * next_val) * (1.0 - next_done)
        return ret, ret

    @jax.jit
    def compute_q_lambda(agent_state: TrainState, next_obs, next_done, storage: Storage):
        next_q   = q_network.apply(agent_state.params, next_obs)
        next_val = jnp.max(next_q, axis=-1)

        next_values_t = jnp.concatenate([storage.values[1:], next_val[None]], axis=0)
        next_dones_t  = jnp.concatenate([storage.dones[1:],  next_done[None]], axis=0)

        _, returns = jax.lax.scan(
            compute_q_lambda_once,
            next_val,
            (storage.rewards, next_values_t, next_dones_t),
            reverse=True,
        )
        return storage.replace(returns=returns)

    @jax.jit
    def update_pqn(q_state: TrainState, storage: Storage, key):
        def update_epoch(carry, _):
            q_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x):
                x = jax.random.permutation(subkey, x)
                return jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])

            flat     = jax.tree_util.tree_map(flatten, storage)
            shuffled = jax.tree_util.tree_map(convert_data, flat)

            def update_minibatch(q_state, mb: Storage):
                def loss_fn(params):
                    q_vals = q_network.apply(params, mb.obs)
                    q_sel  = q_vals[jnp.arange(minibatch_size), mb.actions]
                    loss   = jnp.mean((mb.returns - q_sel) ** 2)
                    return loss, q_sel
                (loss, q_sel), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_state.params)
                q_state = q_state.apply_gradients(grads=grads)
                return q_state, (loss, q_sel.mean())

            q_state, (loss, q_val) = jax.lax.scan(update_minibatch, q_state, shuffled)
            return (q_state, key), (loss, q_val)

        (q_state, key), (loss, q_val) = jax.lax.scan(
            update_epoch, (q_state, key), (), length=update_epochs
        )
        return q_state, loss, q_val, key

    def save_and_eval(step_count, q_state):
        model_path = None
        if save_path is not None:
            model_path = f"{save_path}/{run_name}_{int(time.time())}.cleanrl_model"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes((None, q_state.params)))
            print(f"Model saved to {model_path}")

        metrics = {}
        for mods_cfg, mod_label in eval_configs:
            episodic_returns, first_states_history, first_done = eval_fns[mod_label](
                q_state.params, eval_reset_keys, 0.05
            )
            avg_eval_return = float(jnp.mean(episodic_returns))
            return_key = f"eval/episodic_return_{mod_label}"
            metrics[return_key] = avg_eval_return
            print(f"final eval ({mod_label}): average return = {avg_eval_return}")
            wandb.log({return_key: avg_eval_return}, step=step_count)

            if config.get("CAPTURE_VIDEO", False):
                clean_renderer = jaxatari.make(game, mods=mods_cfg).renderer
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

    eval_every = config.get("EVAL_EVERY", 100)
    eval_during_train = config.get("EVAL_DURING_TRAIN", True)

    steps_per_chunk = batch_size  # num_envs * num_steps per outer iteration
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
        runner_state, last_eval = carry
        q_state, env_states, next_obs, next_done, key, global_step = runner_state
        (_, env_states, next_obs, next_done, key, global_step), storage, infos = rollout(
            q_state.params, env_states, next_obs, next_done, key, global_step
        )
        storage = compute_q_lambda(q_state, next_obs, next_done, storage)
        q_state, loss, q_val, key = update_pqn(q_state, storage, key)

        if eval_during_train:
            eval_return = jax.lax.cond(
                (i % eval_every) == 0,
                lambda p: inscan_eval_fn(p, eval_reset_keys, 0.05),
                lambda p: last_eval,
                q_state.params,
            )
        else:
            eval_return = last_eval

        epsilon = jnp.maximum(
            end_e,
            start_e + (end_e - start_e) * global_step.astype(jnp.float32) / exploration_steps,
        )
        metrics = {
            "charts/global_step": global_step,
            "charts/avg_episodic_return": infos["returned_episode_returns"][-1].mean(),
            "charts/avg_episodic_length": infos["returned_episode_lengths"][-1].mean().astype(jnp.float32),
            "charts/epsilon": epsilon,
            "losses/td_loss": loss[-1, -1],
            "losses/q_values": q_val[-1, -1],
        }
        if eval_during_train:
            metrics[f"eval/episodic_return_{train_label}"] = eval_return
        jax.debug.callback(log_cb, metrics)

        return ((q_state, env_states, next_obs, next_done, key, global_step), eval_return), None

    runner_state = (q_state, env_states, next_obs, next_done, key, jnp.int32(0))

    @partial(jax.jit, donate_argnums=(0,))
    def train(runner_state):
        if eval_during_train:
            init_eval = inscan_eval_fn(runner_state[0].params, eval_reset_keys, 0.05)
        else:
            init_eval = jnp.float32(0.0)
        carry, _ = _tqdx.scan(outer_step, (runner_state, init_eval), jnp.arange(1, num_iterations + 1))
        return carry

    print(f"[pqn_scan] compiling one scan of {num_iterations} iterations x {batch_size} steps...")
    start_time = time.time()
    (runner_state, _last_eval) = jax.block_until_ready(train(runner_state))
    wall = time.time() - start_time

    q_state = runner_state[0]
    total_steps = int(runner_state[5])
    print(f"[pqn_scan] {total_steps} steps in {wall:.1f}s incl. compile -> {int(total_steps / wall)} SPS (compile-inclusive)")

    eval_metrics = save_and_eval(total_steps, q_state)
    wandb.finish()
    print("[PQN] Training complete.")
    return eval_metrics
