# Adapted from https://github.com/mttga/purejaxql
import os
import time
from functools import partial

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from rtpt import RTPT

import jaxtari
from jaxtari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
)
from agents.pqn.pqn_eval import evaluate


def make_env(env_id, mods=None, pixel_based=True, native_downscaling=True, is_eval=False):
    mods = list(mods) if mods else None
    if not is_eval and mods:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        env = jaxtari.make(env_id, mods=mods)
        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=not is_eval,
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
                clip_reward=not is_eval,
            )
        else:
            env = FlattenObservationWrapper(
                NormalizeObservationWrapper(
                    ObjectCentricWrapper(
                        env,
                        frame_stack_size=4,
                        frame_skip=4,
                        clip_reward=not is_eval,
                    )
                )
            )
        return LogWrapper(env)

    return thunk


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))  # (batch, frames, H, W) -> (batch, H, W, frames)
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID", kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID", kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID", kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)


class MLP_QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(461, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    rewards: jnp.array
    dones: jnp.array
    values: jnp.array
    returns: jnp.array


def eval_mod_configs(config: dict):
    """(mods, label) pairs to evaluate on: the default env plus each eval mod."""
    eval_mods = list(config["EVAL_MODS"] or config["TRAIN_MODS"] or [])
    mod_configs = [([], "default")]
    for mod in eval_mods:
        mods = list(mod) if isinstance(mod, (list, tuple)) else [mod]
        label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods)
        mod_configs.append((mods, label))
    return mod_configs


def single_run(config: dict):
    """Train NUM_SEEDS independent PQN agents at once by vmapping over their seeds."""
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    env_id = config["ENV_ID"]
    pixel_based = config["PIXEL_BASED"]
    num_seeds = config["NUM_SEEDS"]
    run_name = f"{env_id}_{config['EXP_NAME']}_{'pixel' if pixel_based else 'oc'}"

    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY"),
        config=config,
        name=run_name,
        save_code=True,
        mode=config.get("WANDB_MODE", "online"),
    )
    wandb.define_metric("*", step_metric="charts/global_step")

    gamma = config.get("GAMMA", 0.99)
    q_lambda = config.get("Q_LAMBDA", 0.65)
    num_steps = config.get("NUM_STEPS", 32)
    num_envs = config["NUM_ENVS"]
    num_minibatches = config.get("NUM_MINIBATCHES", 4)
    update_epochs = config.get("UPDATE_EPOCHS", 4)
    max_grad_norm = config.get("MAX_GRAD_NORM", 10.0)
    total_timesteps = config["TOTAL_TIMESTEPS"]
    start_e = config.get("START_E", 1.0)
    end_e = config.get("END_E", 0.01)
    exploration_steps = config.get("EXPLORATION_FRACTION", 0.10) * total_timesteps
    learning_rate = config.get("LEARNING_RATE", 2.5e-4)
    anneal_lr = config.get("ANNEAL_LR", True)

    env = make_env(env_id, list(config.get("TRAIN_MODS") or []), pixel_based, config.get("NATIVE_DOWNSCALING", True))()
    action_dim = env.action_space().n
    obs_shape = env.observation_space().shape
    if pixel_based:
        obs_shape = obs_shape[:-1]  # drop the trailing grayscale channel

    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_updates = total_timesteps // batch_size
    if num_updates == 0:
        raise ValueError(f"TOTAL_TIMESTEPS={total_timesteps} is smaller than one update ({batch_size} steps)")
    # updates per jitted chunk; the run is trimmed to a whole number of chunks
    scan_steps = min(config.get("SCAN_STEPS", 1000), num_updates)
    num_chunks = num_updates // scan_steps
    steps_per_chunk = scan_steps * batch_size

    network = QNetwork(action_dim=action_dim) if pixel_based else MLP_QNetwork(action_dim=action_dim)
    total_grad_steps = num_updates * update_epochs * num_minibatches
    lr = optax.linear_schedule(learning_rate, 0.0, total_grad_steps) if anneal_lr else learning_rate
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.radam(lr))

    def reset_envs(rng):
        obs, state = jax.vmap(env.reset)(rng)
        return obs.reshape(num_envs, *obs_shape), state

    def step_envs(state, action):
        obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        done = jnp.logical_or(terminated, truncated)
        return obs.reshape(num_envs, *obs_shape), state, reward, done, info

    def init_carry(seed_key):
        q_key, reset_key, run_key = jax.random.split(seed_key, 3)
        q_state = TrainState.create(
            apply_fn=network.apply,
            params=network.init(q_key, jnp.zeros((1, *obs_shape))),
            tx=tx,
        )
        obs, env_state = reset_envs(jax.random.split(reset_key, num_envs))
        return (
            q_state,
            env_state,
            obs,
            jnp.zeros(num_envs, dtype=jnp.float32),  # done flags of the last step
            run_key,
            jnp.array(0, dtype=jnp.int32),  # env steps taken by this seed
        )

    def pqn_update(carry):
        """One rollout of num_steps per env, Q(lambda) targets, and update_epochs of minibatch SGD."""
        q_state, env_state, obs, last_done, key, global_step = carry

        def rollout_step(c, _):
            env_state, last_obs, last_done, key, global_step = c
            epsilon = jnp.maximum(
                end_e,
                start_e + (end_e - start_e) * global_step.astype(jnp.float32) / exploration_steps,
            )
            q_vals = network.apply(q_state.params, last_obs)
            greedy = jnp.argmax(q_vals, axis=-1)

            key, act_key, exp_key = jax.random.split(key, 3)
            random_actions = jax.random.randint(act_key, (num_envs,), 0, action_dim)
            explore = jax.random.uniform(exp_key, (num_envs,)) < epsilon
            actions = jnp.where(explore, random_actions, greedy)

            next_obs, next_state, reward, next_done, info = step_envs(env_state, actions)
            storage = Storage(
                obs=last_obs,
                actions=actions,
                rewards=reward,
                dones=last_done,
                values=jnp.max(q_vals, axis=-1),
                returns=jnp.zeros_like(reward),
            )
            new_c = (next_state, next_obs, next_done.astype(jnp.float32), key, global_step + num_envs)
            return new_c, (storage, info)

        (env_state, next_obs, next_done, key, global_step), (storage, infos) = jax.lax.scan(
            rollout_step,
            (env_state, obs, last_done, key, global_step),
            None,
            length=num_steps,
        )

        # Q(lambda) returns, computed backwards through the rollout
        next_val = jnp.max(network.apply(q_state.params, next_obs), axis=-1)
        next_values = jnp.concatenate([storage.values[1:], next_val[None]], axis=0)
        next_dones = jnp.concatenate([storage.dones[1:], next_done[None]], axis=0)

        def q_lambda_step(next_return, inp):
            reward, next_value, nd = inp
            ret = reward + gamma * (q_lambda * next_return + (1.0 - q_lambda) * next_value) * (1.0 - nd)
            return ret, ret

        _, returns = jax.lax.scan(
            q_lambda_step,
            next_val,
            (storage.rewards, next_values, next_dones),
            reverse=True,
        )
        storage = storage.replace(returns=returns)

        def update_epoch(carry, _):
            q_state, key = carry
            key, perm_key = jax.random.split(key)
            flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), storage)
            perm = jax.random.permutation(perm_key, batch_size)
            shuffled = jax.tree.map(
                lambda x: x[perm].reshape((num_minibatches, -1) + x.shape[1:]),
                flat,
            )

            def update_minibatch(q_state, mb):
                def loss_fn(params):
                    q_vals = network.apply(params, mb.obs)
                    q_sel = q_vals[jnp.arange(minibatch_size), mb.actions]
                    loss = 0.5 * jnp.mean((mb.returns - q_sel) ** 2)
                    return loss, q_sel.mean()

                (loss, q_mean), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_state.params)
                return q_state.apply_gradients(grads=grads), (loss, q_mean)

            q_state, (loss, q_mean) = jax.lax.scan(update_minibatch, q_state, shuffled)
            return (q_state, key), (loss, q_mean)

        (q_state, key), (loss, q_mean) = jax.lax.scan(update_epoch, (q_state, key), None, length=update_epochs)

        new_carry = (q_state, env_state, next_obs, next_done, key, global_step)
        return new_carry, (infos, loss[-1, -1], q_mean[-1, -1])

    def train_chunk(carry):
        return jax.lax.scan(lambda c, _: pqn_update(c), carry, None, length=scan_steps)

    # compile once for all seeds, without running a warmup chunk
    carry = jax.vmap(init_carry)(jax.random.split(jax.random.PRNGKey(config["SEED"]), num_seeds))
    print("[pqn] compiling...")
    compile_start = time.perf_counter()
    chunk_fn = jax.jit(jax.vmap(train_chunk)).lower(carry).compile()
    print(f"[pqn] compilation time: {time.perf_counter() - compile_start:.2f}s")

    def save_and_eval(q_params, step_count):
        """Evaluate each seed's greedy policy on the default env and each eval mod."""
        save_root = config.get("SAVE_PATH")
        mod_configs = eval_mod_configs(config)
        returns_per_label = {label: [] for _, label in mod_configs}
        for seed_idx in range(num_seeds):
            params = jax.tree.map(lambda x: x[seed_idx], q_params)
            if save_root is not None:
                save_dir = os.path.join(save_root, run_name)
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, f"{config['EXP_NAME']}_seed{seed_idx}_{step_count}_{int(time.time())}.cleanrl_model")
                with open(model_path, "wb") as f:
                    f.write(flax.serialization.to_bytes([config, params]))
                print(f"model saved to {model_path}")

            print(f"running evaluation for seed {seed_idx} at step {step_count}...")
            for mods, label in mod_configs:
                episodic_returns, env_states = evaluate(
                    params,
                    partial(
                        make_env,
                        mods=mods,
                        pixel_based=pixel_based,
                        native_downscaling=config.get("NATIVE_DOWNSCALING", True),
                        is_eval=True,
                    ),
                    env_id,
                    eval_episodes=10,
                    Model=QNetwork if pixel_based else MLP_QNetwork,
                    seed=config["SEED"] + 42 + seed_idx,  # distinct from the training seed
                )
                mean_return = float(np.mean(jax.device_get(episodic_returns)))
                returns_per_label[label].append(mean_return)
                wandb.log({f"seed{seed_idx}/eval/episodic_return_{label}": mean_return}, step=step_count)

                if config.get("CAPTURE_VIDEO", False) and seed_idx == 0:
                    # render with a clean renderer, not the downscaled training env
                    renderer = jaxtari.make(env_id, mods=mods).renderer
                    frames = jnp.transpose(jax.vmap(renderer.render)(env_states), (0, 3, 1, 2))
                    video = wandb.Video(np.array(frames), fps=30, format="mp4")
                    wandb.log({f"eval/video_{label}": video}, step=step_count)
                    print(f"Video (eval) logged to wandb with {frames.shape[0]} frames ({label}).")

        metrics = {}
        for label, returns in returns_per_label.items():
            metrics[label] = float(np.mean(returns))
            wandb.log({f"eval/episodic_return_{label}": metrics[label]}, step=step_count)
        return metrics

    rtpt = RTPT(
        name_initials=config["NAME_INITIALS"],
        experiment_name=run_name,
        max_iterations=num_chunks,
    )
    rtpt.start()
    print(f"[pqn] starting training: {num_chunks} chunks x {steps_per_chunk} steps per seed")
    run_time = time.perf_counter()

    def seed_mean(x):
        """Mean over the last update of a chunk, per seed."""
        return np.asarray(jax.device_get(x[:, -1])).reshape(num_seeds, -1).mean(axis=1)

    for chunk in range(num_chunks):
        rtpt.step()
        if config.get("EVAL_DURING_TRAIN", False) and chunk > 0 and chunk % config["EVAL_EVERY"] == 0:
            save_and_eval(carry[0].params, int(carry[-1][0]))

        chunk_start = time.perf_counter()
        carry, (infos, loss, q_mean) = chunk_fn(carry)
        global_step = int(jax.device_get(carry[-1][0]))

        per_seed = {
            "charts/avg_episodic_return": seed_mean(infos["returned_episode_returns"]),
            "charts/avg_episodic_length": seed_mean(infos["returned_episode_lengths"]),
            "losses/td_loss": np.asarray(jax.device_get(loss[:, -1])),
            "losses/q_values": np.asarray(jax.device_get(q_mean[:, -1])),
        }
        metrics = {}
        for name, values in per_seed.items():
            metrics[name] = float(values.mean())
            for seed_idx, value in enumerate(values):
                metrics[f"seed{seed_idx}/{name}"] = float(value)

        elapsed = time.perf_counter() - run_time
        metrics.update({
            "charts/SPS": int(global_step / elapsed),
            "charts/SPS_update": int(steps_per_chunk / (time.perf_counter() - chunk_start)),
            "charts/time": elapsed,
            "charts/global_step": global_step,
        })
        wandb.log(metrics, step=global_step)
        print(
            f"[pqn] chunk {chunk + 1}/{num_chunks} | global_step {global_step} | "
            f"avg_return {metrics['charts/avg_episodic_return']:.2f} | "
            f"td_loss {metrics['losses/td_loss']:.4f} | q_val {metrics['losses/q_values']:.4f} | "
            f"SPS {metrics['charts/SPS']}"
        )

    eval_metrics = save_and_eval(carry[0].params, global_step + 1)
    wandb.finish()
    return eval_metrics
