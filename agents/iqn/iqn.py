import os
import random
import tempfile
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
from flax.linen.initializers import constant, orthogonal
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
from agents.iqn.iqn_eval import evaluate
from rtpt import RTPT

def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False):
    assert mods is None or isinstance(mods, list), "mods must be None or a list of strings"
    if mods is not None and len(mods) == 0:
        mods = None
    if not eval and mods is not None and len(mods) > 0:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        env = jaxatari.make(env_id, mods=mods)
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

class QNetwork(nn.Module):
    action_dim: int
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, x, tau):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.astype(jnp.float32)
        x = x / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        psi = nn.Dense(512)(x)
        psi = nn.relu(psi)
        i = jnp.arange(1, self.embedding_dim + 1, dtype=jnp.float32)
        cos_feat = jnp.cos(jnp.pi * tau[:, :, None] * i[None, None, :])
        phi = nn.Dense(512)(cos_feat)
        phi = nn.relu(phi)
        combined = psi[:, None, :] * phi
        x = nn.Dense(self.action_dim)(combined)
        return x

class MLP_QNetwork(nn.Module):
    action_dim: int
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, x, tau):
        x = nn.Dense(461, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        psi = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        psi = nn.relu(psi)
        i = jnp.arange(1, self.embedding_dim + 1, dtype=jnp.float32)
        cos_feat = jnp.cos(jnp.pi * tau[:, :, None] * i[None, None, :])
        phi = nn.Dense(512)(cos_feat)
        phi = nn.relu(phi)
        combined = psi[:, None, :] * phi
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(combined)
        return x

class IQNTrainState(TrainState):
    target_params: flax.core.FrozenDict

def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    if config.get("PIXEL_BASED", True) and config.get("NUM_ENVS", 1) > 16:
        print("Warning: More than 16 environments may cause OOM on GPU when using pixel-based observations.")

    run_name = f"{config['ENV_ID']}_{config['EXP_NAME']}_{'oc' if not config['PIXEL_BASED'] else 'pixel'}_{config['SEED']}"

    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
    )
    wandb.define_metric("*", step_metric="charts/global_step")

    # do not modify the seeding
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    train_mods = list(config.get("TRAIN_MODS", []))

    env = make_env(
        config.get("ENV_ID"),
        train_mods,
        config.get("PIXEL_BASED", True),
        config.get("NATIVE_DOWNSCALING", True),
        False,
    )()

    action_dim = env.action_space().n
    obs_shape = env.observation_space().shape
    if config.get("PIXEL_BASED", True):
        obs_shape = obs_shape[:-1]

    num_envs = config["NUM_ENVS"]

    @jax.jit
    def vmap_reset(rng):
        obs, state = jax.vmap(env.reset)(rng)
        return obs.reshape(rng.shape[0], *obs_shape), state


    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        next_done = jnp.logical_or(terminated, truncated)
        return next_obs.reshape(action.shape[0], *obs_shape), state, reward, next_done, info


    key, q_key = jax.random.split(key, 2)
    embedding_dim = config.get("QUANTILE_EMBEDDING_DIM", 64)
    network = QNetwork(action_dim=action_dim, embedding_dim=embedding_dim) if config.get("PIXEL_BASED", True) else MLP_QNetwork(action_dim=action_dim, embedding_dim=embedding_dim)

    dummy_obs = jnp.zeros((1, *obs_shape))
    dummy_tau = jnp.zeros((1, config.get("K_TAU_SAMPLES", 32)))
    q_params = network.init(q_key, dummy_obs, dummy_tau)

    tx = optax.adam(learning_rate=config.get("LEARNING_RATE", 0.0001), eps=config.get("ADAM_EPS", 1e-4))

    agent_state = IQNTrainState.create(
        apply_fn=network.apply,
        params=q_params,
        target_params=jax.tree.map(jnp.copy, q_params),
        tx=tx,
    )

    obs_dtype = jnp.uint8 if config.get("PIXEL_BASED", True) else jnp.float32
    replay_buffer = fbx.make_item_buffer(
        max_length=config.get("BUFFER_SIZE", 1000000),
        min_length=config.get("LEARNING_STARTS", 80000),
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

    eval_mods_list = list(config.get("EVAL_MODS", [])) or list(config.get("TRAIN_MODS", []))
    eval_configs = [([], "default")]
    for mod in eval_mods_list:
        mods_cfg = list(mod) if isinstance(mod, (list, tuple)) else [mod]
        mod_label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods_cfg)
        eval_configs.append((mods_cfg, mod_label))

    eval_episodes = 10

    def step_once(carry, _):
        state, buffer_state, env_state, obs, rng, global_step = carry

        rng, action_rng, explore_rng, tau_act_rng = jax.random.split(rng, 4)
        epsilon = jnp.interp(
            global_step,
            jnp.array([0, config.get("EXPLORATION_FRACTION", 0.10) * config.get("TOTAL_TIMESTEPS", 10000000)]),
            jnp.array([config.get("START_E", 1.0), config.get("END_E", 0.05)])
        )

        tau_act = jax.random.uniform(tau_act_rng, (config["NUM_ENVS"], config.get("K_TAU_SAMPLES", 32)))
        q_values = jnp.mean(state.apply_fn(state.params, obs, tau_act), axis=1)
        greedy_actions = q_values.argmax(axis=-1)
        random_actions = jax.random.randint(action_rng, (config["NUM_ENVS"],), 0, action_dim)

        explore_mask = jax.random.uniform(explore_rng, (config["NUM_ENVS"],)) < epsilon
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

        updates_per_step = max(1, config["NUM_ENVS"] // config.get("TRAIN_FREQUENCY", 4))

        B = config.get("BATCH_SIZE", 32)
        N = config.get("N_TAU_SAMPLES", 64)
        N_prime = config.get("N_TAU_PRIME_SAMPLES", 64)
        K = config.get("K_TAU_SAMPLES", 32)
        kappa = config.get("HUBER_KAPPA", 1.0)
        gamma = config.get("GAMMA", 0.99)

        def do_update(update_carry, _):
            u_state, u_key = update_carry
            u_key, sample_key, tau_key, tau_prime_key, tau_act_key = jax.random.split(u_key, 5)

            batch = replay_buffer.sample(buffer_state, sample_key).experience
            b_obs = batch["obs"]
            b_act = batch["action"].reshape(-1)
            b_rew = batch["reward"]
            b_don = batch["done"]
            b_nobs = batch["next_obs"]

            def q_loss_fn(params):
                tau = jax.random.uniform(tau_key, (B, N))

                tau_act = jax.random.uniform(tau_act_key, (B, K))
                next_q  = jnp.mean(
                    u_state.apply_fn(u_state.target_params, b_nobs, tau_act), axis=1
                )
                next_action = jnp.argmax(next_q, axis=-1)

                tau_prime        = jax.random.uniform(tau_prime_key, (B, N_prime))
                target_quantiles = u_state.apply_fn(
                    u_state.target_params, b_nobs, tau_prime
                )
                next_idx         = jnp.broadcast_to(next_action[:, None, None], (B, N_prime, 1))
                target_at_action = jnp.take_along_axis(
                    target_quantiles, next_idx, axis=-1
                ).squeeze(-1)

                T_theta = jax.lax.stop_gradient(
                    b_rew[:, None]
                    + (1.0 - b_don[:, None].astype(jnp.float32)) * gamma * target_at_action
                )

                online_quantiles = u_state.apply_fn(params, b_obs, tau)
                act_idx  = jnp.broadcast_to(b_act[:, None, None], (B, N, 1))
                theta_i  = jnp.take_along_axis(
                    online_quantiles, act_idx, axis=-1
                ).squeeze(-1)

                delta = T_theta[:, None, :] - theta_i[:, :, None]

                abs_delta = jnp.abs(delta)
                huber = jnp.where(
                    abs_delta <= kappa,
                    0.5 * delta ** 2,
                    kappa * (abs_delta - 0.5 * kappa),
                )

                indicator = (delta < 0).astype(jnp.float32)
                rho = jnp.abs(tau[:, :, None] - indicator) * huber / kappa

                loss = jnp.mean(jnp.sum(jnp.mean(rho, axis=-1), axis=-1))

                return loss, theta_i

            (loss, q_val), grads = jax.value_and_grad(q_loss_fn, has_aux=True)(u_state.params)
            new_state = u_state.apply_gradients(grads=grads)

            return (new_state, u_key), (loss, q_val.mean())

        def run_updates(s_state, s_key):
            (new_s_state, new_s_key), (losses, q_vals) = jax.lax.scan(do_update, (s_state, s_key), None, length=updates_per_step)
            return new_s_state, new_s_key, jnp.mean(losses), jnp.mean(q_vals)

        should_train_step = (global_step % config.get("TRAIN_FREQUENCY", 4)) < config["NUM_ENVS"]
        can_train = jnp.logical_and(replay_buffer.can_sample(buffer_state), should_train_step)

        state, rng, avg_loss, avg_q_val = jax.lax.cond(
            can_train,
            lambda c: run_updates(c[0], c[1]),
            lambda c: (c[0], c[1], 0.0, 0.0),
            (state, rng)
        )

        update_target_flag = jnp.logical_and(
            can_train,
            (global_step % config.get("TARGET_NETWORK_FREQUENCY", 1000)) < config["NUM_ENVS"]
        )

        new_target_params = jax.lax.cond(
            update_target_flag,
            lambda _: optax.incremental_update(state.params, state.target_params, config.get("TAU", 1.0)),
            lambda _: state.target_params,
            None
        )
        state = state.replace(target_params=new_target_params)

        global_step += config["NUM_ENVS"]
        return (state, buffer_state, next_env_state, next_obs, rng, global_step), (infos, avg_loss, avg_q_val, epsilon)



    def save_and_eval(step_count):
        agent_state = iqn_carry[0]
        if config.get("SAVE_PATH", "./models") is not None:
            model_path = f'{config.get("SAVE_PATH", "./models")}/{run_name}/{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(
                    flax.serialization.to_bytes(
                        (None, agent_state.params)
                    )
                )
            print(f"model saved to {model_path}")
        else:
            # evaluate() restores the parameters from a checkpoint file, so without a
            # SAVE_PATH write a temporary one for it; it is removed after evaluation.
            fd, model_path = tempfile.mkstemp(suffix=".cleanrl_model")
            with os.fdopen(fd, "wb") as f:
                f.write(flax.serialization.to_bytes((None, agent_state.params)))

        print(f"running evaluation at step {step_count}...")

        metrics = {}
        for mods_cfg, mod_label in eval_configs:
            episodic_returns, env_states_until_done = evaluate(
                model_path,
                partial(
                    make_env,
                    mods=mods_cfg,
                    pixel_based=config.get("PIXEL_BASED", True),
                    native_downscaling=config.get("NATIVE_DOWNSCALING", True),
                    eval=True,
                ),
                config["ENV_ID"],
                eval_episodes=eval_episodes,
                # bind embedding_dim so evaluate() rebuilds the same parameter tree as training
                Model=partial(QNetwork if config.get("PIXEL_BASED", True) else MLP_QNetwork, embedding_dim=embedding_dim),
                epsilon=0.05,
                seed=config["SEED"],
                k_tau_samples=config.get("K_TAU_SAMPLES", 32),
            )

            avg_eval_return = float(jnp.mean(episodic_returns))
            return_key = f"eval/episodic_return_{mod_label}"
            metrics[return_key] = avg_eval_return
            print(f"evaluation at step {step_count} ({mod_label}): average return = {avg_eval_return}")

            wandb.log({return_key: avg_eval_return}, step=step_count)

            if config.get("CAPTURE_VIDEO", False):
                # Instantiate a clean renderer immune to the training env's downscaling
                clean_renderer = jaxatari.make(config["ENV_ID"], mods=mods_cfg).renderer
                frames = jax.vmap(clean_renderer.render)(env_states_until_done)
                # shape: (N, H, W, C) -> (N, C, H, W)
                frames = jnp.transpose(frames, (0, 3, 1, 2))
                video = wandb.Video(np.array(frames), fps=30, format="mp4")
                video_key = f"eval/video_{mod_label}"
                wandb.log({video_key: video}, step=step_count)
                print(f"video (eval) logged to wandb with {frames.shape} frames ({mod_label}).")

        if config.get("SAVE_PATH", "./models") is None:
            os.remove(model_path)
        return metrics

    # we step n_envs each iteration
    print(f"[iqn] start compile...")
    start_compile = time.perf_counter()
    CHUNK_SIZE = config["NUM_STEPS"] // config["NUM_ENVS"]

    key, reset_key = jax.random.split(key)
    obs, env_state = vmap_reset(jax.random.split(reset_key, num_envs))
    global_step = jnp.array(0, dtype=jnp.int32)
    iqn_carry = (agent_state, buffer_state, env_state, obs, key, global_step)

    @partial(jax.jit, donate_argnums=(0,))
    def scanned_steps(carry):
        return jax.lax.scan(step_once, carry, None, length=CHUNK_SIZE)

    # warmup to trigger compilation
    # scanned_steps donates its carry, so we compile ahead-of-time instead of calling it:
    # an actual warmup call would delete the carry buffers we still need for the run.
    _ = scanned_steps.lower(iqn_carry).compile()
    end_compile = time.perf_counter()
    print(f"[iqn] compilation time: {end_compile - start_compile:.2f}s")
    # step_once advances ONE env-step (global_step += NUM_ENVS), unlike the DQN reference's
    # step fn which scans TRAIN_FREQUENCY of them -- hence no TRAIN_FREQUENCY factor here.
    steps_per_iteration = num_envs * CHUNK_SIZE
    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=config.get("TOTAL_TIMESTEPS") // steps_per_iteration)
    rtpt.start()
    run_time = time.perf_counter()
    print(f"[iqn] starting training for {config.get('TOTAL_TIMESTEPS')} steps...")
    while global_step < config.get("TOTAL_TIMESTEPS"):
        rtpt.step()
        iteration = global_step // steps_per_iteration
        if config["EVAL_DURING_TRAIN"] and iteration > 0 and iteration % config["EVAL_EVERY"] == 0:
            save_and_eval(global_step)
        iteration_time_start = time.perf_counter()
        result = scanned_steps(iqn_carry)
        iqn_carry, (infos, losses, q_vals, epsilons) = result
        global_step = int(iqn_carry[5])
        td_loss = float(jnp.sum(losses) / jnp.maximum(jnp.sum(losses != 0), 1))
        q_val = float(jnp.sum(q_vals) / jnp.maximum(jnp.sum(q_vals != 0), 1))
        print(f"[iqn] iteration {iteration} | global_step {global_step} | avg_return {infos['returned_episode_returns'][-1].mean():.2f} | avg_length {infos['returned_episode_lengths'][-1].mean():.2f} | td_loss {td_loss:.4f} | q_val {q_val:.4f} | SPS {int(global_step / (time.perf_counter() - run_time))} | SPS_update {int(steps_per_iteration / (time.perf_counter() - iteration_time_start))}")
        metrics = {
            "charts/avg_episodic_return": infos["returned_episode_returns"][-1].mean(),
            "charts/avg_episodic_length": infos["returned_episode_lengths"][-1].mean(),
            "losses/td_loss": td_loss,
            "losses/q_values": q_val,
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(steps_per_iteration / (time.perf_counter() - iteration_time_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/epsilon": epsilons[-1].item(),
            "charts/global_step": global_step,
        }
        wandb.log(metrics, step=global_step)

    eval_metrics = save_and_eval(global_step+1)
    wandb.finish()
    return eval_metrics