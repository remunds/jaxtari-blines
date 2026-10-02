# IQN: Dabney, Ostrovski, Silver & Munos (2018), https://arxiv.org/abs/1806.06923
# Network, target and quantile Huber loss follow Dopamine's JAX implementation:
# https://github.com/google/dopamine/blob/master/dopamine/jax/agents/implicit_quantile/implicit_quantile_agent.py
# Training loop, replay buffer and evaluation follow agents/c51/c51.py, adapted from
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_atari_jax.py
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


def cosine_embedding(taus, n_cos):
    """cos(pi * i * tau) for i = 1..n_cos, as in Dopamine: (B, N) -> (B, N, n_cos)."""
    i = jnp.arange(1, n_cos + 1, dtype=jnp.float32)
    return jnp.cos(jnp.pi * taus[..., None] * i)


class IQNCNNNetwork(nn.Module):
    action_dim: int
    n_cos: int = 64

    @nn.compact
    def __call__(self, x, taus):
        # x: (B, 4, 84, 84), taus: (B, N) -> quantile values (B, N, action_dim)
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.astype(jnp.float32)
        x = x / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        psi = x.reshape((x.shape[0], -1))

        phi = nn.relu(nn.Dense(psi.shape[-1])(cosine_embedding(taus, self.n_cos)))
        h = psi[:, None, :] * phi
        h = nn.relu(nn.Dense(512)(h))
        return nn.Dense(self.action_dim)(h)


class IQNMLPNetwork(nn.Module):
    action_dim: int
    n_cos: int = 64

    @nn.compact
    def __call__(self, x, taus):
        # x: (B, obs_dim), taus: (B, N) -> quantile values (B, N, action_dim)
        # JAXtari sizes the 461 layer to match the conv stack's parameter count, so it
        # plays the role of the conv feature extractor psi and 512 that of the FC layer.
        # The tau embedding is therefore merged after 461 and before 512, as in the CNN.
        x = nn.Dense(461, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        psi = nn.relu(x)

        phi = nn.Dense(psi.shape[-1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            cosine_embedding(taus, self.n_cos)
        )
        h = psi[:, None, :] * nn.relu(phi)
        h = nn.relu(nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h))
        return nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(h)


class IQNTrainState(TrainState):
    target_params: flax.core.FrozenDict


@flax.struct.dataclass
class TimeStep:
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    # read every key used more than once up front, so each default lives in one place
    pixel_based = config.get("PIXEL_BASED", True)
    native_downscaling = config.get("NATIVE_DOWNSCALING", True)
    num_envs = config.get("NUM_ENVS", 1)
    train_frequency = config.get("TRAIN_FREQUENCY", 4)
    scan_steps = config.get("SCAN_STEPS", 1000)
    total_timesteps = config.get("TOTAL_TIMESTEPS", 10000000)
    batch_size = config.get("BATCH_SIZE", 32)
    gamma = config.get("GAMMA", 0.99)
    save_path = config.get("SAVE_PATH", "./models")

    # quantile samples: N for the online net, N' for the target, K to pick an action
    n_tau = config.get("N_TAU", 64)
    n_tau_prime = config.get("N_TAU_PRIME", 64)
    n_tau_k = config.get("N_TAU_K", 32)
    n_cos = config.get("N_COS", 64)
    kappa = config.get("KAPPA", 1.0)

    # env steps collected by one full_iqn_step, and by one compiled call of scanned_steps
    steps_per_update = num_envs * train_frequency
    steps_per_iteration = steps_per_update * scan_steps

    run_name = f"{config['ENV_ID']}_{config['EXP_NAME']}_{'pixel' if pixel_based else 'oc'}_{config['SEED']}"

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

    env = make_env(
        config.get("ENV_ID"),
        list(config.get("TRAIN_MODS", [])),
        pixel_based,
        native_downscaling,
        False,
    )()

    action_dim = env.action_space().n
    obs_shape = env.observation_space().shape
    if pixel_based:
        obs_shape = obs_shape[:-1]

    # if -1: we do as many gradient steps as collected samples (stable_baselines3 behavior)
    gradient_steps = config.get("GRADIENT_STEPS", 1)
    if gradient_steps == -1:
        gradient_steps = steps_per_update

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
    Network = IQNCNNNetwork if pixel_based else IQNMLPNetwork
    network = Network(action_dim=action_dim, n_cos=n_cos)
    q_params = network.init(q_key, jnp.zeros((1, *obs_shape)), jnp.zeros((1, n_tau)))

    # Adam eps = 0.01 / batch_size, the same convention as agents/c51/c51.py
    tx = optax.adam(learning_rate=config.get("LEARNING_RATE", 5e-5), eps=0.01 / batch_size)

    agent_state = IQNTrainState.create(
        apply_fn=network.apply,
        params=q_params,
        target_params=jax.tree.map(jnp.copy, q_params),
        tx=tx,
    )

    # uniform sampling: IQN has no prioritised replay
    replay_buffer = fbx.make_flat_buffer(
        max_length=config.get("BUFFER_SIZE", 1000000),
        min_length=config.get("LEARNING_STARTS", 80000),
        sample_batch_size=batch_size,
        add_sequences=False,
        add_batch_size=num_envs,
    )
    replay_buffer = replay_buffer.replace(
        init=jax.jit(replay_buffer.init),
        add=jax.jit(replay_buffer.add, donate_argnums=0),
        sample=jax.jit(replay_buffer.sample),
        can_sample=jax.jit(replay_buffer.can_sample),
    )
    _obs, _state = vmap_reset(jax.random.split(key, num_envs))
    _obs, _state, _reward, _done, _ = vmap_step(_state, jnp.zeros((num_envs,), dtype=jnp.int32))
    _dummy_step = TimeStep(
        obs=_obs[0],
        action=jnp.zeros((), dtype=jnp.int32),
        reward=_reward[0],
        done=_done[0],
    )
    buffer_state = replay_buffer.init(_dummy_step)

    def full_iqn_step(agent_state, buffer_state, env_state, obs, rng, global_step):
        def take_action(carry, _):
            agent_state, buffer_state, env_state, obs, global_step, rng = carry

            rng, action_rng, explore_rng, tau_rng = jax.random.split(rng, 4)
            epsilon = jnp.interp(
                global_step,
                jnp.array([0, config.get("EXPLORATION_FRACTION", 0.10) * total_timesteps]),
                jnp.array([config.get("START_E", 1.0), config.get("END_E", 0.01)]),
            )

            # Q(s, a) is the mean over K sampled quantiles
            taus = jax.random.uniform(tau_rng, (num_envs, n_tau_k))
            q_values = agent_state.apply_fn(agent_state.params, obs, taus).mean(axis=1)
            greedy_actions = q_values.argmax(axis=-1)
            random_actions = jax.random.randint(action_rng, (num_envs,), 0, action_dim)

            explore_mask = jax.random.uniform(explore_rng, (num_envs,)) < epsilon
            actions = jnp.where(explore_mask, random_actions, greedy_actions)

            next_obs, next_env_state, rewards, next_done, info = vmap_step(env_state, actions)

            timestep = TimeStep(
                obs=obs,
                action=actions,
                reward=rewards,
                done=next_done,
            )
            buffer_state = replay_buffer.add(buffer_state, timestep)
            return (agent_state, buffer_state, next_env_state, next_obs, global_step + num_envs, rng), info

        # take TRAIN_FREQUENCY steps in one go
        (agent_state, buffer_state, next_env_state, next_obs, global_step, rng), infos = jax.lax.scan(
            take_action,
            (agent_state, buffer_state, env_state, obs, global_step, rng),
            None,
            length=train_frequency,
        )

        def do_update(update_carry, _):
            u_state, u_key = update_carry
            u_key, sample_key, tau_key, tau_prime_key, tau_k_key = jax.random.split(u_key, 5)

            batch = replay_buffer.sample(buffer_state, sample_key).experience
            b_obs = batch.first.obs
            b_act = batch.first.action
            b_rew = batch.first.reward
            b_don = batch.first.done
            b_nobs = batch.second.obs

            # greedy next action under the target network (no double-Q, as in Dopamine)
            taus_k = jax.random.uniform(tau_k_key, (batch_size, n_tau_k))
            next_q = u_state.apply_fn(u_state.target_params, b_nobs, taus_k).mean(axis=1)
            next_action = jnp.argmax(next_q, axis=-1)

            # target quantiles of that action at N' fresh taus: (B, N')
            taus_prime = jax.random.uniform(tau_prime_key, (batch_size, n_tau_prime))
            next_quantiles = u_state.apply_fn(u_state.target_params, b_nobs, taus_prime)
            next_quantiles = jnp.take_along_axis(next_quantiles, next_action[:, None, None], axis=-1).squeeze(-1)
            target_quantiles = b_rew[:, None] + gamma * (1.0 - b_don[:, None]) * next_quantiles
            target_quantiles = jax.lax.stop_gradient(target_quantiles)

            taus = jax.random.uniform(tau_key, (batch_size, n_tau))

            def q_loss_fn(params):
                quantiles = u_state.apply_fn(params, b_obs, taus)
                pred = jnp.take_along_axis(quantiles, b_act.reshape(-1)[:, None, None], axis=-1).squeeze(-1)

                # quantile Huber loss over all target/prediction pairs: td is (B, N', N)
                td = target_quantiles[:, :, None] - pred[:, None, :]
                abs_td = jnp.abs(td)
                huber = jnp.where(abs_td <= kappa, 0.5 * td ** 2, kappa * (abs_td - 0.5 * kappa))
                weight = jnp.abs(taus[:, None, :] - (td < 0).astype(jnp.float32))
                loss = (weight * huber / kappa).sum(axis=-1).mean(axis=-1).mean()
                return loss, pred.mean()

            (loss, q_val), grads = jax.value_and_grad(q_loss_fn, has_aux=True)(u_state.params)
            new_state = u_state.apply_gradients(grads=grads)

            return (new_state, u_key), (loss, q_val)

        def scanned_update(carry):
            carry, (loss, qval) = jax.lax.scan(do_update, carry, None, length=gradient_steps)
            return carry, (loss[-1], qval[-1])

        # train NN if we have enough samples in the replay buffer (==learning_starts)
        (agent_state, rng), (loss, q_val) = jax.lax.cond(
            replay_buffer.can_sample(buffer_state),
            lambda c: scanned_update(c),
            lambda c: (c, (jnp.array(0.0), jnp.array(0.0))),
            (agent_state, rng),
        )
        update_target_flag = jnp.logical_and(
            replay_buffer.can_sample(buffer_state),
            (global_step % config.get("TARGET_NETWORK_FREQUENCY", 1000)) < steps_per_update
        )
        new_target_params = jax.lax.cond(
            update_target_flag,
            lambda _: optax.incremental_update(agent_state.params, agent_state.target_params, config.get("TAU", 1.0)),
            lambda _: agent_state.target_params,
            None,
        )
        agent_state = agent_state.replace(target_params=new_target_params)

        return (agent_state, buffer_state, next_env_state, next_obs, rng, global_step), (infos, loss, q_val)

    def save_and_eval(step_count, params):
        if save_path is not None:
            model_path = f'{save_path}/{run_name}/{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes([config, params]))
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")

        # evaluate across all mods (and default train env)
        eval_mods = config.get("EVAL_MODS", []) or config.get("TRAIN_MODS", [])
        eval_configs = [([], "default")]
        for mod in eval_mods:
            mods_config = [mod] if not isinstance(mod, (list, tuple)) else list(mod)
            mod_label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods_config)
            eval_configs.append((mods_config, mod_label))

        metrics = {}
        for mods_cfg, mod_label in eval_configs:
            episodic_returns, env_states = evaluate(
                model_path,
                partial(
                    make_env,
                    mods=mods_cfg,
                    pixel_based=pixel_based,
                    native_downscaling=native_downscaling,
                    eval=True,
                ),
                config["ENV_ID"],
                eval_episodes=10,
                Model=Network,
                n_cos=n_cos,
                n_tau_k=n_tau_k,
                seed=config["SEED"] + 42,  # use a different seed for evaluation
                epsilon=0.0,  # JAXtari protocol evaluates greedy policies
            )
            metrics[mod_label] = np.mean(jax.device_get(episodic_returns))
            wandb.log({f"eval/episodic_return_{mod_label}": metrics[mod_label]}, step=step_count)

            if config.get("CAPTURE_VIDEO", True):
                # Long-episode games (enduro, breakout) return tens of thousands of
                # states here. Rendering them all at full resolution on the GPU OOMs
                # against the replay buffer, so cap the length and build the clip on
                # the host. A failed video must never lose the run's eval metrics.
                try:
                    max_frames = config.get("VIDEO_MAX_FRAMES", 1800)
                    clipped_states = jax.tree_util.tree_map(lambda x: x[:max_frames], env_states)
                    cpu = jax.devices("cpu")[0]
                    with jax.default_device(cpu):
                        clipped_states = jax.device_put(clipped_states, cpu)
                        # Instantiate a clean renderer immune to the training env's downscaling
                        clean_renderer = jaxatari.make(config["ENV_ID"], mods=mods_cfg).renderer
                        frames = jax.vmap(clean_renderer.render)(clipped_states)
                        # shape: (N, H, W, C) -> (N, C, H, W)
                        frames = jnp.transpose(frames, (0, 3, 1, 2))
                    video = wandb.Video(np.array(frames), fps=30, format="mp4")
                    wandb.log({f"eval/video_{mod_label}": video}, step=step_count)
                    print(f"Video (eval) logged to wandb with {frames.shape[0]} frames ({mod_label}).")
                except Exception as e:
                    print(f"[WARNING] video capture skipped for {mod_label}: {type(e).__name__}: {e}")
        return metrics

    print("[iqn] start compile...")
    start_compile = time.perf_counter()
    global_step = jnp.array(0, dtype=jnp.int32)
    iqn_carry = (agent_state, buffer_state, _state, _obs, key, global_step)

    def scanned_steps(carry):
        def step_fn(c, _):
            return full_iqn_step(*c)
        return jax.lax.scan(step_fn, carry, None, length=scan_steps)

    # donate the carry so XLA writes the new replay buffer over the old one instead of
    # allocating a second copy; lower/compile AOT so nothing is donated before the loop
    compiled = jax.jit(scanned_steps, donate_argnums=(0,)).lower(iqn_carry).compile()
    end_compile = time.perf_counter()
    print(f"[iqn] compilation time: {end_compile - start_compile:.2f}s")
    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=total_timesteps // steps_per_iteration)
    rtpt.start()
    run_time = time.perf_counter()
    print(f"[iqn] starting training for {total_timesteps} steps...")
    while global_step < total_timesteps:
        rtpt.step()
        iteration = global_step // steps_per_iteration
        if config.get("EVAL_DURING_TRAIN", False) and iteration > 0 and iteration % config.get("EVAL_EVERY", 100) == 0:
            save_and_eval(global_step, iqn_carry[0].params)
        iteration_time_start = time.perf_counter()
        iqn_carry, (infos, loss, q_val) = compiled(iqn_carry)
        global_step = int(iqn_carry[-1])
        now = time.perf_counter()
        sps = int(global_step / (now - run_time))
        sps_update = int(steps_per_iteration / (now - iteration_time_start))
        metrics = {
            "charts/avg_episodic_return": infos["returned_episode_returns"][-1].mean(),
            "charts/avg_episodic_length": infos["returned_episode_lengths"][-1].mean(),
            "losses/td_loss": loss[-1].item(),
            "losses/q_values": q_val[-1].item(),
            "charts/SPS": sps,
            "charts/SPS_update": sps_update,
            "charts/time": now - run_time,
            "charts/global_step": global_step,
        }
        print(f"[iqn] iteration {iteration} | global_step {global_step} | avg_return {metrics['charts/avg_episodic_return']:.2f} | avg_length {metrics['charts/avg_episodic_length']:.2f} | td_loss {metrics['losses/td_loss']:.4f} | q_val {metrics['losses/q_values']:.4f} | SPS {sps} | SPS_update {sps_update}")
        wandb.log(metrics, step=global_step)

    eval_metrics = save_and_eval(global_step + 1, iqn_carry[0].params)
    wandb.finish()
    return eval_metrics
