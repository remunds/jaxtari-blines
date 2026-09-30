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
from agents.rainbow.rainbow_eval import evaluate
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


class NoisyDense(nn.Module):
    features: int
    sigma0: float = 0.5

    @nn.compact
    def __call__(self, x, deterministic=False):
        in_features = x.shape[-1]
        bound = 1.0 / np.sqrt(in_features)
        sigma_init = self.sigma0 / np.sqrt(in_features)

        def mu_init(key, shape, dtype=jnp.float32):
            return jax.random.uniform(key, shape, dtype, -bound, bound)

        kernel_mu = self.param("kernel_mu", mu_init, (in_features, self.features))
        kernel_sigma = self.param(
            "kernel_sigma", nn.initializers.constant(sigma_init), (in_features, self.features)
        )
        bias_mu = self.param("bias_mu", mu_init, (self.features,))
        bias_sigma = self.param(
            "bias_sigma", nn.initializers.constant(sigma_init), (self.features,)
        )

        if deterministic:
            return x @ kernel_mu + bias_mu

        key = self.make_rng("noise")
        k_in, k_out = jax.random.split(key)
        f = lambda v: jnp.sign(v) * jnp.sqrt(jnp.abs(v))
        eps_in = f(jax.random.normal(k_in, (in_features,)))
        eps_out = f(jax.random.normal(k_out, (self.features,)))
        kernel = kernel_mu + kernel_sigma * jnp.outer(eps_in, eps_out)
        bias = bias_mu + bias_sigma * eps_out
        return x @ kernel + bias


class ImpalaResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
        return x + inputs


class ImpalaConvSequence(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ImpalaResidualBlock(self.channels)(x)
        x = ImpalaResidualBlock(self.channels)(x)
        return x


class ImpalaCNN(nn.Module):
    channels: tuple = (16, 32, 32)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))          
        x = x.astype(jnp.float32) / 255.0
        for ch in self.channels:
            x = ImpalaConvSequence(ch)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        return x

    
class RainbowCNNNetwork(nn.Module):
    action_dim: int
    n_atoms: int
    sigma0: float = 0.5
    cnn_type: str = "nature"

    @nn.compact
    def __call__(self, x, deterministic=False):
        if self.cnn_type == "impala":
            x = ImpalaCNN()(x)
        else:
            x = jnp.transpose(x, (0, 2, 3, 1))
            x = x.astype(jnp.float32) / 255.0
            x = nn.relu(nn.Conv(32, (8, 8), (4, 4), padding="VALID")(x))
            x = nn.relu(nn.Conv(64, (4, 4), (2, 2), padding="VALID")(x))
            x = nn.relu(nn.Conv(64, (3, 3), (1, 1), padding="VALID")(x))
            x = x.reshape((x.shape[0], -1))

        v = nn.relu(NoisyDense(512, self.sigma0)(x, deterministic))
        v = NoisyDense(self.n_atoms, self.sigma0)(v, deterministic)

        a = nn.relu(NoisyDense(512, self.sigma0)(x, deterministic))
        a = NoisyDense(self.action_dim * self.n_atoms, self.sigma0)(a, deterministic)
        a = a.reshape((a.shape[0], self.action_dim, self.n_atoms))

        logits = v[:, None, :] + a - jnp.mean(a, axis=1, keepdims=True)
        return jax.nn.softmax(logits, axis=-1)


class RainbowMLPNetwork(nn.Module):
    action_dim: int
    n_atoms: int
    sigma0: float = 0.5

    @nn.compact
    def __call__(self, x, deterministic=False):
        x = x.astype(jnp.float32)
        x = nn.relu(NoisyDense(461, self.sigma0)(x, deterministic))

        v = nn.relu(NoisyDense(512, self.sigma0)(x, deterministic))
        v = NoisyDense(self.n_atoms, self.sigma0)(v, deterministic)

        a = nn.relu(NoisyDense(512, self.sigma0)(x, deterministic))
        a = NoisyDense(self.action_dim * self.n_atoms, self.sigma0)(a, deterministic)
        a = a.reshape((a.shape[0], self.action_dim, self.n_atoms))

        logits = v[:, None, :] + a - jnp.mean(a, axis=1, keepdims=True)
        return jax.nn.softmax(logits, axis=-1)


class RainbowTrainState(TrainState):
    target_params: flax.core.FrozenDict
    atoms: jnp.ndarray


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
    train_label = "default" if not train_mods else "_".join(str(m) for m in train_mods)

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
    # -1: we do as many gradient steps as collected samples (stable_baselines3 behavior)
    gradient_steps = num_envs * config.get("TRAIN_FREQUENCY", 4) if config.get("GRADIENT_STEPS", 1) == -1 else config.get("GRADIENT_STEPS", 1)

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

    n_step = config.get("N_STEP", 3)
    gamma = config.get("GAMMA", 0.99)
    gamma_n = gamma ** n_step
    batch_size = config.get("BATCH_SIZE", 32)
    beta_start = config.get("IS_BETA_START", 0.4)
    beta_end = config.get("IS_BETA_END", 1.0)
    total_timesteps = config.get("TOTAL_TIMESTEPS", 10000000)
    sigma0 = config.get("NOISY_SIGMA0", 0.5)

    key, q_key, noise_key = jax.random.split(key, 3)
    network = RainbowCNNNetwork(action_dim=action_dim, n_atoms=n_atoms, sigma0=sigma0, cnn_type=config.get("CNN_TYPE", "nature")) if config.get("PIXEL_BASED", True) else RainbowMLPNetwork(action_dim=action_dim, n_atoms=n_atoms, sigma0=sigma0)

    dummy_obs = jnp.zeros((1, *obs_shape))
    q_params = network.init({"params": q_key, "noise": noise_key}, dummy_obs, False)

    tx = optax.adam(learning_rate=config.get("LEARNING_RATE", 0.0000625), eps=config.get("ADAM_EPS", 0.00015))

    agent_state = RainbowTrainState.create(
        apply_fn=network.apply,
        params=q_params,
        target_params=jax.tree.map(jnp.copy, q_params),
        atoms=atoms,
        tx=tx,
    )

    obs_dtype = jnp.uint8 if config.get("PIXEL_BASED", True) else jnp.float32
    replay_buffer = fbx.make_prioritised_item_buffer(
        max_length=config.get("BUFFER_SIZE", 1000000),
        min_length=config.get("LEARNING_STARTS", 80000),
        sample_batch_size=batch_size,
        add_batches=True,
        priority_exponent=config.get("PRIORITY_EXPONENT", 0.5),
        device="gpu" if jax.default_backend() == "gpu" else "cpu",
    )
    example_transition = {
        "obs": jnp.zeros(obs_shape, dtype=obs_dtype),
        "action": jnp.zeros((), dtype=jnp.int32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "done": jnp.zeros((), dtype=jnp.bool_),
        "next_obs": jnp.zeros(obs_shape, dtype=obs_dtype),
    }
    buffer_state = replay_buffer.init(example_transition)

    _obs, _state = vmap_reset(jax.random.split(key, num_envs))
    window = (
        jnp.zeros((n_step, num_envs, *obs_shape), dtype=obs_dtype),
        jnp.zeros((n_step, num_envs), dtype=jnp.int32),
        jnp.zeros((n_step, num_envs), dtype=jnp.float32),
        jnp.zeros((n_step, num_envs), dtype=jnp.float32),
    )

    def full_rainbow_step(agent_state, buffer_state, env_state, obs, window, rng, global_step):
        def take_action(carry, _):
            agent_state, buffer_state, env_state, obs, window, global_step, rng = carry

            # noisy nets instead eps greedy
            rng, act_noise_rng = jax.random.split(rng)
            pmfs = agent_state.apply_fn(agent_state.params, obs, False, rngs={"noise": act_noise_rng})
            q_values = (pmfs * agent_state.atoms[None, None, :]).sum(-1)
            actions = q_values.argmax(axis=-1)

            next_obs, next_env_state, rewards, next_done, info = vmap_step(env_state, actions)

            # rolling n-step window 
            w_obs, w_act, w_rew, w_done = window
            w_obs = jnp.concatenate([w_obs[1:], obs.astype(obs_dtype)[None]], axis=0)
            w_act = jnp.concatenate([w_act[1:], actions.astype(jnp.int32)[None]], axis=0)
            w_rew = jnp.concatenate([w_rew[1:], rewards.astype(jnp.float32)[None]], axis=0)
            w_done = jnp.concatenate([w_done[1:], next_done.astype(jnp.float32)[None]], axis=0)
            window = (w_obs, w_act, w_rew, w_done)

            # truncated n-step return
            not_done = 1.0 - w_done
            live = jnp.cumprod(not_done, axis=0)
            reward_mask = jnp.concatenate([jnp.ones((1, num_envs)), live[:-1]], axis=0)
            discounts = (gamma ** jnp.arange(n_step, dtype=jnp.float32))[:, None]
            n_step_return = jnp.sum(w_rew * reward_mask * discounts, axis=0)
            n_step_done = 1.0 - live[-1]

            transition = {
                "obs": w_obs[0],
                "action": w_act[0],
                "reward": n_step_return,
                "done": n_step_done.astype(jnp.bool_),
                "next_obs": next_obs.astype(obs_dtype),
            }
            window_full = global_step >= (n_step - 1) * num_envs
            buffer_state = jax.lax.cond(
                window_full,
                lambda bs: replay_buffer.add(bs, transition),
                lambda bs: bs,
                buffer_state,
            )
            return (agent_state, buffer_state, next_env_state, next_obs, window, global_step + num_envs, rng), info

        (agent_state, buffer_state, next_env_state, next_obs, window, global_step, rng), infos = jax.lax.scan(
            take_action,
            (agent_state, buffer_state, env_state, obs, window, global_step, rng),
            None,
            length=config.get("TRAIN_FREQUENCY", 4),
        )

        beta = jnp.interp(
            global_step,
            jnp.array([0, total_timesteps]),
            jnp.array([beta_start, beta_end]),
        )

        def do_update(update_carry, _):
            u_state, u_buffer_state, u_key = update_carry
            u_key, sample_key, select_noise, target_noise, online_noise = jax.random.split(u_key, 5)

            sampled = replay_buffer.sample(u_buffer_state, sample_key)
            batch = sampled.experience
            b_obs = batch["obs"]
            b_act = batch["action"].reshape(-1)
            b_rew = batch["reward"]
            b_don = batch["done"].astype(jnp.float32)
            b_nobs = batch["next_obs"]

            # importance sampling weights
            is_weights = (1.0 / (sampled.probabilities + 1e-10)) ** beta
            is_weights = is_weights / jnp.max(is_weights)

            # double DQN
            next_pmfs_online = u_state.apply_fn(u_state.params, b_nobs, False, rngs={"noise": select_noise})
            next_q_online = (next_pmfs_online * u_state.atoms[None, None, :]).sum(-1)
            next_action = jnp.argmax(next_q_online, axis=-1)

            next_pmfs = u_state.apply_fn(u_state.target_params, b_nobs, False, rngs={"noise": target_noise})
            next_pmfs = next_pmfs[jnp.arange(batch_size), next_action]

            # categorical projection
            next_atoms = b_rew[:, None] + gamma_n * u_state.atoms[None, :] * (1.0 - b_don[:, None])
            tz = jnp.clip(next_atoms, v_min, v_max)
            b = (tz - v_min) / delta_z
            l = jnp.clip(jnp.floor(b).astype(jnp.int32), 0, n_atoms - 1)
            u = jnp.clip(jnp.ceil(b).astype(jnp.int32), 0, n_atoms - 1)
            d_l = (u.astype(jnp.float32) + (l == u).astype(jnp.float32) - b) * next_pmfs
            d_u = (b - l.astype(jnp.float32)) * next_pmfs

            target_pmfs = jnp.zeros((batch_size, n_atoms))

            def project_sample(i, val):
                val = val.at[i, l[i]].add(d_l[i])
                val = val.at[i, u[i]].add(d_u[i])
                return val

            target_pmfs = jax.lax.fori_loop(0, batch_size, project_sample, target_pmfs)
            target_pmfs = jax.lax.stop_gradient(target_pmfs)

            def loss_fn(params):
                pmfs = u_state.apply_fn(params, b_obs, False, rngs={"noise": online_noise})
                p = pmfs[jnp.arange(batch_size), b_act]
                p = jnp.clip(p, 1e-5, 1.0)
                cross_entropy = -(target_pmfs * jnp.log(p)).sum(-1)
                loss = jnp.mean(is_weights * cross_entropy)
                q_val = (p * u_state.atoms[None, :]).sum(-1).mean()
                return loss, (cross_entropy, q_val)

            (loss, (cross_entropy, q_val)), grads = jax.value_and_grad(loss_fn, has_aux=True)(u_state.params)
            new_state = u_state.apply_gradients(grads=grads)

            target_entropy_term = jnp.sum(target_pmfs * jnp.log(jnp.clip(target_pmfs, 1e-5, 1.0)), axis=-1)
            kl = jnp.clip(target_entropy_term + cross_entropy, 1e-6, None)
            new_buffer_state = replay_buffer.set_priorities(u_buffer_state, sampled.indices, kl)

            return (new_state, new_buffer_state, u_key), (loss, q_val)

        def scanned_update(carry):
            carry, (loss, qval) = jax.lax.scan(do_update, carry, None, length=gradient_steps)
            return carry, (loss[-1], qval[-1])

        (agent_state, buffer_state, rng), (loss, q_val) = jax.lax.cond(
            replay_buffer.can_sample(buffer_state),
            lambda c: scanned_update(c),
            lambda c: (c, (jnp.array(0.0), jnp.array(0.0))),
            (agent_state, buffer_state, rng),
        )

        steps_per_update = config.get("TRAIN_FREQUENCY", 4) * config.get("NUM_ENVS", 1)
        update_target_flag = jnp.logical_and(
            replay_buffer.can_sample(buffer_state),
            (global_step % config.get("TARGET_NETWORK_FREQUENCY", 8000)) < steps_per_update
        )
        new_target_params = jax.lax.cond(
            update_target_flag,
            lambda _: optax.incremental_update(agent_state.params, agent_state.target_params, config.get("TAU", 1.0)),
            lambda _: agent_state.target_params,
            None,
        )
        agent_state = agent_state.replace(target_params=new_target_params)

        return (agent_state, buffer_state, next_env_state, next_obs, window, rng, global_step), (infos, loss, q_val, beta)

    def save_and_eval(step_count):
        if config.get("SAVE_PATH", "./models") is not None:
            model_path = f'{config.get("SAVE_PATH", "./models")}/{run_name}/{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes([config, rainbow_carry[0].params]))
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")

        eval_mods = config["EVAL_MODS"] if len(config["EVAL_MODS"]) > 0 else config["TRAIN_MODS"]
        eval_configs = [([], "default")]
        if len(eval_mods) > 0:
            mods_list = list(eval_mods)
            for mod in mods_list:
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
                    pixel_based=config["PIXEL_BASED"],
                    native_downscaling=config["NATIVE_DOWNSCALING"],
                    eval=True,
                ),
                config["ENV_ID"],
                eval_episodes=10,
                Model=partial(RainbowCNNNetwork, cnn_type=config.get("CNN_TYPE", "nature")) if config.get("PIXEL_BASED", True) else RainbowMLPNetwork,
                n_atoms=n_atoms,
                v_min=v_min,
                v_max=v_max,
                seed=config["SEED"] + 42,  # different seed for evaluation
            )
            metrics[mod_label] = np.mean(jax.device_get(episodic_returns))
            wandb.log({f"eval/episodic_return_{mod_label}": np.mean(jax.device_get(episodic_returns))}, step=step_count)

            if config["CAPTURE_VIDEO"]:
                clean_renderer = jaxatari.make(config["ENV_ID"], mods=mods_cfg).renderer
                frames = jax.vmap(clean_renderer.render)(env_states)
                frames = jnp.transpose(frames, (0, 3, 1, 2))
                video = wandb.Video(np.array(frames), fps=30, format="mp4")
                wandb.log({f"eval/video_{mod_label}": video}, step=step_count)
                print(f"Video (eval) logged to wandb with {frames.shape[0]} frames ({mod_label}).")
        return metrics

    print(f"[rainbow] start compile...")
    start_compile = time.perf_counter()
    global_step = jnp.array(0, dtype=jnp.int32)
    rainbow_carry = (agent_state, buffer_state, _state, _obs, window, key, global_step)

    @jax.jit
    def scanned_steps(carry):
        def step_fn(c, _):
            return full_rainbow_step(*c)
        return jax.lax.scan(step_fn, carry, None, length=config.get("SCAN_STEPS", 1000))

    # compilation trigger warmup
    _ = jax.block_until_ready(scanned_steps(rainbow_carry))
    end_compile = time.perf_counter()
    print(f"[rainbow] compilation time: {end_compile - start_compile:.2f}s")
    steps_per_iteration = config.get("NUM_ENVS") * config.get("TRAIN_FREQUENCY") * config.get("SCAN_STEPS")
    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=config.get("TOTAL_TIMESTEPS") // steps_per_iteration)
    rtpt.start()
    run_time = time.perf_counter()
    print(f"[rainbow] starting training for {config.get('TOTAL_TIMESTEPS')} steps...")
    while global_step < config.get("TOTAL_TIMESTEPS"):
        rtpt.step()
        iteration = global_step // steps_per_iteration
        if config["EVAL_DURING_TRAIN"] and iteration > 0 and iteration % config["EVAL_EVERY"] == 0:
            save_and_eval(global_step)
        iteration_time_start = time.perf_counter()
        rainbow_carry, (infos, loss, q_val, beta) = scanned_steps(rainbow_carry)
        global_step = int(rainbow_carry[-1])
        print(f"[rainbow] iteration {iteration} | global_step {global_step} | avg_return {infos['returned_episode_returns'][-1].mean():.2f} | avg_length {infos['returned_episode_lengths'][-1].mean():.2f} | td_loss {loss[-1]:.4f} | q_val {q_val[-1]:.4f} | SPS {int(global_step / (time.perf_counter() - run_time))} | SPS_update {int(steps_per_iteration / (time.perf_counter() - iteration_time_start))}")
        metrics = {
            "charts/avg_episodic_return": infos["returned_episode_returns"][-1].mean(),
            "charts/avg_episodic_length": infos["returned_episode_lengths"][-1].mean(),
            "losses/td_loss": loss[-1].item(),
            "losses/q_values": q_val[-1].item(),
            "charts/is_beta": beta[-1].item(),
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(steps_per_iteration / (time.perf_counter() - iteration_time_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }
        wandb.log(metrics, step=global_step)

    eval_metrics = save_and_eval(global_step + 1)
    wandb.finish()
    return eval_metrics