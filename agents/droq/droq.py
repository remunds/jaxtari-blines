# Adapted from https://github.com/TakuyaHiraoka/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning
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
from agents.droq.droq_eval import evaluate
from agents.run_utils import eval_mod_configs, plan_chunks


def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False):
    assert mods is None or isinstance(mods, list), "mods must be None or a list of strings"
    if mods is not None and len(mods) == 0:
        mods = None
    if not eval and mods is not None and len(mods) > 0:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        env = jaxtari.make(env_id, mods=mods)
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


class Actor(nn.Module):
    action_dim: int

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
        x = nn.Dense(self.action_dim)(x)
        return x


class MLPActor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return x


class DroQCritic(nn.Module):
    action_dim: int
    dropout_rate: float = 0.01

    @nn.compact
    def __call__(self, x, training: bool = True):
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
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = nn.Dense(512)(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class MLPDroQCritic(nn.Module):
    action_dim: int
    dropout_rate: float = 0.01

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return x


class CriticTrainState(TrainState):
    target_params: flax.core.FrozenDict


@flax.struct.dataclass
class TimeStep:
    obs: jnp.array
    action: jnp.array
    reward: jnp.array
    done: jnp.array


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    pixel_based = config["PIXEL_BASED"]
    env_id = config["ENV_ID"]
    num_envs = config["NUM_ENVS"]

    if pixel_based and num_envs > 16:
        print("Warning: More than 16 environments may cause OOM on GPU when using pixel-based observations.")

    run_name = f"{env_id}_{config['EXP_NAME']}_{'oc' if not pixel_based else 'pixel'}_{config['SEED']}"

    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
        mode=config.get("WANDB_MODE", "online"),
    )
    wandb.define_metric("*", step_metric="charts/global_step")

    # same seeding order as dqn/pqn: python/numpy seeds, then key -> network keys -> env resets
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    env = make_env(
        env_id,
        list(config.get("TRAIN_MODS") or []),
        pixel_based,
        config.get("NATIVE_DOWNSCALING", True),
        False,
    )()

    action_dim = env.action_space().n
    obs_shape = env.observation_space().shape
    if pixel_based:
        obs_shape = obs_shape[:-1]  # drop the trailing grayscale channel

    # if -1: we do as many gradient steps as collected samples (stable_baselines3 behavior)
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

    gamma = config.get("GAMMA", 0.99)
    tau = config.get("TAU", 0.005)
    batch_size = config.get("BATCH_SIZE", 256)
    M = config.get("NUM_CRITICS", 2)
    dropout_rate = config.get("DROPOUT_RATE", 0.01)
    target_entropy_scale = config.get("TARGET_ENTROPY_SCALE", 0.98)
    target_entropy = target_entropy_scale * jnp.log(action_dim)
    initial_log_alpha = jnp.array(np.log(config.get("INIT_ALPHA", 1.0)), dtype=jnp.float32)

    # one scan iteration advances every env by TRAIN_FREQUENCY steps
    scan_steps, num_chunks, steps_per_chunk = plan_chunks(
        config["TOTAL_TIMESTEPS"],
        num_envs * config.get("TRAIN_FREQUENCY", 4),
        config.get("SCAN_STEPS", 1000),
    )

    actor_net = Actor(action_dim=action_dim) if pixel_based else MLPActor(action_dim=action_dim)
    critic_net = (
        DroQCritic(action_dim=action_dim, dropout_rate=dropout_rate)
        if pixel_based
        else MLPDroQCritic(action_dim=action_dim, dropout_rate=dropout_rate)
    )

    dummy_obs = jnp.zeros((1, *obs_shape))
    # same root split as dqn/c51/pqn, so env resets for a given seed match across agents
    key, init_key = jax.random.split(key, 2)
    actor_init_key, critic_init_key = jax.random.split(init_key)
    actor_params = actor_net.init(actor_init_key, dummy_obs)

    # ensemble of M critics: vmap over the ensemble axis (not over seeds)
    critic_init_keys = jax.random.split(critic_init_key, M)
    critic_params_stacked = jax.vmap(
        lambda k: critic_net.init({"params": k, "dropout": k}, dummy_obs, training=False)
    )(critic_init_keys)
    target_critic_params_stacked = jax.tree.map(jnp.copy, critic_params_stacked)

    actor_tx = optax.adam(learning_rate=config.get("ACTOR_LR", 3e-4))
    critic_tx = optax.adam(learning_rate=config.get("CRITIC_LR", 3e-4))
    alpha_tx = optax.adam(learning_rate=config.get("ALPHA_LR", 3e-4))

    actor_state = TrainState.create(apply_fn=actor_net.apply, params=actor_params, tx=actor_tx)
    critic_state = CriticTrainState.create(
        apply_fn=critic_net.apply,
        params=critic_params_stacked,
        target_params=target_critic_params_stacked,
        tx=critic_tx,
    )
    alpha_params = {"log_alpha": initial_log_alpha}
    alpha_state = TrainState.create(apply_fn=lambda p, x: p["log_alpha"], params=alpha_params, tx=alpha_tx)

    replay_buffer = fbx.make_flat_buffer(
        max_length=config.get("BUFFER_SIZE", 1000000),
        min_length=config.get("LEARNING_STARTS", 20000),
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
    _obs, _state, _reward, _done, _info = vmap_step(_state, jnp.zeros((num_envs,), dtype=jnp.int32))
    _dummy_step = TimeStep(
        obs=_obs[0],
        action=jnp.zeros((), dtype=jnp.int32),
        reward=_reward[0],
        done=_done[0],
    )
    buffer_state = replay_buffer.init(_dummy_step)

    def critic_apply_all(params_stacked, obs, dropout_keys_stacked):
        return jax.vmap(
            lambda p, k: critic_net.apply(p, obs, training=True, rngs={"dropout": k}),
            in_axes=(0, 0),
        )(params_stacked, dropout_keys_stacked)

    def full_droq_step(actor_state, critic_state, alpha_state, buffer_state, env_state, obs, rng, global_step):
        def take_action(carry, _):
            actor_state, critic_state, alpha_state, buffer_state, env_state, obs, global_step, rng = carry
            rng, act_key = jax.random.split(rng)
            logits = actor_state.apply_fn(actor_state.params, obs)
            actions = jax.random.categorical(act_key, logits, axis=-1)

            next_obs, next_env_state, rewards, next_done, info = vmap_step(env_state, actions)

            timestep = TimeStep(
                obs=obs,
                action=actions,
                reward=rewards,
                done=next_done,
            )
            buffer_state = replay_buffer.add(buffer_state, timestep)
            return (actor_state, critic_state, alpha_state, buffer_state, next_env_state, next_obs, global_step + num_envs, rng), info

        (actor_state, critic_state, alpha_state, buffer_state, next_env_state, next_obs, global_step, rng), infos = jax.lax.scan(
            take_action,
            (actor_state, critic_state, alpha_state, buffer_state, env_state, obs, global_step, rng),
            None,
            length=config.get("TRAIN_FREQUENCY", 4),
        )

        def do_update(update_carry, _):
            actor_state, critic_state, alpha_state, u_key = update_carry
            u_key, sample_key, tgt_drop_key, cri_drop_key, act_drop_key = jax.random.split(u_key, 5)

            batch = replay_buffer.sample(buffer_state, sample_key).experience
            b_obs = batch.first.obs
            b_act = batch.first.action
            b_rew = batch.first.reward
            b_don = batch.first.done
            b_nobs = batch.second.obs

            alpha_val = jnp.exp(alpha_state.params["log_alpha"])

            tgt_drop_keys = jax.random.split(tgt_drop_key, M)
            target_q_all = critic_apply_all(critic_state.target_params, b_nobs, tgt_drop_keys)
            target_q_min = target_q_all.min(axis=0)

            next_logits = actor_state.apply_fn(actor_state.params, b_nobs)
            next_log_probs = jax.nn.log_softmax(next_logits, axis=-1)
            next_probs = jnp.exp(next_log_probs)
            V_next = (next_probs * (target_q_min - alpha_val * next_log_probs)).sum(axis=-1)
            y = jax.lax.stop_gradient(b_rew + gamma * (1.0 - b_don) * V_next)

            def critic_loss_fn(critic_params):
                cri_drop_keys = jax.random.split(cri_drop_key, M)
                q_all = critic_apply_all(critic_params, b_obs, cri_drop_keys)
                q_selected = q_all[:, jnp.arange(batch_size), b_act.reshape(-1)]
                loss = ((q_selected - y[None, :]) ** 2).mean()
                return loss, q_selected.mean()

            (c_loss, q_mean), c_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_state.params)
            critic_state = critic_state.apply_gradients(grads=c_grads)

            def actor_loss_fn(actor_params):
                logits = actor_state.apply_fn(actor_params, b_obs)
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                probs = jnp.exp(log_probs)
                act_drop_keys = jax.random.split(act_drop_key, M)
                q_all_a = critic_apply_all(critic_state.params, b_obs, act_drop_keys)
                q_min_a = jax.lax.stop_gradient(q_all_a.min(axis=0))  # (batch, |A|)
                loss = (probs * (alpha_val * log_probs - q_min_a)).sum(axis=-1).mean()
                entropy = -(probs * log_probs).sum(axis=-1).mean()
                return loss, entropy

            (a_loss, entropy), a_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
            actor_state = actor_state.apply_gradients(grads=a_grads)

            def alpha_loss_fn(alpha_params):
                log_alpha = alpha_params["log_alpha"]
                return log_alpha * jax.lax.stop_gradient(entropy - target_entropy)

            al_grads = jax.grad(alpha_loss_fn)(alpha_state.params)
            alpha_state = alpha_state.apply_gradients(grads=al_grads)

            new_target = optax.incremental_update(critic_state.params, critic_state.target_params, tau)
            critic_state = critic_state.replace(target_params=new_target)

            return (actor_state, critic_state, alpha_state, u_key), (c_loss, a_loss, alpha_val, entropy, q_mean)

        def scanned_updates(carry):
            carry, (c_loss, a_loss, alpha_v, ent, q_m) = jax.lax.scan(do_update, carry, None, length=gradient_steps)
            return carry, (c_loss[-1], a_loss[-1], alpha_v[-1], ent[-1], q_m[-1])

        # update only once the replay buffer holds LEARNING_STARTS samples
        (actor_state, critic_state, alpha_state, rng), (c_loss, a_loss, alpha_v, ent, q_m) = jax.lax.cond(
            replay_buffer.can_sample(buffer_state),
            lambda c: scanned_updates(c),
            lambda c: (c, (jnp.array(0.0), jnp.array(0.0), jnp.exp(initial_log_alpha), jnp.array(0.0), jnp.array(0.0))),
            (actor_state, critic_state, alpha_state, rng),
        )

        return (actor_state, critic_state, alpha_state, buffer_state, next_env_state, next_obs, rng, global_step), (infos, c_loss, a_loss, alpha_v, ent, q_m)

    def save_and_eval(params, step_count):
        if config.get("SAVE_PATH") is not None:
            save_dir = os.path.join(config["SAVE_PATH"], run_name)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model')
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes([config, params]))
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")
        metrics = {}
        for mods_cfg, mod_label in eval_mod_configs(config):
            episodic_returns, env_states = evaluate(
                params,
                partial(
                    make_env,
                    mods=mods_cfg,
                    pixel_based=pixel_based,
                    native_downscaling=config.get("NATIVE_DOWNSCALING", True),
                    eval=True,
                ),
                env_id,
                eval_episodes=10,
                Model=Actor if pixel_based else MLPActor,
                seed=config["SEED"] + 42,  # use a different seed for evaluation
            )
            metrics[mod_label] = float(np.mean(jax.device_get(episodic_returns)))
            wandb.log({f"eval/episodic_return_{mod_label}": metrics[mod_label]}, step=step_count)

            if config.get("CAPTURE_VIDEO", False):
                # Instantiate a clean renderer immune to the training env's downscaling
                clean_renderer = jaxtari.make(env_id, mods=mods_cfg).renderer
                frames = jnp.transpose(jax.vmap(clean_renderer.render)(env_states), (0, 3, 1, 2))
                video = wandb.Video(np.array(frames), fps=30, format="mp4")
                wandb.log({f"eval/video_{mod_label}": video}, step=step_count)
                print(f"Video (eval) logged to wandb with {frames.shape[0]} frames ({mod_label}).")
        return metrics

    global_step = jnp.array(0, dtype=jnp.int32)
    carry = (actor_state, critic_state, alpha_state, buffer_state, _state, _obs, key, global_step)

    def scanned_steps(carry):
        def step_fn(c, _):
            return full_droq_step(*c)
        return jax.lax.scan(step_fn, carry, None, length=scan_steps)

    print("[droq] start compile...")
    start_compile = time.perf_counter()
    # donate the carry so XLA reuses the replay buffer's memory; lower/compile AOT so nothing is donated before the loop
    compiled = jax.jit(scanned_steps, donate_argnums=(0,)).lower(carry).compile()
    print(f"[droq] compilation time: {time.perf_counter() - start_compile:.2f}s")

    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=num_chunks)
    rtpt.start()
    run_time = time.perf_counter()
    print(f"[droq] starting training: {num_chunks} chunks x {steps_per_chunk} env steps")
    for chunk in range(num_chunks):
        rtpt.step()
        if config["EVAL_DURING_TRAIN"] and chunk > 0 and chunk % config["EVAL_EVERY"] == 0:
            save_and_eval(carry[0].params, int(carry[-1]))

        chunk_start = time.perf_counter()
        carry, (infos, c_loss, a_loss, alpha_v, ent, q_m) = compiled(carry)
        global_step = int(carry[-1])
        metrics = {
            "charts/avg_episodic_return": float(infos["returned_episode_returns"][-1].mean()),
            "charts/avg_episodic_length": float(infos["returned_episode_lengths"][-1].mean()),
            "losses/critic_loss": c_loss[-1].item(),
            "losses/actor_loss": a_loss[-1].item(),
            "losses/q_values": q_m[-1].item(),
            "losses/alpha": alpha_v[-1].item(),
            "losses/entropy": ent[-1].item(),
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(steps_per_chunk / (time.perf_counter() - chunk_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }
        wandb.log(metrics, step=global_step)
        print(
            f"[droq] chunk {chunk + 1}/{num_chunks} | global_step {global_step} | "
            f"avg_return {metrics['charts/avg_episodic_return']:.2f} | "
            f"avg_length {metrics['charts/avg_episodic_length']:.2f} | "
            f"c_loss {metrics['losses/critic_loss']:.4f} | a_loss {metrics['losses/actor_loss']:.4f} | "
            f"alpha {metrics['losses/alpha']:.4f} | H {metrics['losses/entropy']:.4f} | "
            f"q {metrics['losses/q_values']:.4f} | SPS {metrics['charts/SPS']}"
        )

    eval_metrics = save_and_eval(carry[0].params, global_step + 1)
    wandb.finish()
    return eval_metrics
