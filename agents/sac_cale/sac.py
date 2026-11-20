# Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
# Continuous-action SAC on Atari via CALE: the actor outputs a (r, theta, fire) joystick action
# that ContinuousActionWrapper maps onto the 18 ALE actions.
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
import jaxtari
from jaxtari.wrappers import (
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper,
    ContinuousActionWrapper,
)
from agents.sac_cale.sac_eval import evaluate
from agents.run_utils import eval_mod_configs, plan_chunks, rm_metrics
from agents.networks import impala_cnn
from agents.occam.occam import OCCAMWrapper, log_occam_comparison_video
from reward_machines.rm_registry import rm_run_suffix, rm_settings, wrap_reward_machine
from rtpt import RTPT


def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False, tau_cale=0.5, occam_mask=None, reward_machine=None):
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
            full_action_space=True,  # required by ContinuousActionWrapper
        )
        if pixel_based and occam_mask:
            # object masks instead of frames; same frame skip, pooling and stacking as PixelObsWrapper
            env = OCCAMWrapper(env, mask_mode=occam_mask, clip_reward=not eval, game_name=env_id,
                               mods=mods, native_downscaling=native_downscaling)
        elif pixel_based:
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
        if reward_machine is not None:
            # RM reward (game reward in eval) and the RM state appended to the observation
            env = wrap_reward_machine(env, env_id, eval=eval, **reward_machine)
        env = LogWrapper(env)
        # actions within [-TAU_CALE, TAU_CALE] on an axis count as centred (deadzone)
        env = ContinuousActionWrapper(env, tau=tau_cale)
        return env
    return thunk


class CNNEncoder(nn.Module):
    """Shared encoder for pixel observations."""
    cnn_type: str = "nature"

    @nn.compact
    def __call__(self, x):
        if self.cnn_type == "impala":
            x = impala_cnn(x, kernel_init=orthogonal(np.sqrt(2)))
        else:
            x = jnp.transpose(x, (0, 2, 3, 1))
            x = x.astype(jnp.float32) / 255.0
            x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID", kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID", kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID", kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class MLPEncoder(nn.Module):
    """Shared encoder for object-centric observations."""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class Actor(nn.Module):
    """Gaussian policy head; actions are squashed with tanh into the env's Box."""
    action_dim: int
    log_std_min: float = -1.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        log_std = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        log_std = jnp.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        return mean, log_std


class SoftQNetwork(nn.Module):
    """Q(s, a) head on the shared features; the polar joystick action is embedded first."""
    @nn.compact
    def __call__(self, x, a):
        r, theta, fire = a[..., 0], a[..., 1], a[..., 2]
        a_feat = jnp.stack(
            [r, jnp.sin(theta), jnp.cos(theta), fire, r * jnp.cos(theta), r * jnp.sin(theta)],
            axis=-1,
        )
        a_feat = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.1))(a_feat)
        a_feat = nn.relu(a_feat)
        x = jnp.concatenate([x, a_feat], axis=-1)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return nn.Dense(1, kernel_init=orthogonal(1.0))(x)


class SACTrainState(TrainState):
    target_params: flax.core.FrozenDict


@flax.struct.dataclass
class TimeStep:
    obs: jnp.array
    action: jnp.array
    reward: jnp.array
    done: jnp.array  # true termination only; truncated episodes still bootstrap


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    if config.get("PIXEL_BASED", True) and config.get("NUM_ENVS", 1) > 16:
        print("Warning: More than 16 environments may cause OOM on GPU when using pixel-based observations.")

    run_name = f"{config['ENV_ID']}_{config['EXP_NAME']}_{'oc' + rm_run_suffix(config) if not config['PIXEL_BASED'] else 'pixel' + ('_occam-' + config['OCCAM_MASK'] if config.get('OCCAM_MASK') else '') + ('_impala' if config.get('CNN_TYPE', 'nature') == 'impala' else '')}_{config['SEED']}"

    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
        mode=config.get("WANDB_MODE", "online"),
    )
    wandb.define_metric("*", step_metric="charts/global_step")

    # do not modify the seeding
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    train_mods = list(config.get("TRAIN_MODS", []))
    tau_cale = config.get("TAU_CALE", 0.5)

    env = make_env(
        config.get("ENV_ID"),
        train_mods,
        config.get("PIXEL_BASED", True),
        config.get("NATIVE_DOWNSCALING", True),
        False,
        tau_cale,
        occam_mask=config.get("OCCAM_MASK") if config["PIXEL_BASED"] else None,
        reward_machine=rm_settings(config),
    )()

    action_space = env.action_space()
    action_dim = action_space.shape[0]
    low = jnp.array(action_space.low)
    high = jnp.array(action_space.high)
    action_scale = (high - low) / 2.0
    action_bias = (high + low) / 2.0

    obs_shape = env.observation_space().shape
    if config.get("PIXEL_BASED", True):
        obs_shape = obs_shape[:-1]

    num_envs = config["NUM_ENVS"]
    gradient_steps = config.get("GRADIENT_STEPS", 1)

    @jax.jit
    def vmap_reset(rng):
        obs, state = jax.vmap(env.reset)(rng)
        return obs.reshape(rng.shape[0], *obs_shape), state

    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        return next_obs.reshape(action.shape[0], *obs_shape), state, reward, terminated, info

    gamma = config.get("GAMMA", 0.99)
    tau = config.get("TAU", 0.005)
    batch_size = config.get("BATCH_SIZE", 64)
    learning_starts = config.get("LEARNING_STARTS", 10000)
    steps_per_update = config.get("TRAIN_FREQUENCY", 1) * num_envs
    # Dopamine-style reward scale: clipped rewards are small next to the entropy bonus
    reward_scale = config.get("REWARD_SCALE_FACTOR", 10.0)
    # soft bounds on log(alpha): a quadratic penalty pulls it back once it leaves the range
    log_alpha_min = config.get("LOG_ALPHA_MIN", -5.0)
    log_alpha_max = config.get("LOG_ALPHA_MAX", 5.0)

    # same budget rule for every agent: whole jitted chunks within TOTAL_TIMESTEPS
    scan_steps, num_chunks, steps_per_chunk = plan_chunks(
        config["TOTAL_TIMESTEPS"],
        steps_per_update,
        config.get("SCAN_STEPS", 1000),
    )

    # same root split as dqn/c51/pqn, so env resets for a given seed match across agents
    key, init_key = jax.random.split(key, 2)
    encoder_key, actor_key, qf1_key, qf2_key = jax.random.split(init_key, 4)

    encoder_net = CNNEncoder(cnn_type=config.get("CNN_TYPE", "nature")) if config.get("PIXEL_BASED", True) else MLPEncoder()
    actor_net = Actor(action_dim=action_dim)
    critic_net = SoftQNetwork()

    dummy_obs = jnp.zeros((1, *obs_shape))
    dummy_action = jnp.zeros((1, action_dim))
    encoder_params = encoder_net.init(encoder_key, dummy_obs)
    dummy_hidden = encoder_net.apply(encoder_params, dummy_obs)
    qf1_params = critic_net.init(qf1_key, dummy_hidden, dummy_action)
    qf2_params = critic_net.init(qf2_key, dummy_hidden, dummy_action)

    def make_tx(lr):
        return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr, eps=1e-8))

    # the encoder is trained by the critic loss only; the actor sees stop-gradient features
    encoder_state = SACTrainState.create(
        apply_fn=encoder_net.apply,
        params=encoder_params,
        target_params=jax.tree.map(jnp.copy, encoder_params),
        tx=make_tx(config.get("ENCODER_LR", config["Q_LR"])),
    )
    actor_state = TrainState.create(
        apply_fn=actor_net.apply,
        params=actor_net.init(actor_key, dummy_hidden),
        tx=make_tx(config["POLICY_LR"]),
    )
    qf1_state = SACTrainState.create(
        apply_fn=critic_net.apply,
        params=qf1_params,
        target_params=jax.tree.map(jnp.copy, qf1_params),
        tx=make_tx(config["Q_LR"]),
    )
    qf2_state = SACTrainState.create(
        apply_fn=critic_net.apply,
        params=qf2_params,
        target_params=jax.tree.map(jnp.copy, qf2_params),
        tx=make_tx(config["Q_LR"]),
    )

    log_alpha = jnp.zeros(1)
    a_optimizer = optax.adam(learning_rate=config.get("ALPHA_LR", config["Q_LR"]), eps=1e-8)
    a_opt_state = a_optimizer.init(log_alpha)
    target_entropy = -float(action_dim)  # CleanRL default for continuous actions

    def sample_env_action(mean, log_std, key):
        """Sample a tanh-squashed Gaussian action in the env's Box and its log-prob."""
        std = jnp.exp(log_std)
        u = mean + std * jax.random.normal(key, shape=mean.shape)
        action = action_bias + action_scale * jnp.tanh(u)
        log_prob = -0.5 * (((u - mean) / (std + 1e-8)) ** 2 + 2.0 * jnp.log(std + 1e-8) + jnp.log(2.0 * jnp.pi))
        # log|d action / d u| of the tanh and affine maps
        log_det = 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u)) + jnp.log(action_scale)
        return action, (log_prob - log_det).sum(axis=-1)

    replay_buffer = fbx.make_flat_buffer(
        max_length=config.get("BUFFER_SIZE", 100000),
        min_length=learning_starts,
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
    _obs, _state, _reward, _done, _info = vmap_step(_state, jnp.zeros((num_envs, action_dim)))
    _dummy_step = TimeStep(
        obs=_obs[0],
        action=jnp.zeros((action_dim,)),
        reward=_reward[0],
        done=_done[0],
    )
    buffer_state = replay_buffer.init(_dummy_step)

    def full_SAC_step(encoder_state, actor_state, qf1_state, qf2_state, log_alpha, a_opt_state, buffer_state, env_state, obs, rng, global_step):

        def take_action(carry, _):
            encoder_state, actor_state, buffer_state, env_state, obs, global_step, rng = carry
            rng, random_key, actor_sample_key = jax.random.split(rng, 3)

            random_actions = jax.random.uniform(random_key, (num_envs, action_dim), minval=low, maxval=high)
            mean, log_std = actor_net.apply(actor_state.params, encoder_net.apply(encoder_state.params, obs))
            samples, _ = sample_env_action(mean, log_std, actor_sample_key)

            actions = jnp.where(global_step < learning_starts, random_actions, samples)
            next_obs, next_env_state, rewards, terminated, info = vmap_step(env_state, actions)

            timestep = TimeStep(
                obs=obs,
                action=actions,
                reward=rewards,
                done=terminated,
            )
            buffer_state = replay_buffer.add(buffer_state, timestep)
            return (encoder_state, actor_state, buffer_state, next_env_state, next_obs, global_step + num_envs, rng), info

        (encoder_state, actor_state, buffer_state, next_env_state, next_obs, global_step, rng), infos = jax.lax.scan(
            take_action,
            (encoder_state, actor_state, buffer_state, env_state, obs, global_step, rng),
            None,
            length=config.get("TRAIN_FREQUENCY", 1),
        )

        update_actor = (global_step % config.get("POLICY_FREQUENCY", 1)) < steps_per_update

        def do_update(update_carry, _):
            u_encoder_state, u_actor_state, u_qf1_state, u_qf2_state, log_alpha, a_opt_state, u_key = update_carry
            u_key, sample_key, next_action_key, actor_sample_key, alpha_sample_key = jax.random.split(u_key, 5)
            alpha = jnp.exp(log_alpha) if config.get("AUTOTUNE", True) else jnp.full((1,), config.get("ALPHA", 0.2))

            batch = replay_buffer.sample(buffer_state, sample_key).experience
            b_obs = batch.first.obs
            b_act = batch.first.action
            b_rew = batch.first.reward
            b_don = batch.first.done
            b_nobs = batch.second.obs

            # next action from the online actor, evaluated by the target encoder + target critics
            next_mean, next_log_std = actor_net.apply(u_actor_state.params, encoder_net.apply(u_encoder_state.params, b_nobs))
            next_action, next_log_prob = sample_env_action(next_mean, next_log_std, next_action_key)
            next_hidden = encoder_net.apply(u_encoder_state.target_params, b_nobs)
            q1_next_target = critic_net.apply(u_qf1_state.target_params, next_hidden, next_action).squeeze(-1)
            q2_next_target = critic_net.apply(u_qf2_state.target_params, next_hidden, next_action).squeeze(-1)
            min_q_next_target = jnp.minimum(q1_next_target, q2_next_target) - alpha * next_log_prob
            next_q_value = reward_scale * b_rew + (1.0 - b_don) * gamma * min_q_next_target
            next_q_value = jax.lax.stop_gradient(next_q_value)

            def qf_loss_fn(encoder_params, qf1_params, qf2_params):
                hidden = encoder_net.apply(encoder_params, b_obs)
                qf1_a_values = critic_net.apply(qf1_params, hidden, b_act).squeeze(-1)
                qf2_a_values = critic_net.apply(qf2_params, hidden, b_act).squeeze(-1)
                qf1_loss = jnp.mean((qf1_a_values - next_q_value) ** 2)
                qf2_loss = jnp.mean((qf2_a_values - next_q_value) ** 2)
                return qf1_loss + qf2_loss, qf1_a_values.mean()

            (qf_loss, qf1_a_values), (encoder_grads, qf1_grads, qf2_grads) = jax.value_and_grad(
                qf_loss_fn, argnums=(0, 1, 2), has_aux=True
            )(u_encoder_state.params, u_qf1_state.params, u_qf2_state.params)
            new_encoder_state = u_encoder_state.apply_gradients(grads=encoder_grads)
            new_qf1_state = u_qf1_state.apply_gradients(grads=qf1_grads)
            new_qf2_state = u_qf2_state.apply_gradients(grads=qf2_grads)

            def actor_and_alpha_update(c):
                c_actor_state, log_alpha, a_opt_state = c
                # stop gradient: the actor must not reshape the critic's encoder features
                hidden = jax.lax.stop_gradient(encoder_net.apply(new_encoder_state.params, b_obs))

                def actor_loss_fn(actor_params):
                    mean, log_std = actor_net.apply(actor_params, hidden)
                    pi_action, log_prob = sample_env_action(mean, log_std, actor_sample_key)
                    qf1_pi = critic_net.apply(new_qf1_state.params, hidden, pi_action).squeeze(-1)
                    qf2_pi = critic_net.apply(new_qf2_state.params, hidden, pi_action).squeeze(-1)
                    return jnp.mean(jax.lax.stop_gradient(alpha) * log_prob - jnp.minimum(qf1_pi, qf2_pi))

                actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(c_actor_state.params)
                new_actor_state = c_actor_state.apply_gradients(grads=actor_grads)

                # alpha is tuned on a fresh sample from the updated actor
                mean, log_std = actor_net.apply(new_actor_state.params, hidden)
                _, log_prob = sample_env_action(mean, log_std, alpha_sample_key)
                log_prob = jax.lax.stop_gradient(log_prob)

                def alpha_loss_fn(a_log_alpha):
                    range_penalty = (
                        jnp.mean(jnp.maximum(log_alpha_min - a_log_alpha, 0.0) ** 2)
                        + jnp.mean(jnp.maximum(a_log_alpha - log_alpha_max, 0.0) ** 2)
                    )
                    return -jnp.mean(jnp.exp(a_log_alpha) * (log_prob + target_entropy)) + 0.1 * range_penalty

                if config.get("AUTOTUNE", True):
                    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(log_alpha)
                    updates, a_opt_state = a_optimizer.update(alpha_grad, a_opt_state, log_alpha)
                    log_alpha = optax.apply_updates(log_alpha, updates)
                else:
                    alpha_loss = jnp.array(0.0)
                return (new_actor_state, log_alpha, a_opt_state), (actor_loss, alpha_loss, log_prob.mean())

            (new_actor_state, log_alpha, a_opt_state), (actor_loss, alpha_loss, log_prob_mean) = jax.lax.cond(
                update_actor,
                actor_and_alpha_update,
                lambda c: (c, (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))),
                (u_actor_state, log_alpha, a_opt_state),
            )

            return (
                (new_encoder_state, new_actor_state, new_qf1_state, new_qf2_state, log_alpha, a_opt_state, u_key),
                (qf_loss, actor_loss, qf1_a_values, alpha, alpha_loss, log_prob_mean),
            )

        def scanned_update(carry):
            carry, metrics = jax.lax.scan(do_update, carry, None, length=gradient_steps)
            return carry, jax.tree.map(lambda m: m[-1].squeeze(), metrics)

        zero = jnp.array(0.0)
        (encoder_state, actor_state, qf1_state, qf2_state, log_alpha, a_opt_state, rng), update_metrics = jax.lax.cond(
            replay_buffer.can_sample(buffer_state),
            scanned_update,
            lambda c: (c, (zero, zero, zero, zero, zero, zero)),
            (encoder_state, actor_state, qf1_state, qf2_state, log_alpha, a_opt_state, rng),
        )

        def update_target_networks(c):
            return tuple(
                s.replace(target_params=optax.incremental_update(s.params, s.target_params, tau)) for s in c
            )

        update_target_flag = jnp.logical_and(
            replay_buffer.can_sample(buffer_state),
            (global_step % config.get("TARGET_UPDATE_FREQUENCY", 1)) < steps_per_update,
        )
        encoder_state, qf1_state, qf2_state = jax.lax.cond(
            update_target_flag,
            update_target_networks,
            lambda c: c,
            (encoder_state, qf1_state, qf2_state),
        )

        return (encoder_state, actor_state, qf1_state, qf2_state, log_alpha, a_opt_state, buffer_state, next_env_state, next_obs, rng, global_step), (infos, *update_metrics)

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
                    pixel_based=config["PIXEL_BASED"],
                    native_downscaling=config["NATIVE_DOWNSCALING"],
                    eval=True,
                    occam_mask=config.get("OCCAM_MASK") if config["PIXEL_BASED"] else None,
                    reward_machine=rm_settings(config),
                    tau_cale=tau_cale,
                ),
                config["ENV_ID"],
                eval_episodes=10,
                Model=(partial(CNNEncoder, cnn_type=config.get("CNN_TYPE", "nature")) if config["PIXEL_BASED"] else MLPEncoder, Actor),
                seed=config["SEED"] + 42,  # use a different seed for evaluation
            )
            metrics[mod_label] = float(np.mean(jax.device_get(episodic_returns)))
            wandb.log({f"eval/episodic_return_{mod_label}": metrics[mod_label]}, step=step_count)

            if config.get("CAPTURE_VIDEO", False):
                # Instantiate a clean renderer immune to the training env's downscaling
                clean_renderer = jaxtari.make(config["ENV_ID"], mods=mods_cfg).renderer
                frames = jnp.transpose(jax.vmap(clean_renderer.render)(env_states), (0, 3, 1, 2))
                video = wandb.Video(np.array(frames), fps=30, format="mp4")
                wandb.log({f"eval/video_{mod_label}": video}, step=step_count)
                print(f"Video (eval) logged to wandb with {frames.shape[0]} frames ({mod_label}).")
                if config["PIXEL_BASED"] and config.get("OCCAM_MASK"):
                    log_occam_comparison_video(config["ENV_ID"], config["OCCAM_MASK"], env_states,
                                               mods=mods_cfg, mod_label=mod_label, step=step_count)
        return metrics

    def eval_params(carry):
        return {"encoder": carry[0].params, "actor": carry[1].params}

    print("[sac_cale] start compile...")
    compile_start = time.perf_counter()
    global_step = jnp.array(0, dtype=jnp.int32)
    carry = (encoder_state, actor_state, qf1_state, qf2_state, log_alpha, a_opt_state, buffer_state, _state, _obs, key, global_step)

    def train_chunk(carry):
        return jax.lax.scan(lambda c, _: full_SAC_step(*c), carry, None, length=scan_steps)

    # donate the carry so XLA reuses the replay buffer's memory; lower/compile AOT so nothing is donated before the loop
    compiled = jax.jit(train_chunk, donate_argnums=(0,)).lower(carry).compile()
    print(f"[sac_cale] compilation time: {time.perf_counter() - compile_start:.2f}s")

    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=num_chunks)
    rtpt.start()
    run_time = time.perf_counter()
    print(f"[sac_cale] starting training: {num_chunks} chunks x {steps_per_chunk} env steps")
    for chunk in range(num_chunks):
        rtpt.step()
        if config["EVAL_DURING_TRAIN"] and chunk > 0 and chunk % config["EVAL_EVERY"] == 0:
            save_and_eval(eval_params(carry), int(carry[-1]))

        chunk_start = time.perf_counter()
        carry, (infos, qf_loss, actor_loss, qf1_val, alpha, alpha_loss, log_prob_mean) = compiled(carry)
        global_step = int(carry[-1])
        metrics = {
            "charts/avg_episodic_return": float(infos["returned_episode_returns"][-1].mean()),
            "charts/avg_episodic_length": float(infos["returned_episode_lengths"][-1].mean()),
            "losses/qf_loss": float(qf_loss[-1]),
            "losses/actor_loss": float(actor_loss[-1]),
            "losses/qf1_values": float(qf1_val[-1]),
            "losses/alpha_values": float(alpha[-1]),
            "losses/alpha_loss": float(alpha_loss[-1]),
            "losses/log_prob_mean": float(log_prob_mean[-1]),
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(steps_per_chunk / (time.perf_counter() - chunk_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }
        metrics.update(rm_metrics(infos))
        wandb.log(metrics, step=global_step)
        print(
            f"[sac_cale] chunk {chunk + 1}/{num_chunks} | global_step {global_step} | "
            f"avg_return {metrics['charts/avg_episodic_return']:.2f} | "
            f"avg_length {metrics['charts/avg_episodic_length']:.2f} | "
            f"qf_loss {metrics['losses/qf_loss']:.4f} | act_loss {metrics['losses/actor_loss']:.4f} | "
            f"alpha {metrics['losses/alpha_values']:.4f} | SPS {metrics['charts/SPS']}"
        )

    eval_metrics = save_and_eval(eval_params(carry), global_step + 1)
    wandb.finish()
    return eval_metrics
