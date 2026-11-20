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
from flax.linen.initializers import constant, ones, zeros
from flax.training.train_state import TrainState
import jaxtari
from jaxtari.wrappers import (
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper
)
from agents.crossq.crossq_eval import evaluate
from agents.run_utils import eval_mod_configs, plan_chunks, rm_metrics
from agents.networks import impala_cnn
from agents.occam.occam import OCCAMWrapper, log_occam_comparison_video
from reward_machines.rm_registry import rm_run_suffix, rm_settings, wrap_reward_machine
from rtpt import RTPT

# CrossQ for Atari performs poorly, it either doesn't converge or converge worse than SAC.

def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False, occam_mask=None, reward_machine=None):
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
        return env
    return thunk

class BatchRenorm(nn.Module):
    configs: dict
    network: int

    use_running_average: bool = False
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-3

    use_scale: bool = True
    use_bias: bool = True
    scale_init: callable = ones
    bias_init: callable = zeros

    dtype: object = None
    param_dtype: object = jnp.float32

    @nn.compact
    def __call__(self, x, step):
        x = jnp.asarray(x)
        x = x.astype(jnp.promote_types(x.dtype, jnp.float32))

        axis = self.axis if self.axis >= 0 else x.ndim + self.axis

        warmup_steps = self.configs.get(
            "BRN_WARMUP_STEPS", 100_000
        )

        relaxation_steps = self.configs.get(
            "BRN_RELAXATION_STEPS", 100_000
        )

        final_rmax = self.configs.get(
            "BRN_RMAX", 3.0
        )

        final_dmax = self.configs.get(
            "BRN_DMAX", 5.0
        )

        feature_shape = (x.shape[axis],)

        if self.use_scale:
            scale = self.param(
                "scale",
                self.scale_init,
                feature_shape,
                self.param_dtype,
            )
        else:
            scale = None

        if self.use_bias:
            bias = self.param(
                "bias",
                self.bias_init,
                feature_shape,
                self.param_dtype,
            )
        else:
            bias = None

        running_mean = self.variable(
            "batch_stats",
            "mean",
            lambda: jnp.zeros(
                feature_shape,
                dtype=self.param_dtype,
            ),
        )

        running_var = self.variable(
            "batch_stats",
            "var",
            lambda: jnp.ones(
                feature_shape,
                dtype=self.param_dtype,
            ),
        )

        reduce_axes = tuple(
            i for i in range(x.ndim)
            if i != axis
        )

        broadcast_shape = [1] * x.ndim
        broadcast_shape[axis] = x.shape[axis]

        if self.use_running_average:
            mean = running_mean.value.reshape(broadcast_shape)
            var = running_var.value.reshape(broadcast_shape)

            y = (x - mean) * jax.lax.rsqrt(var + self.epsilon)

        else:
            batch_mean = jnp.mean(
                x,
                axis=reduce_axes,
            )

            batch_var = jnp.mean(
                jnp.square(
                    x - batch_mean.reshape(broadcast_shape)
                ),
                axis=reduce_axes,
            )

            batch_std = jnp.sqrt(
                batch_var + self.epsilon
            )

            old_mean = running_mean.value
            old_var = running_var.value
            old_std = jnp.sqrt(
                old_var + self.epsilon
            )
            r_raw = batch_std / old_std
            d_raw = (batch_mean - old_mean) / old_std
            relaxation_progress = jnp.clip(
                (step - warmup_steps) / relaxation_steps,
                0.0,
                1.0,
            )
            relaxed_rmax = (
                1.0
                + relaxation_progress * (final_rmax - 1.0)
            )
            relaxed_dmax = (
                relaxation_progress * final_dmax
            )
            r = jnp.clip(
                r_raw,
                1.0 / relaxed_rmax,
                relaxed_rmax,
            )
            d = jnp.clip(
                d_raw,
                -relaxed_dmax,
                relaxed_dmax,
            )
            r = jax.lax.stop_gradient(r)
            d = jax.lax.stop_gradient(d)
            batch_mean_b = batch_mean.reshape(
                broadcast_shape
            )

            batch_std_b = batch_std.reshape(
                broadcast_shape
            )

            r_b = r.reshape(
                broadcast_shape
            )

            d_b = d.reshape(
                broadcast_shape
            )

            x_hat = (
                (x - batch_mean_b) / batch_std_b
            ) * r_b + d_b

            y = x_hat
            running_mean.value = (
                self.momentum * old_mean
                + (1.0 - self.momentum) * batch_mean
            )

            running_var.value = (
                self.momentum * old_var
                + (1.0 - self.momentum) * batch_var
            )
        if scale is not None:
            y = y * scale.reshape(broadcast_shape)

        if bias is not None:
            y = y + bias.reshape(broadcast_shape)

        if self.dtype is not None:
            y = y.astype(self.dtype)

        return y

class Pixel_Actor_Discrete(nn.Module):
    action_dim: int
    configs: dict
    cnn_type: str = "nature"

    @nn.compact
    def __call__(self, x, step, key, train=False):
        if self.cnn_type == "impala":
            x = impala_cnn(
                x,
                kernel_init=nn.initializers.he_normal(),
                norm=lambda h: BatchRenorm(use_running_average=not train,
                    momentum=self.configs.get("BATCHNORM_MOMENTUM", 0.99),
                    configs=self.configs,
                    network=1,
                )(h, step),
            )
        else:
            x = jnp.transpose(x, (0, 2, 3, 1))
            x = x.astype(jnp.float32) / 255.0
            x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
            x = BatchRenorm(use_running_average=not train,
                momentum=self.configs.get("BATCHNORM_MOMENTUM", 0.99),
                configs=self.configs,
                network=1,
            )(x, step)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
            x = BatchRenorm(use_running_average=not train,
                momentum=self.configs.get("BATCHNORM_MOMENTUM", 0.99),
                configs=self.configs,
                network=1,
            )(x, step)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
            x = BatchRenorm(use_running_average=not train,
                momentum=self.configs.get("BATCHNORM_MOMENTUM", 0.99),
                configs=self.configs,
                network=1,
            )(x, step)
            x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0)  )(x)
        x = BatchRenorm(use_running_average=not train,
            momentum=self.configs.get("BATCHNORM_MOMENTUM", 0.99),
            configs=self.configs,
            network=1,
        )(x, step)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        sample = jax.random.categorical(key, x)
        action_probs = jax.nn.softmax(x, axis=-1)
        log_probs = nn.log_softmax(x, axis=-1)
        return sample, log_probs, action_probs

class Pixel_Critic(nn.Module):
    action_dim: int
    configs: dict
    cnn_type: str = "nature"

    @nn.compact
    def __call__(self, x, step, train = False):
        if self.cnn_type == "impala":
            x = impala_cnn(
                x,
                kernel_init=nn.initializers.he_normal(),
                norm=lambda h: BatchRenorm(use_running_average=not train,
                    momentum=self.configs.get("BATCHNORM_MOMENTUM", 0.99),
                    configs=self.configs,
                    network=1,
                )(h, step),
            )
        else:
            x = jnp.transpose(x, (0, 2, 3, 1))
            x = x.astype(jnp.float32) / 255.0
            x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
            x = BatchRenorm(use_running_average=not train,
                momentum=self.configs.get("BATCHNORM_MOMENTUM", 0.99),
                configs=self.configs,
                network=1,
            )(x, step)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
            x = BatchRenorm(use_running_average=not train,
                momentum=self.configs.get("BATCHNORM_MOMENTUM", 0.99),
                configs=self.configs,
                network=1,
            )(x, step)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
            x = BatchRenorm(use_running_average=not train,
                momentum=self.configs.get("BATCHNORM_MOMENTUM", 0.99),
                configs=self.configs,
                network=1,
            )(x, step)
            x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))
        x = nn.Dense(2048, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = BatchRenorm(use_running_average=not train,
            momentum=self.configs.get("BATCHNORM_MOMENTUM", 0.99),
            configs=self.configs,
            network=1,
        )(x, step)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        return x

class MLP_Actor_Discrete(nn.Module):
    action_dim: int
    configs: dict

    @nn.compact
    def __call__(self, x, step, key, train=False):
        x = nn.Dense(256, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = BatchRenorm(use_running_average=not train,
            momentum=self.configs.get("BATCHNORM_MOMENTUM", 0.99),
            configs=self.configs,
            network=1,
        )(x, step)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = BatchRenorm(use_running_average=not train,
            momentum=self.configs.get("BATCHNORM_MOMENTUM", 0.99),
            configs=self.configs,
            network=1,
        )(x, step)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        sample = jax.random.categorical(key, x)
        action_probs = jax.nn.softmax(x, axis=-1)
        log_probs = nn.log_softmax(x, axis=-1)
        return sample, log_probs, action_probs

class MLP_Critic(nn.Module):
    action_dim: int
    configs: dict

    @nn.compact
    def __call__(self, x, step, train = False):
        x = nn.Dense(2048, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = BatchRenorm(use_running_average=not train,
            momentum=self.configs.get("BATCHNORM_MOMENTUM", 0.99),
            configs=self.configs,
            network=0,
        )(x, step)
        x = nn.relu(x)
        x = nn.Dense(2048, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = BatchRenorm(use_running_average=not train,
            momentum=self.configs.get("BATCHNORM_MOMENTUM", 0.99),
            configs=self.configs,
            network=0,
        )(x, step)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        return x


class CrossQTrainState(TrainState):
    batch_stats: flax.core.FrozenDict

@flax.struct.dataclass
class TimeStep:
    obs: jnp.array
    action: jnp.array
    reward: jnp.array
    done: jnp.array

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

    env = make_env(
        config.get("ENV_ID"),
        train_mods,
        config.get("PIXEL_BASED", True),
        config.get("NATIVE_DOWNSCALING", True),
        False,
        occam_mask=config.get("OCCAM_MASK") if config["PIXEL_BASED"] else None,
        reward_machine=rm_settings(config),
    )()

    action_dim = env.action_space().n


    obs_shape = env.observation_space().shape
    if config.get("PIXEL_BASED", True):
        obs_shape = obs_shape[:-1]

    num_envs = config["NUM_ENVS"]
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
    batch_size = config.get("BATCH_SIZE", 64)
    learning_starts = config.get("LEARNING_STARTS", 20000)
    use_target_network = config.get("USE_TARGET_NETWORK", True)
    tau = config.get("TAU", 1.0)
    steps_per_update = config.get("TRAIN_FREQUENCY", 4) * config.get("NUM_ENVS", 1)

    # same budget rule for every agent: whole jitted chunks within TOTAL_TIMESTEPS
    scan_steps, num_chunks, steps_per_chunk = plan_chunks(
        config["TOTAL_TIMESTEPS"],
        steps_per_update,
        config.get("SCAN_STEPS", 1000),
    )

    TwinCritic = nn.vmap(
        Pixel_Critic if config.get("PIXEL_BASED", True) else MLP_Critic,
        variable_axes={"params": 0, "batch_stats": 0},
        split_rngs={"params": True},
        in_axes=(None, None, None),
        out_axes=0,
        axis_size=2
    )

    # same root split as dqn/c51/pqn, so env resets for a given seed match across agents
    key, init_key = jax.random.split(key, 2)
    actor_key, actor_key2, qf_key = jax.random.split(init_key, 3)

    if config.get("PIXEL_BASED", True):
        actor_net = Pixel_Actor_Discrete(action_dim=action_dim, configs=config, cnn_type=config.get("CNN_TYPE", "nature"))
        critic_net = TwinCritic(action_dim=action_dim, configs=config, cnn_type=config.get("CNN_TYPE", "nature"))
    else:
        actor_net = MLP_Actor_Discrete(action_dim=action_dim, configs=config)
        critic_net = TwinCritic(action_dim=action_dim, configs=config)


    dummy_step = 1
    dummy_obs = jnp.zeros((1, *obs_shape))
    critic_variables = critic_net.init(qf_key, dummy_obs, dummy_step, True)
    actor_variables = actor_net.init(actor_key, dummy_obs, dummy_step, actor_key2, True)

    actor_state = CrossQTrainState.create(
        apply_fn=actor_net.apply,
        params=actor_variables["params"],
        batch_stats=actor_variables["batch_stats"],
        tx=optax.adam(learning_rate=config.get("LEARNING_RATE", 3e-4), eps=1e-4, b1=0.5),
    )

    qf_state = CrossQTrainState.create(
        apply_fn=critic_net.apply,
        params=critic_variables["params"],
        batch_stats=critic_variables["batch_stats"],
        tx=optax.adam(learning_rate=config.get("LEARNING_RATE", 3e-4), eps=1e-4, b1=0.5),
    )
    # copy so the target does not alias qf_state's buffers (the carry is donated)
    qf_target = jax.tree.map(jnp.copy, {"params": critic_variables["params"], "batch_stats": critic_variables["batch_stats"]})

    replay_buffer = fbx.make_prioritised_flat_buffer(
        max_length=config.get("BUFFER_SIZE", int(1e6)),
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
    _obs, _state, _reward, _done, _info = vmap_step(_state, jnp.zeros((num_envs,), dtype=jnp.int32))

    _dummy_step = TimeStep(
        obs=_obs[0],
        action=jnp.zeros((), dtype=jnp.int32),
        reward=_reward[0],
        done=_done[0],
    )
    buffer_state = replay_buffer.init(_dummy_step)

    log_alpha = jnp.zeros(1)
    a_optimizer = optax.adam(learning_rate=config.get("LEARNING_RATE", 3e-4), eps=1e-4)
    a_opt_state = a_optimizer.init(log_alpha)
    target_entropy = -config.get("TARGET_ENTROPY_SCALE", 0.89) * jnp.log(1 / action_dim)

    log_alpha_min = jnp.log(jnp.asarray(config.get("ALPHA_MIN", 1e-4), dtype=log_alpha.dtype))
    log_alpha_max = jnp.log(jnp.asarray(config.get("ALPHA_MAX", 1.0), dtype=log_alpha.dtype))

    policy_delay = max(int(config.get("POLICY_DELAY", 1)), 1)


    def full_CrossQ_step(actor_state, qf_state, qf_target, log_alpha, a_opt_state, n_updates, last_actor_loss, buffer_state, env_state, obs, rng, global_step):

        def take_action(carry, _):
            actor_state, buffer_state, env_state, obs, global_step, rng = carry
            rng, action_rng, actor_sample_key = jax.random.split(rng, 3)
            action_sample_keys = jax.random.split(action_rng, num_envs)

            p_actor = {"params": actor_state.params, "batch_stats": actor_state.batch_stats}
            random_actions = jax.vmap(env.action_space().sample)(action_sample_keys)
            samples, _, _ = actor_state.apply_fn(p_actor, obs, global_step, actor_sample_key, False)

            actions = jnp.where(global_step < learning_starts, random_actions, samples)
            next_obs, next_env_state, rewards, next_done, info = vmap_step(env_state, actions)

            timestep = TimeStep(
                obs=obs,
                action=actions,
                reward=rewards,
                done=next_done,
            )
            buffer_state = replay_buffer.add(buffer_state, timestep)
            return (actor_state, buffer_state, next_env_state, next_obs, global_step + num_envs, rng), info

        (actor_state, buffer_state, next_env_state, next_obs, global_step, rng), infos = jax.lax.scan(
            take_action,
            (actor_state, buffer_state, env_state, obs, global_step, rng),
            None,
            length=config.get("TRAIN_FREQUENCY", 4),
        )

        def do_update(update_carry, _):
            u_actor_state, u_qf_state, log_alpha, a_opt_state, n_updates, last_actor_loss, u_key = update_carry
            u_key, sample_key, sample_key2, sample_key3 = jax.random.split(u_key, 4)
            alpha = jnp.exp(log_alpha) if config.get("AUTOTUNE", True) else config.get("ALPHA", 0.2)

            batch = replay_buffer.sample(buffer_state, sample_key).experience
            b_obs = batch.first.obs
            b_act = batch.first.action
            b_rew = batch.first.reward
            b_don = batch.first.done
            b_nobs = batch.second.obs

            p_actor = {"params": u_actor_state.params, "batch_stats": u_actor_state.batch_stats}
            _, next_state_log_pi, next_state_action_probs = u_actor_state.apply_fn(p_actor, b_nobs, global_step, sample_key2, False)

            if use_target_network:
                next_q_target = u_qf_state.apply_fn(qf_target, b_nobs, n_updates, False) # (2, B, A)

            def qf_loss_fn(qf_params, qf_state):
                p = {"params": qf_params, "batch_stats": qf_state.batch_stats}
                all_q, new_critic_batch_stats = qf_state.apply_fn(p, jnp.concatenate([b_obs, b_nobs]), n_updates, True, mutable=["batch_stats"]) # (2, 2B, A)
                qf_preds, next_q_values = jnp.split(all_q, 2, axis=1) # (2, B, A), (2, B, A)
                if use_target_network:
                    next_q_values = next_q_target
                min_next_q = jnp.min(next_q_values, axis=0) # (B, A)
                min_next_q = jnp.sum(next_state_action_probs * (min_next_q - alpha * next_state_log_pi), axis=-1) # (B,)
                min_next_q = jax.lax.stop_gradient(b_rew.flatten() + (1.0 - b_don.flatten()) * gamma * min_next_q) # (B,)
                qf_pred_a_values = qf_preds[:,jnp.arange(qf_preds.shape[1]), b_act.flatten()] # (2, B)

                qf_loss = jnp.mean((qf_pred_a_values - min_next_q) ** 2, axis=1).sum(axis=0)
                return qf_loss, (qf_pred_a_values[0, ...].mean(), new_critic_batch_stats)

            (qf_loss, (qf1_pred_a_values, new_critic_batch_stats)), grad = jax.value_and_grad(qf_loss_fn, has_aux=True)(u_qf_state.params, u_qf_state)
            new_qf_state = u_qf_state.apply_gradients(grads=grad)
            new_qf_state = new_qf_state.replace(batch_stats=new_critic_batch_stats["batch_stats"])

            def update_actor_and_alpha(delayed_carry):
                d_actor_state, d_log_alpha, d_a_opt_state = delayed_carry

                def actor_loss_fn(actor_params, actor_state):
                    p_critic = {"params": new_qf_state.params, "batch_stats": new_qf_state.batch_stats}
                    p_actor = {"params": actor_params, "batch_stats": actor_state.batch_stats}
                    new_qf_preds = new_qf_state.apply_fn(p_critic, b_obs, global_step, False)

                    (_, log_pi, action_probs), new_actor_batch_stats = actor_state.apply_fn(p_actor, b_obs, n_updates // policy_delay, sample_key3, True, mutable=["batch_stats"])

                    min_qf_values = jax.lax.stop_gradient(jnp.min(new_qf_preds, axis=0))
                    actor_loss = jnp.sum((action_probs * ((alpha * log_pi) - min_qf_values)), axis=-1).mean()
                    return actor_loss, (log_pi, action_probs, new_actor_batch_stats)

                (actor_loss, (log_pi, action_probs, new_actor_batch_stats)), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(d_actor_state.params, d_actor_state)
                new_actor_state = d_actor_state.apply_gradients(grads=actor_grads)
                new_actor_state = new_actor_state.replace(batch_stats=new_actor_batch_stats["batch_stats"])

                if config.get("AUTOTUNE", True):
                    def alpha_loss_fn(a_log_alpha):
                        entropy_diff = jax.lax.stop_gradient(jnp.sum(action_probs * (log_pi + target_entropy), axis=-1))
                        return -jnp.mean(a_log_alpha * entropy_diff)

                    _, alpha_grad = jax.value_and_grad(alpha_loss_fn)(d_log_alpha)
                    updates, d_a_opt_state = a_optimizer.update(alpha_grad, d_a_opt_state, d_log_alpha)
                    d_log_alpha = optax.apply_updates(d_log_alpha, updates)
                    d_log_alpha = jnp.clip(d_log_alpha, log_alpha_min, log_alpha_max)

                return (new_actor_state, d_log_alpha, d_a_opt_state), actor_loss

            def skip_actor_and_alpha(delayed_carry):
                return delayed_carry, last_actor_loss

            (new_actor_state, log_alpha, a_opt_state), actor_loss = jax.lax.cond(
                ((n_updates + 1) % policy_delay) == 0,
                update_actor_and_alpha,
                skip_actor_and_alpha,
                (u_actor_state, log_alpha, a_opt_state),
            )

            return (new_actor_state, new_qf_state, log_alpha, a_opt_state, n_updates + 1, actor_loss, u_key), (qf_loss, actor_loss, qf1_pred_a_values, alpha)


        def scanned_update(carry):
            carry, metrics = jax.lax.scan(do_update, carry, None, length=gradient_steps)
            qf_l, act_l, qf1_v, a_val = metrics
            return carry, (qf_l[-1], act_l[-1], qf1_v[-1], a_val[-1].squeeze())

        (actor_state, qf_state, log_alpha, a_opt_state, n_updates, last_actor_loss, rng), (qf_loss, actor_loss, qf1_val, alpha) = jax.lax.cond(
            replay_buffer.can_sample(buffer_state),
            lambda c: scanned_update(c),
            lambda c: (c, (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))),
            (actor_state, qf_state, log_alpha, a_opt_state, n_updates, last_actor_loss, rng),
        )

        if use_target_network:
            def update_target_network(c_qf_target):
                return {
                    "params": optax.incremental_update(qf_state.params, c_qf_target["params"], tau),
                    "batch_stats": optax.incremental_update(qf_state.batch_stats, c_qf_target["batch_stats"], tau),
                }

            update_target_flag = jnp.logical_and(
                replay_buffer.can_sample(buffer_state),
                (global_step % config.get("TARGET_UPDATE_FREQUENCY", 8000)) < steps_per_update
            )

            qf_target = jax.lax.cond(
                update_target_flag,
                update_target_network,
                lambda c: c,
                qf_target
            )

        return (actor_state, qf_state, qf_target, log_alpha, a_opt_state, n_updates, last_actor_loss, buffer_state, next_env_state, next_obs, rng, global_step), (infos, qf_loss, actor_loss, qf1_val, alpha)

    def save_and_eval(actor_vars, step_count):
        # the actor uses BatchRenorm: inference needs its params and running statistics (batch_stats)
        if config.get("SAVE_PATH") is not None:
            save_dir = os.path.join(config["SAVE_PATH"], run_name)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model')
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes([config, actor_vars]))
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")
        metrics = {}
        for mods_cfg, mod_label in eval_mod_configs(config):
            episodic_returns, env_states = evaluate(
                actor_vars,
                partial(
                    make_env,
                    mods=mods_cfg,
                    pixel_based=config["PIXEL_BASED"],
                    native_downscaling=config["NATIVE_DOWNSCALING"],
                    eval=True,
                    occam_mask=config.get("OCCAM_MASK") if config["PIXEL_BASED"] else None,
                    reward_machine=rm_settings(config),
                ),
                config["ENV_ID"],
                eval_episodes=10,
                Model=partial(partial(Pixel_Actor_Discrete, cnn_type=config.get("CNN_TYPE", "nature")) if config["PIXEL_BASED"] else MLP_Actor_Discrete, configs=config),
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

    def actor_variables_of(carry):
        return {"params": carry[0].params, "batch_stats": carry[0].batch_stats}

    print("[crossq] start compile...")
    compile_start = time.perf_counter()
    global_step = jnp.array(0, dtype=jnp.int32)
    n_updates = jnp.array(0, dtype=jnp.int32)
    last_actor_loss = jnp.array(0.0)
    carry = (actor_state, qf_state, qf_target, log_alpha, a_opt_state, n_updates, last_actor_loss, buffer_state, _state, _obs, key, global_step)

    def train_chunk(carry):
        return jax.lax.scan(lambda c, _: full_CrossQ_step(*c), carry, None, length=scan_steps)

    # donate the carry so XLA reuses the replay buffer's memory; lower/compile AOT so nothing is donated before the loop
    compiled = jax.jit(train_chunk, donate_argnums=(0,)).lower(carry).compile()
    print(f"[crossq] compilation time: {time.perf_counter() - compile_start:.2f}s")

    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=num_chunks)
    rtpt.start()
    run_time = time.perf_counter()
    print(f"[crossq] starting training: {num_chunks} chunks x {steps_per_chunk} env steps")
    for chunk in range(num_chunks):
        rtpt.step()
        if config["EVAL_DURING_TRAIN"] and chunk > 0 and chunk % config["EVAL_EVERY"] == 0:
            save_and_eval(actor_variables_of(carry), int(carry[-1]))

        chunk_start = time.perf_counter()
        carry, (infos, qf_loss, actor_loss, qf1_val, alpha) = compiled(carry)
        global_step = int(carry[-1])
        metrics = {
            "charts/avg_episodic_return": float(infos["returned_episode_returns"][-1].mean()),
            "charts/avg_episodic_length": float(infos["returned_episode_lengths"][-1].mean()),
            "losses/qf_loss": float(qf_loss[-1]),
            "losses/actor_loss": float(actor_loss[-1]),
            "losses/qf1_values": float(qf1_val[-1]),
            "losses/alpha_values": float(alpha[-1]),
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(steps_per_chunk / (time.perf_counter() - chunk_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }
        metrics.update(rm_metrics(infos))
        wandb.log(metrics, step=global_step)
        print(
            f"[crossq] chunk {chunk + 1}/{num_chunks} | global_step {global_step} | "
            f"avg_return {metrics['charts/avg_episodic_return']:.2f} | "
            f"avg_length {metrics['charts/avg_episodic_length']:.2f} | "
            f"qf_loss {metrics['losses/qf_loss']:.4f} | act_loss {metrics['losses/actor_loss']:.4f} | "
            f"alpha {metrics['losses/alpha_values']:.4f} | SPS {metrics['charts/SPS']}"
        )

    eval_metrics = save_and_eval(actor_variables_of(carry), global_step + 1)
    wandb.finish()
    return eval_metrics
