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
    FlattenObservationWrapper,
    LogWrapper
)
from agents.simba_sac.simba_sac_eval import evaluate
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


class SimBaResidualBlock(nn.Module):
    hidden_dim: int
    expansion_factor: int = 4

    @nn.compact
    def __call__(self, x):
        res = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim * self.expansion_factor, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        return res + x


class SimBaTrunk(nn.Module):
    hidden_dim: int
    num_blocks: int
    expansion_factor: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        for _ in range(self.num_blocks):
            x = SimBaResidualBlock(self.hidden_dim, self.expansion_factor)(x)
        return nn.LayerNorm()(x)


class SimBa_Pixel_Actor_Discrete(nn.Module):
    action_dim: int
    hidden_dim: int = 128
    num_blocks: int = 1
    expansion_factor: int = 4

    @nn.compact
    def __call__(self, x, key):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = SimBaTrunk(self.hidden_dim, self.num_blocks, self.expansion_factor)(x)
        x = nn.Dense(self.action_dim, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        sample = jax.random.categorical(key, x)
        action_probs = jax.nn.softmax(x, axis=-1)
        log_probs = nn.log_softmax(x, axis=-1)
        return sample, log_probs, action_probs

class SimBa_Pixel_Critic(nn.Module):
    action_dim: int
    hidden_dim: int = 512
    num_blocks: int = 2
    expansion_factor: int = 4

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID", kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = SimBaTrunk(self.hidden_dim, self.num_blocks, self.expansion_factor)(x)
        x = nn.Dense(self.action_dim, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        return x


class SimBa_MLP_Actor_Discrete(nn.Module):
    action_dim: int
    hidden_dim: int = 128
    num_blocks: int = 1
    expansion_factor: int = 4

    @nn.compact
    def __call__(self, x, key):
        x = SimBaTrunk(self.hidden_dim, self.num_blocks, self.expansion_factor)(x)
        x = nn.Dense(self.action_dim, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        sample = jax.random.categorical(key, x)
        action_probs = jax.nn.softmax(x, axis=-1)
        log_probs = nn.log_softmax(x, axis=-1)
        return sample, log_probs, action_probs

class SimBa_MLP_Critic(nn.Module):
    action_dim: int
    hidden_dim: int = 512
    num_blocks: int = 2
    expansion_factor: int = 4

    @nn.compact
    def __call__(self, x):
        x = SimBaTrunk(self.hidden_dim, self.num_blocks, self.expansion_factor)(x)
        x = nn.Dense(self.action_dim, kernel_init=nn.initializers.he_normal(), bias_init=constant(0.0))(x)
        return x


class SACTrainState(TrainState):
    target_params: flax.core.FrozenDict


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

    run_name = f"{config['ENV_ID']}_{config['EXP_NAME']}_{'oc' if not config['PIXEL_BASED'] else 'pixel'}_{config['SEED']}"

    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
    )
    wandb.define_metric("*", step_metric="charts/global_step")

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
    tau = config.get("TAU", 1.0)
    batch_size = config.get("BATCH_SIZE", 64)
    learning_starts = config.get("LEARNING_STARTS", 20000)
    steps_per_update = config.get("TRAIN_FREQUENCY", 4) * config.get("NUM_ENVS", 1)


    key, actor_key, actor_key2, qf1_key, qf2_key = jax.random.split(key, 5)
    
    expansion_factor = config.get("SIMBA_EXPANSION_FACTOR", 4)
    actor_kwargs = dict(
        hidden_dim=config.get("SIMBA_ACTOR_HIDDEN_DIM", 128),
        num_blocks=config.get("SIMBA_ACTOR_NUM_BLOCKS", 1),
        expansion_factor=expansion_factor,
    )
    critic_kwargs = dict(
        hidden_dim=config.get("SIMBA_CRITIC_HIDDEN_DIM", 512),
        num_blocks=config.get("SIMBA_CRITIC_NUM_BLOCKS", 2),
        expansion_factor=expansion_factor,
    )
    if config.get("PIXEL_BASED", True):
        Actor = partial(SimBa_Pixel_Actor_Discrete, **actor_kwargs)
        Critic = partial(SimBa_Pixel_Critic, **critic_kwargs)
    else:
        Actor = partial(SimBa_MLP_Actor_Discrete, **actor_kwargs)
        Critic = partial(SimBa_MLP_Critic, **critic_kwargs)
    actor_net = Actor(action_dim=action_dim)
    critic_net = Critic(action_dim=action_dim)

    dummy_obs = jnp.zeros((1, *obs_shape))

    actor_state = TrainState.create(
        apply_fn=actor_net.apply,
        params=actor_net.init(actor_key, dummy_obs, actor_key2),
        tx=optax.adam(learning_rate=config.get("LEARNING_RATE", 3e-4), eps=1e-4),
    )
    qf1_state = SACTrainState.create(
        apply_fn=critic_net.apply,
        params=critic_net.init(qf1_key, dummy_obs),
        target_params=critic_net.init(qf1_key, dummy_obs),
        tx=optax.adam(learning_rate=config.get("LEARNING_RATE", 3e-4), eps=1e-4),
    )
    qf2_state = SACTrainState.create(
        apply_fn=critic_net.apply,
        params=critic_net.init(qf2_key, dummy_obs),
        target_params=critic_net.init(qf2_key, dummy_obs),
        tx=optax.adam(learning_rate=config.get("LEARNING_RATE", 3e-4), eps=1e-4),
    )

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
    key, reset_key = jax.random.split(key)
    _obs, _state = vmap_reset(jax.random.split(reset_key, num_envs))
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


    def full_SAC_step(actor_state, qf1_state, qf2_state, log_alpha, a_opt_state, buffer_state, env_state, obs, rng, global_step):
        
        def take_action(carry, _):
            actor_state, buffer_state, env_state, obs, global_step, rng = carry
            rng, action_rng, actor_sample_key = jax.random.split(rng, 3)
            action_sample_keys = jax.random.split(action_rng, num_envs)

            random_actions = jax.vmap(env.action_space().sample)(action_sample_keys)
            samples, _, _ = actor_state.apply_fn(actor_state.params, obs, actor_sample_key)

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
            u_actor_state, u_qf1_state, u_qf2_state, log_alpha, a_opt_state, u_key = update_carry
            u_key, sample_key, sample_key2, sample_key3 = jax.random.split(u_key, 4)
            alpha = jnp.exp(log_alpha) if config.get("AUTOTUNE", True) else config.get("ALPHA", 0.2)

            batch = replay_buffer.sample(buffer_state, sample_key).experience
            b_obs = batch.first.obs
            b_act = batch.first.action
            b_rew = batch.first.reward
            b_don = batch.first.done
            b_nobs = batch.second.obs

            _, next_state_log_pi, next_state_action_probs = u_actor_state.apply_fn(u_actor_state.params, b_nobs, sample_key2)

            q1_next_target = u_qf1_state.apply_fn(u_qf1_state.target_params, b_nobs)
            q2_next_target = u_qf2_state.apply_fn(u_qf2_state.target_params, b_nobs)
            min_q_next_target = next_state_action_probs * (jnp.minimum(q1_next_target, q2_next_target) - alpha * next_state_log_pi)
            min_q_next_target = jnp.sum(min_q_next_target, axis=1)
            next_q_value = (b_rew.flatten() + (1.0 - b_don.flatten()) * gamma * min_q_next_target)
            next_q_value = jax.lax.stop_gradient(next_q_value)

            def qf_loss_fn(qf1_params, qf2_params, qf1_state, qf2_state):
                qf1_pred = qf1_state.apply_fn(qf1_params, b_obs)
                qf2_pred = qf2_state.apply_fn(qf2_params, b_obs)
                qf1_pred_a_values = jax.vmap(lambda q, a: q[a])(qf1_pred, b_act)
                qf2_pred_a_values = jax.vmap(lambda q, a: q[a])(qf2_pred, b_act)
                qf1_loss = jnp.mean((qf1_pred_a_values - next_q_value) ** 2)
                qf2_loss = jnp.mean((qf2_pred_a_values - next_q_value) ** 2)
                return qf1_loss + qf2_loss, (qf1_pred, qf2_pred, qf1_pred_a_values.mean(), qf2_pred_a_values.mean())

            (qf_loss, (qf1_pred, qf2_pred, qf1_pred_a_values, qf2_pred_a_values)), grads = jax.value_and_grad(qf_loss_fn, argnums=(0,1),has_aux=True)(u_qf1_state.params, u_qf2_state.params, u_qf1_state, u_qf2_state)
            qf1_grads, qf2_grads = grads

            new_qf1_state = u_qf1_state.apply_gradients(grads=qf1_grads)
            new_qf2_state = u_qf2_state.apply_gradients(grads=qf2_grads)

            def actor_loss_fn(actor_params, actor_state):
                new_qf1_pred = new_qf1_state.apply_fn(new_qf1_state.params, b_obs)
                new_qf2_pred = new_qf2_state.apply_fn(new_qf2_state.params, b_obs)
                _, log_pi, action_probs = actor_state.apply_fn(actor_params, b_obs, sample_key3)
                min_qf_values = jax.lax.stop_gradient(jnp.minimum(new_qf1_pred, new_qf2_pred))
                actor_loss = jnp.sum((action_probs * ((alpha * log_pi) - min_qf_values)), axis=-1).mean()
                return actor_loss, (log_pi, action_probs)
            
            (actor_loss, (log_pi, action_probs)), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(u_actor_state.params, u_actor_state)
            new_actor_state = u_actor_state.apply_gradients(grads=actor_grads)
            
        
            if config.get("AUTOTUNE", True):
                def alpha_loss_fn(log_alpha):
                    action_probs_detached = jax.lax.stop_gradient(action_probs)
                    entropy_diff = jax.lax.stop_gradient(log_pi + target_entropy)
                    return jnp.mean(jnp.sum((action_probs_detached * (-jnp.exp(log_alpha) * entropy_diff)), axis=-1))

                _, alpha_grad = jax.value_and_grad(alpha_loss_fn)(log_alpha)
                updates, a_opt_state = a_optimizer.update(alpha_grad, a_opt_state, log_alpha)
                log_alpha = optax.apply_updates(log_alpha, updates)

            policy_entropy = -jnp.sum(action_probs * log_pi, axis=-1).mean()
            q_abs_max = jnp.maximum(jnp.max(jnp.abs(qf1_pred)), jnp.max(jnp.abs(qf2_pred)))

            return (new_actor_state, new_qf1_state, new_qf2_state, log_alpha, a_opt_state, u_key), (qf_loss, actor_loss, qf1_pred_a_values, alpha, policy_entropy, q_abs_max)

    
        def scanned_update(carry):
            carry, metrics = jax.lax.scan(do_update, carry, None, length=gradient_steps)
            qf_l, act_l, qf1_v, a_val, p_ent, q_max = metrics
            return carry, (qf_l[-1], act_l[-1], qf1_v[-1], a_val[-1].squeeze(), p_ent[-1], q_max[-1])

        (actor_state, qf1_state, qf2_state, log_alpha, a_opt_state, rng), (qf_loss, actor_loss, qf1_val, alpha, policy_entropy, q_abs_max) = jax.lax.cond(
            replay_buffer.can_sample(buffer_state),
            lambda c: scanned_update(c),
            lambda c: (c, (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))),
            (actor_state, qf1_state, qf2_state, log_alpha, a_opt_state, rng), 
        )

        def update_target_networks(c):
            c_qf1, c_qf2 = c
            updated_qf1 = c_qf1.replace(
                target_params=optax.incremental_update(c_qf1.params, c_qf1.target_params, tau)
            )
            updated_qf2 = c_qf2.replace(
                target_params=optax.incremental_update(c_qf2.params, c_qf2.target_params, tau)
            )
            return updated_qf1, updated_qf2
        
        update_target_flag = jnp.logical_and(
            replay_buffer.can_sample(buffer_state),
            (global_step % config.get("TARGET_UPDATE_FREQUENCY", 8000)) < steps_per_update
        )

        qf1_state, qf2_state = jax.lax.cond(
            update_target_flag,
            update_target_networks, 
            lambda c: c,
            (qf1_state, qf2_state)
            )

        return (actor_state, qf1_state, qf2_state, log_alpha, a_opt_state, buffer_state, next_env_state, next_obs, rng, global_step), (infos, qf_loss, actor_loss, qf1_val, alpha, policy_entropy, q_abs_max)

    def save_and_eval(step_count):
        if config.get("SAVE_PATH", "./models") is not None:
            model_path = f'{config.get("SAVE_PATH", "./models")}/{run_name}/{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(
                    flax.serialization.to_bytes(
                        [
                            config,
                            sac_carry[0].params,
                            sac_carry[1].params,
                            sac_carry[2].params
                         ]
                    )
                )
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
                Model=(Actor, Critic),
                seed=config["SEED"]+42,
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

    print(f"[simba_sac] target_entropy {float(target_entropy):.4f} "
          f"(max {float(jnp.log(action_dim)):.4f} for {action_dim} actions, "
          f"ratio {float(target_entropy / jnp.log(action_dim)):.2f})")
    print(f"[simba_sac] start compile...")
    start_compile = time.perf_counter()
    global_step = jnp.array(0, dtype=jnp.int32)
    sac_carry = (actor_state, qf1_state, qf2_state, log_alpha, a_opt_state, buffer_state, _state, _obs, key, global_step)

    @jax.jit
    def scanned_steps(carry):
        def step_fn(c, _):
            return full_SAC_step(*c)
        return jax.lax.scan(step_fn, carry, None, length=config.get("SCAN_STEPS", 1000))

    _ = jax.block_until_ready(scanned_steps(sac_carry))
    end_compile = time.perf_counter()
    print(f"[simba_sac] compilation time: {end_compile - start_compile:.2f}s")
    
    steps_per_iteration = config.get("NUM_ENVS") * config.get("TRAIN_FREQUENCY") * config.get("SCAN_STEPS")
    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=config.get("TOTAL_TIMESTEPS") // steps_per_iteration)
    rtpt.start()
    run_time = time.perf_counter()
    
    print(f"[simba_sac] starting training for {config.get('TOTAL_TIMESTEPS')} steps...")
    while global_step < config.get("TOTAL_TIMESTEPS"):
        rtpt.step()
        iteration = global_step // steps_per_iteration
        if config["EVAL_DURING_TRAIN"] and iteration > 0 and iteration % config["EVAL_EVERY"] == 0:
           save_and_eval(global_step) 
           
        iteration_time_start = time.perf_counter()
        result = scanned_steps(sac_carry)
        sac_carry, (infos, qf_loss, actor_loss, qf1_val, alpha, pol_ent, q_abs_max) = result
        global_step = int(sac_carry[-1])
        
        print(
            f"[simba_sac] iteration {iteration} | step {global_step}"
            f" | avg_return {infos['returned_episode_returns'][-1].mean():.2f}"
            f" | qf_loss {qf_loss[-1]:.4f} | act_loss {actor_loss[-1]:.4f}"
            f" | alpha {alpha[-1]:.4f} | entropy {pol_ent[-1]:.4f} | q_max {q_abs_max[-1]:.2f}"
            f" | SPS {int(global_step / (time.perf_counter() - run_time))}"
        )
        
        metrics = {
            "charts/avg_episodic_return": infos["returned_episode_returns"][-1].mean(), 
            "charts/avg_episodic_length": infos["returned_episode_lengths"][-1].mean(),
            "losses/qf_loss": qf_loss[-1].item(),
            "losses/actor_loss": actor_loss[-1].item(),
            "losses/qf1_values": qf1_val[-1].item(),
            "losses/alpha_values": alpha[-1].item(),
            "losses/policy_entropy": pol_ent[-1].item(),
            "losses/q_abs_max": q_abs_max[-1].item(),
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(config["NUM_ENVS"] * config["TRAIN_FREQUENCY"] * config["SCAN_STEPS"]  / (time.perf_counter() - iteration_time_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }
        wandb.log(metrics, step=global_step)

    eval_metrics = save_and_eval(global_step+1)
    wandb.finish()
    return eval_metrics
