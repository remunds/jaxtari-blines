# Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari_jax.py
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
    LogWrapper,
    FlattenObservationWrapper,
    LogWrapper
)
from agents.dqn.dqn_eval import evaluate
from agents.run_utils import eval_mod_configs, plan_chunks
from rtpt import RTPT


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

class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
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
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

class MLP_QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(461, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return x

class DQNTrainState(TrainState):
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
        mode=config.get("WANDB_MODE", "online"),
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
    batch_size = config.get("BATCH_SIZE", 32)
    total_timesteps = config.get("TOTAL_TIMESTEPS", 10000000)

    key, q_key = jax.random.split(key, 2)
    network = QNetwork(action_dim=action_dim) if config.get("PIXEL_BASED", True) else MLP_QNetwork(action_dim=action_dim)

    dummy_obs = jnp.zeros((1, *obs_shape))
    q_params = network.init(q_key, dummy_obs)

    tx = optax.adam(learning_rate=config.get("LEARNING_RATE"), eps=1e-4)

    agent_state = DQNTrainState.create(
        apply_fn=network.apply,
        params=q_params,
        target_params=jax.tree.map(jnp.copy, q_params),
        tx=tx,
    )

    replay_buffer = fbx.make_prioritised_flat_buffer(
        max_length=config.get("BUFFER_SIZE", 1000000),
        min_length=config.get("LEARNING_STARTS", 80000),
        sample_batch_size=config.get("BATCH_SIZE", 32),
        add_sequences=False,
        add_batch_size=config["NUM_ENVS"],
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

    def full_dqn_step(agent_state, buffer_state, env_state, obs, rng, global_step):
        def take_action(carry, _):
            agent_state, buffer_state, env_state, obs, global_step, rng = carry

            rng, action_rng, explore_rng = jax.random.split(rng, 3)
            epsilon = jnp.interp(
                global_step,
                jnp.array([0, config.get("EXPLORATION_FRACTION", 0.10) * total_timesteps]),
                jnp.array([config.get("START_E", 1.0), config.get("END_E", 0.05)]),
            )

            q_values = agent_state.apply_fn(agent_state.params, obs)
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
            #TODO: Could move adding outside of scan (and use sequential add from fbx)
            buffer_state = replay_buffer.add(buffer_state, timestep)
            return (agent_state, buffer_state, next_env_state, next_obs, global_step + num_envs, rng), info

        # take TRAIN_FREQUENCY steps in one go
        (agent_state, buffer_state, next_env_state, next_obs, global_step, rng), infos = jax.lax.scan(
            take_action,
            (agent_state, buffer_state, env_state, obs, global_step, rng), 
            None,
            length=config.get("TRAIN_FREQUENCY", 4),
        )

        def do_update(update_carry, _):
            u_state, u_key = update_carry
            u_key, sample_key = jax.random.split(u_key)

            batch = replay_buffer.sample(buffer_state, sample_key).experience
            b_obs = batch.first.obs
            b_act = batch.first.action
            b_rew = batch.first.reward
            b_don = batch.first.done
            b_nobs = batch.second.obs

            def q_loss_fn(params):
                q_pred = u_state.apply_fn(params, b_obs)
                q_pred = q_pred[jnp.arange(batch_size), b_act.reshape(-1)]

                q_next = u_state.apply_fn(u_state.target_params, b_nobs)
                target = jax.lax.stop_gradient(
                    b_rew + (1.0 - b_don) * gamma * q_next.max(axis=-1)
                )
                error = q_pred - target
                loss = jnp.mean(error ** 2)
                return loss, q_pred.mean()

            (loss, q_val), grads = jax.value_and_grad(q_loss_fn, has_aux=True)(u_state.params)
            new_state = u_state.apply_gradients(grads=grads)

            return (new_state, u_key), (loss, q_val)
        
        def scanned_update(carry):
            # take gradient_steps in one go, if -1: we do as many gradient steps as collected samples
            carry, (loss, qval) = jax.lax.scan(do_update, carry, None, length=gradient_steps)
            return carry, (loss[-1], qval[-1])

        
        # train NN if we have enough samples in the replay buffer (==learning_starts) 
        (agent_state, rng), (loss, q_val) = jax.lax.cond(
            replay_buffer.can_sample(buffer_state),
            lambda c: scanned_update(c),
            lambda c: (c, (jnp.array(0.0), jnp.array(0.0))),
            (agent_state, rng),
        )
        steps_per_update = config.get("TRAIN_FREQUENCY", 4) * config.get("NUM_ENVS", 1)
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

    def save_and_eval(q_params, step_count):
        if config.get("SAVE_PATH") is not None:
            save_dir = os.path.join(config["SAVE_PATH"], run_name)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model')
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes([config, q_params]))
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")
        metrics = {}
        for mods_cfg, mod_label in eval_mod_configs(config):
            episodic_returns, env_states = evaluate(
                q_params,
                partial(
                    make_env,
                    mods=mods_cfg,
                    pixel_based=config["PIXEL_BASED"],
                    native_downscaling=config["NATIVE_DOWNSCALING"],
                    eval=True,
                ),
                config["ENV_ID"],
                eval_episodes=10,
                Model=QNetwork if config["PIXEL_BASED"] else MLP_QNetwork,
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
        return metrics

    # same budget rule for every agent: whole jitted chunks within TOTAL_TIMESTEPS
    scan_steps, num_chunks, steps_per_chunk = plan_chunks(
        config["TOTAL_TIMESTEPS"],
        config["NUM_ENVS"] * config.get("TRAIN_FREQUENCY", 4),
        config.get("SCAN_STEPS", 1000),
    )

    print(f"[dqn] start compile...")
    start_compile = time.perf_counter()
    global_step = jnp.array(0, dtype=jnp.int32)
    carry = (agent_state, buffer_state, _state, _obs, key, global_step)

    def scanned_steps(carry):
        def step_fn(c, _):
            return full_dqn_step(*c)
        return jax.lax.scan(step_fn, carry, None, length=scan_steps)

    # donate the carry so XLA reuses the replay buffer's memory; lower/compile AOT so nothing is donated before the loop
    compiled = jax.jit(scanned_steps, donate_argnums=(0,)).lower(carry).compile()
    print(f"[dqn] compilation time: {time.perf_counter() - start_compile:.2f}s")

    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=num_chunks)
    rtpt.start()
    run_time = time.perf_counter()
    print(f"[dqn] starting training: {num_chunks} chunks x {steps_per_chunk} env steps")
    for chunk in range(num_chunks):
        rtpt.step()
        if config["EVAL_DURING_TRAIN"] and chunk > 0 and chunk % config["EVAL_EVERY"] == 0:
            save_and_eval(carry[0].params, int(carry[-1]))

        chunk_start = time.perf_counter()
        carry, (infos, loss, q_val) = compiled(carry)
        global_step = int(carry[-1])
        metrics = {
            "charts/avg_episodic_return": float(infos["returned_episode_returns"][-1].mean()),
            "charts/avg_episodic_length": float(infos["returned_episode_lengths"][-1].mean()),
            "losses/td_loss": loss[-1].item(),
            "losses/q_values": q_val[-1].item(),
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(steps_per_chunk / (time.perf_counter() - chunk_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }
        wandb.log(metrics, step=global_step)
        print(
            f"[dqn] chunk {chunk + 1}/{num_chunks} | global_step {global_step} | "
            f"avg_return {metrics['charts/avg_episodic_return']:.2f} | "
            f"avg_length {metrics['charts/avg_episodic_length']:.2f} | "
            f"td_loss {metrics['losses/td_loss']:.4f} | q_val {metrics['losses/q_values']:.4f} | "
            f"SPS {metrics['charts/SPS']}"
        )

    eval_metrics = save_and_eval(carry[0].params, global_step + 1)
    wandb.finish()
    return eval_metrics
