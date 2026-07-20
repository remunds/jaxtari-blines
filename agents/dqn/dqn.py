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
import jaxatari
from jaxatari.wrappers import (
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    AtariWrapper,
    LogWrapper,
    FlattenObservationWrapper,
    LogWrapper
)
# import tqdx as _tqdx
from agents.dqn.dqn_eval import evaluate
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

# def build_eval_fn(env, apply_fn, eval_episodes, max_steps, action_dim):

#     def wrapped_reset(key):
#         next_obs, state = env.reset(key)
#         return next_obs.squeeze()[None, ...], state

#     def wrapped_step(state, action):
#         next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
#         done = jnp.logical_or(terminated, truncated)
#         return next_obs.squeeze()[None, ...], next_state, reward, done, info

#     def get_action(params, obs, key, epsilon):
#         q_values = apply_fn(params, obs)
#         greedy_action = jnp.argmax(q_values, axis=1)

#         key, subkey = jax.random.split(key)
#         random_action = jax.random.randint(subkey, greedy_action.shape, 0, action_dim)
#         explore = jax.random.uniform(key, greedy_action.shape) < epsilon
#         action = jnp.where(explore, random_action, greedy_action)
#         return action, key

#     def step_fn(carry, _):
#         obs, env_state, keys, params, epsilon = carry
#         actions, keys = jax.vmap(get_action, in_axes=(None, 0, 0, None))(params, obs, keys, epsilon)
#         next_obs, next_env_state, reward, done, info = jax.vmap(wrapped_step)(env_state, actions)
#         first_state = jax.tree.map(lambda x: x[0], next_env_state)
#         return (next_obs, next_env_state, keys, params, epsilon), (first_state, done, reward)

#     @jax.jit
#     def eval_fn(params, reset_keys, epsilon):
#         obs, env_state = jax.vmap(wrapped_reset)(reset_keys)
#         _, (first_states_history, dones, rewards) = jax.lax.scan(
#             step_fn, (obs, env_state, reset_keys, params, epsilon), None, length=max_steps
#         )
#         has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
#         mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
#         masked_rewards = rewards * (1 - mask_after_first_done)
#         episodic_returns = jnp.sum(masked_rewards, axis=0)
#         first_done = jnp.argmax(dones, axis=0)
#         return episodic_returns, first_states_history, first_done

#     return eval_fn


# def build_eval_return_fn(env, apply_fn, max_steps):

#     def wrapped_reset(key):
#         obs, state = env.reset(key)
#         return obs.squeeze()[None, ...], state

#     def wrapped_step(state, action):
#         obs, state, reward, terminated, truncated, info = env.step(state, action.squeeze())
#         done = jnp.logical_or(terminated, truncated)
#         return obs.squeeze()[None, ...], state, reward, done

#     def get_action(params, obs):
#         q_values = apply_fn(params, obs)
#         return jnp.argmax(q_values, axis=1)

#     def step_fn(carry, _):
#         obs, state, params = carry
#         actions = jax.vmap(get_action, in_axes=(None, 0))(params, obs)
#         obs, state, reward, done = jax.vmap(wrapped_step)(state, actions)
#         return (obs, state, params), (done, reward)

#     def eval_return_fn(params, reset_keys):
#         obs, state = jax.vmap(wrapped_reset)(reset_keys)
#         _, (dones, rewards) = jax.lax.scan(
#             step_fn, (obs, state, params), None, length=max_steps
#         )
#         has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
#         mask = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
#         masked = rewards * (1 - mask)
#         return jnp.mean(jnp.sum(masked, axis=0))

#     return eval_return_fn

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

    obs_dtype = jnp.uint8 if config.get("PIXEL_BASED", True) else jnp.float32
    # replay_buffer = fbx.make_item_buffer(
    #     max_length=config.get("BUFFER_SIZE", 1000000),
    #     min_length=config.get("LEARNING_STARTS", 80000),
    #     sample_batch_size=batch_size,
    #     add_batches=True,
    # )
    # example_transition = {
    #     "obs": jnp.zeros(obs_shape, dtype=obs_dtype),
    #     "action": jnp.zeros((), dtype=jnp.int32),
    #     "reward": jnp.zeros((), dtype=jnp.float32),
    #     "done": jnp.zeros((), dtype=jnp.bool_),
    #     "next_obs": jnp.zeros(obs_shape, dtype=obs_dtype),
    # }
    # buffer_state = replay_buffer.init(example_transition)

    replay_buffer = fbx.make_flat_buffer(
        max_length=config.get("BUFFER_SIZE", 1000000),
        min_length=config.get("LEARNING_STARTS", 80000),
        sample_batch_size=config.get("BATCH_SIZE", 32),
        # add_batches=True,
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
        # info=_info
    )
    buffer_state = replay_buffer.init(_dummy_step)

    # eval_mods_list = list(config.get("EVAL_MODS", [])) or list(config.get("TRAIN_MODS", []))
    # eval_configs = [([], "default")]
    # for mod in eval_mods_list:
    #     mods_cfg = list(mod) if isinstance(mod, (list, tuple)) else [mod]
    #     mod_label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods_cfg)
    #     eval_configs.append((mods_cfg, mod_label))

    # eval_episodes = 10
    # eval_max_steps = 10000

    # eval_fns = {}
    # for mods_cfg, mod_label in eval_configs:
    #     eval_env = make_env(
    #         config["ENV_ID"],
    #         mods=mods_cfg,
    #         pixel_based=config.get("PIXEL_BASED", True),
    #         native_downscaling=config.get("NATIVE_DOWNSCALING", True),
    #         eval=True,
    #     )()
    #     eval_fns[mod_label] = build_eval_fn(
    #         env=eval_env,
    #         apply_fn=network.apply,
    #         eval_episodes=eval_episodes,
    #         max_steps=eval_max_steps,
    #         action_dim=action_dim,
    #     )

    # inscan_eval_env = make_env(
    #     config["ENV_ID"],
    #     mods=train_mods,
    #     pixel_based=config.get("PIXEL_BASED", True),
    #     native_downscaling=config.get("NATIVE_DOWNSCALING", True),
    #     eval=True,
    # )()
    # inscan_eval_fn = build_eval_return_fn(inscan_eval_env, network.apply, eval_max_steps)
    # eval_reset_keys = jax.random.split(jax.random.PRNGKey(config["SEED"]), eval_episodes)

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
                # info=infos,
            )
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

        # train NN if we have enough samples in the replay buffer 
        (agent_state, rng), (loss, q_val) = jax.lax.cond(
            replay_buffer.can_sample(buffer_state),
            lambda c: do_update(c, None),
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

    def save_and_eval(step_count):
        if config.get("SAVE_PATH", "./models") is not None:
            model_path = f'{config.get("SAVE_PATH", "./models")}/{run_name}/{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(
                    flax.serialization.to_bytes(
                        [
                            config,
                            dqn_carry[0].params
                         ]
                    )
                )
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")

        # evaluate across all mods (and default train env)
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
                Model=QNetwork if config["PIXEL_BASED"] else MLP_QNetwork,
                seed=config["SEED"]+42, # use a different seed for evaluation 
            )
            metrics[mod_label] = np.mean(jax.device_get(episodic_returns))
            wandb.log({f"eval/episodic_return_{mod_label}": np.mean(jax.device_get(episodic_returns))}, step=step_count)

            if config["CAPTURE_VIDEO"]: 
                # Instantiate a clean renderer immune to the training env's downscaling
                clean_renderer = jaxatari.make(config["ENV_ID"], mods=mods_cfg).renderer
                frames = jax.vmap(clean_renderer.render)(env_states)
                # shape: (N, H, W, C) -> (N, C, H, W)
                frames = jnp.transpose(frames, (0, 3, 1, 2))
                video = wandb.Video(np.array(frames), fps=30, format="mp4")
                wandb.log(
                    {
                        f"eval/video_{mod_label}": video,
                    },
                    step=step_count,
                )
                print(f"Video (eval) logged to wandb with {frames.shape[0]} frames ({mod_label}).")
        return metrics

    # CHUNK_SIZE = config["NUM_STEPS"] // num_envs
    # total_iterations = total_timesteps // (num_envs * CHUNK_SIZE)
    # eval_every = config.get("EVAL_EVERY", 10)
    # eval_during_train = config.get("EVAL_DURING_TRAIN", True)

    # steps_per_chunk = num_envs * CHUNK_SIZE
    # _timing = {"start": None, "start_step": 0, "last": None}

    # def log_cb(m):
    #     now = time.time()
    #     step = int(m["charts/global_step"])
    #     d = {k: float(v) for k, v in m.items()}
    #     d["charts/global_step"] = step
    #     if _timing["start"] is None:
    #         _timing.update(start=now, start_step=step, last=now)
    #     else:
    #         dt = now - _timing["last"]
    #         elapsed = now - _timing["start"]
    #         d["charts/SPS_update"] = int(steps_per_chunk / dt) if dt > 0 else 0
    #         d["charts/SPS"] = int((step - _timing["start_step"]) / elapsed) if elapsed > 0 else 0
    #         _timing["last"] = now
    #     wandb.log(d, step=step)

    # def outer_step(carry, i):
    #     base_carry, last_eval = carry
    #     base_carry, (losses, epsilons) = jax.lax.scan(step_once, base_carry, None, length=CHUNK_SIZE)
    #     state, buffer_state, env_state, obs, rng, global_step, ep_stats = base_carry

    #     if eval_during_train:
    #         eval_return = jax.lax.cond(
    #             (i % eval_every) == 0,
    #             lambda p: inscan_eval_fn(p, eval_reset_keys),
    #             lambda p: last_eval,
    #             state.params,
    #         )
    #     else:
    #         eval_return = last_eval

    #     avg_loss = jnp.sum(losses) / jnp.maximum(jnp.sum(losses != 0), 1)
    #     metrics = {
    #         "charts/global_step": global_step,
    #         "charts/avg_episodic_return": ep_stats.returned_episode_returns.mean(),
    #         "charts/avg_episodic_length": ep_stats.returned_episode_lengths.mean().astype(jnp.float32),
    #         "charts/epsilon": epsilons[-1],
    #         "losses/td_loss": avg_loss,
    #     }
    #     if eval_during_train:
    #         metrics[f"eval/episodic_return_{train_label}"] = eval_return
    #     jax.debug.callback(log_cb, metrics)

    #     return (base_carry, eval_return), None

    # key, reset_key = jax.random.split(key)
    # obs, env_state = vmap_reset(jax.random.split(reset_key, num_envs))
    # global_step = jnp.array(0, dtype=jnp.int32)

    # base_carry = (agent_state, buffer_state, env_state, obs, key, global_step, episode_stats)

    # @partial(jax.jit, donate_argnums=(0,))
    # def train(base_carry):
    #     if eval_during_train:
    #         init_eval = inscan_eval_fn(base_carry[0].params, eval_reset_keys)
    #     else:
    #         init_eval = jnp.float32(0.0)
    #     carry, _ = _tqdx.scan(outer_step, (base_carry, init_eval), jnp.arange(1, total_iterations + 1))
    #     return carry

    # (base_carry, _last_eval) = jax.block_until_ready(train(base_carry))
    # wall = time.time() - start_time

    # agent_state = base_carry[0]
    # total_steps = int(base_carry[5])
    # print(f"[dqn] {total_steps} steps in {wall:.1f}s incl. compile -> {int(total_steps / wall)} SPS (compile-inclusive)")

    # eval_metrics = save_and_eval(total_steps, agent_state)
    # wandb.finish()
    # return eval_metrics

    # we step n_envs each iteration
    print(f"[dqn] start compile...")
    start_compile = time.perf_counter()
    global_step = jnp.array(0, dtype=jnp.int32)
    dqn_carry = (agent_state, buffer_state, _state, _obs, key, global_step)

    @jax.jit
    def scanned_steps(carry):
        def step_fn(c, _):
            return full_dqn_step(*c)
        return jax.lax.scan(step_fn, carry, None, length=config.get("SCAN_STEPS", 1000))

    # warmup to trigger compilation
    _ = jax.block_until_ready(scanned_steps(dqn_carry))
    end_compile = time.perf_counter()
    print(f"[dqn] compilation time: {end_compile - start_compile:.2f}s")
    steps_per_iteration = config.get("NUM_ENVS") * config.get("TRAIN_FREQUENCY") * config.get("SCAN_STEPS")
    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=config.get("TOTAL_TIMESTEPS") // steps_per_iteration)
    rtpt.start()
    run_time = time.perf_counter()
    print(f"[dqn] starting training for {config.get('TOTAL_TIMESTEPS')} steps...")
    while global_step < config.get("TOTAL_TIMESTEPS"):
        rtpt.step()
        iteration = global_step // steps_per_iteration
        if config["EVAL_DURING_TRAIN"] and iteration > 0 and iteration % config["EVAL_EVERY"] == 0:
           save_and_eval(global_step) 
        iteration_time_start = time.perf_counter()
        result = scanned_steps(dqn_carry)
        # result = jax.block_until_ready(compiled(*dqn_carry))
        dqn_carry, (infos, loss, q_val) = result
        global_step = int(dqn_carry[-1])
        print(f"[dqn] iteration {iteration} | global_step {global_step} | avg_return {infos['returned_episode_returns'][-1].mean():.2f} | avg_length {infos['returned_episode_lengths'][-1].mean():.2f} | td_loss {loss[-1]:.4f} | q_val {q_val[-1]:.4f} | SPS {int(global_step / (time.perf_counter() - run_time))} | SPS_update {int(config['NUM_ENVS'] * config['TRAIN_FREQUENCY'] * config['SCAN_STEPS'] / (time.perf_counter() - iteration_time_start))}")
        metrics = {
            "charts/avg_episodic_return": infos["returned_episode_returns"][-1].mean(), 
            "charts/avg_episodic_length": infos["returned_episode_lengths"][-1].mean(),
            "losses/td_loss": loss[-1].item(),
            "losses/q_values": q_val[-1].item(), 
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(config["NUM_ENVS"] * config["TRAIN_FREQUENCY"] * config["SCAN_STEPS"]  / (time.perf_counter() - iteration_time_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }
        wandb.log(metrics, step=global_step)

    eval_metrics = save_and_eval(global_step+1)
    return eval_metrics


