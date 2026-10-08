# Adapted from https://github.com/mttga/purejaxql
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
from agents.pqn.pqn_eval import evaluate
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
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        return x

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
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return x

@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    rewards: jnp.array
    dones: jnp.array
    values: jnp.array
    returns: jnp.array

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
    q_lambda_val = config.get("Q_LAMBDA", 0.65)
    num_steps = config.get("NUM_STEPS", 32)
    num_minibatches = config.get("NUM_MINIBATCHES", 4)
    update_epochs = config.get("UPDATE_EPOCHS", 4)
    max_grad_norm = config.get("MAX_GRAD_NORM", 10.0)
    total_timesteps = config.get("TOTAL_TIMESTEPS", 10000000)
    start_e = config.get("START_E", 1.0)
    end_e = config.get("END_E", 0.01)
    exploration_fraction = config.get("EXPLORATION_FRACTION", 0.10)

    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_iterations = total_timesteps // batch_size
    exploration_steps = exploration_fraction * total_timesteps

    key, q_key = jax.random.split(key, 2)
    network = QNetwork(action_dim=action_dim) if config.get("PIXEL_BASED", True) else MLP_QNetwork(action_dim=action_dim)

    dummy_obs = jnp.zeros((1, *obs_shape))
    q_params = network.init(q_key, dummy_obs)

    anneal_lr = config.get("ANNEAL_LR", True)
    learning_rate = config.get("LEARNING_RATE", 2.5e-4)
    total_grad_steps = num_iterations * update_epochs * num_minibatches
    lr = optax.linear_schedule(learning_rate, 0.0, total_grad_steps) if anneal_lr else learning_rate
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.radam(lr))

    q_state = TrainState.create(
        apply_fn=network.apply,
        params=q_params,
        tx=tx,
    )

    key, reset_key = jax.random.split(key)
    _obs, _state = vmap_reset(jax.random.split(reset_key, num_envs))
    _done = jnp.zeros(num_envs, dtype=jnp.float32)

    def full_pqn_step(q_state, env_state, obs, last_done, rng, global_step):
        def step_once(carry, _):
            q_params, env_state, last_obs, last_done, key, global_step = carry

            #eps greedy
            epsilon = jnp.maximum(
                end_e,
                start_e + (end_e - start_e) * global_step.astype(jnp.float32) / exploration_steps,
            )

            #forward pass values for qlambda
            q_vals = network.apply(q_params, last_obs)
            max_actions = jnp.argmax(q_vals, axis=-1)
            max_vals = q_vals[jnp.arange(num_envs), max_actions]

            #action selection
            key, act_key, exp_key = jax.random.split(key, 3)
            rnd = jax.random.randint(act_key, (num_envs,), 0, action_dim)
            explore = jax.random.uniform(exp_key, (num_envs,)) < epsilon
            actions = jnp.where(explore, rnd, max_actions)

            #vectorized one step of all envs
            next_obs, new_states, rewards, next_done, info = vmap_step(env_state, actions)
            done = next_done.astype(jnp.float32)

            #store
            storage = Storage(
                obs=last_obs, actions=actions, rewards=rewards,
                dones=last_done, values=max_vals,
                returns=jnp.zeros_like(rewards),
            )
            new_carry = (q_params, new_states, next_obs, done, key, global_step + num_envs)
            return new_carry, (storage, info)
        #rollout loop, stack all stored values for each step
        (q_params, env_state, next_obs, next_done, rng, global_step), (storage, infos) = jax.lax.scan(
            step_once,
            (q_state.params, env_state, obs, last_done, rng, global_step),
            None,
            length=num_steps,
        )

        # recursive q lamda update
        def compute_q_lambda_once(carry, inp):
            next_return = carry
            reward, next_val, nd = inp
            ret = reward + gamma * (q_lambda_val * next_return + (1.0 - q_lambda_val) * next_val) * (1.0 - nd)
            return ret, ret

        next_q = network.apply(q_state.params, next_obs)
        next_val = jnp.max(next_q, axis=-1)
        next_values_t = jnp.concatenate([storage.values[1:], next_val[None]], axis=0)
        next_dones_t = jnp.concatenate([storage.dones[1:], next_done[None]], axis=0)
        _, returns = jax.lax.scan(
            compute_q_lambda_once, next_val,
            (storage.rewards, next_values_t, next_dones_t), reverse=True, #backward through time iteration
        )
        storage = storage.replace(returns=returns) #multi step storage ready for gradient updates

        def update_epoch(carry, _):
            q_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x): #turn into shuffled batches
                x = jax.random.permutation(subkey, x)
                return jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])

            flat = jax.tree_util.tree_map(flatten, storage)
            shuffled = jax.tree_util.tree_map(convert_data, flat)

            def update_minibatch(q_state, mb): #mb gradient descent
                def loss_fn(params):
                    q_vals = network.apply(params, mb.obs)
                    q_sel = q_vals[jnp.arange(minibatch_size), mb.actions]
                    return jnp.mean((mb.returns - q_sel) ** 2), q_sel.mean()
                (loss, q_val), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_state.params)
                q_state = q_state.apply_gradients(grads=grads)
                return q_state, (loss, q_val)

            q_state, (loss, q_val) = jax.lax.scan(update_minibatch, q_state, shuffled) #iterate mbgd updates in one epoch
            return (q_state, key), (loss, q_val)

        (q_state, rng), (loss, q_val) = jax.lax.scan(update_epoch, (q_state, rng), (), length=update_epochs) #run for #update epochs training

        return (q_state, env_state, next_obs, next_done, rng, global_step), (infos, loss[-1, -1], q_val[-1, -1])

    def save_and_eval(step_count):
        if config.get("SAVE_PATH", "./models") is not None:
            model_path = f'{config.get("SAVE_PATH", "./models")}/{run_name}/{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(
                    flax.serialization.to_bytes(
                        [
                            config,
                            pqn_carry[0].params
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

    # we step n_envs each iteration
    print(f"[pqn] start compile...")
    start_compile = time.perf_counter()
    global_step = jnp.array(0, dtype=jnp.int32)
    pqn_carry = (q_state, _state, _obs, _done, key, global_step)

    @jax.jit #jit full steps
    def scanned_steps(carry):
        def step_fn(c, _):
            return full_pqn_step(*c)
        return jax.lax.scan(step_fn, carry, None, length=config.get("SCAN_STEPS", 1000))

    # warmup to trigger compilation
    _ = jax.block_until_ready(scanned_steps(pqn_carry))
    end_compile = time.perf_counter()
    print(f"[pqn] compilation time: {end_compile - start_compile:.2f}s")
    steps_per_iteration = num_envs * num_steps * config.get("SCAN_STEPS", 1000)
    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=config.get("TOTAL_TIMESTEPS") // steps_per_iteration)
    rtpt.start()
    run_time = time.perf_counter()
    print(f"[pqn] starting training for {config.get('TOTAL_TIMESTEPS')} steps...")
    while global_step < config.get("TOTAL_TIMESTEPS"):
        rtpt.step()
        iteration = global_step // steps_per_iteration
        if config["EVAL_DURING_TRAIN"] and iteration > 0 and iteration % config["EVAL_EVERY"] == 0:
           save_and_eval(global_step)
        iteration_time_start = time.perf_counter()
        result = scanned_steps(pqn_carry)
        pqn_carry, (infos, loss, q_val) = result
        global_step = int(pqn_carry[-1])
        print(f"[pqn] iteration {iteration} | global_step {global_step} | avg_return {infos['returned_episode_returns'][-1].mean():.2f} | avg_length {infos['returned_episode_lengths'][-1].mean():.2f} | td_loss {loss[-1]:.4f} | q_val {q_val[-1]:.4f} | SPS {int(global_step / (time.perf_counter() - run_time))} | SPS_update {int(steps_per_iteration / (time.perf_counter() - iteration_time_start))}")
        metrics = {
            "charts/avg_episodic_return": infos["returned_episode_returns"][-1].mean(),
            "charts/avg_episodic_length": infos["returned_episode_lengths"][-1].mean(),
            "losses/td_loss": loss[-1].item(),
            "losses/q_values": q_val[-1].item(),
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(steps_per_iteration / (time.perf_counter() - iteration_time_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }
        wandb.log(metrics, step=global_step)

    eval_metrics = save_and_eval(global_step+1)
    wandb.finish()
    return eval_metrics
