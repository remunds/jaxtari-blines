# pqn + adaptive batch scaling 
import os
import random
import time
from collections import deque
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
from agents.apqn.apqn_eval import evaluate
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
    num_minibatches = config.get("NUM_MINIBATCHES", 4)
    update_epochs_base = config.get("UPDATE_EPOCHS", 2)  
    max_grad_norm = config.get("MAX_GRAD_NORM", 10.0)
    total_timesteps = config.get("TOTAL_TIMESTEPS", 10000000)
    start_e = config.get("START_E", 1.0)
    end_e = config.get("END_E", 0.001)
    exploration_fraction = config.get("EXPLORATION_FRACTION", 0.10)


    num_steps_min = config.get("NUM_STEPS_MIN", 16)
    num_steps_max = config.get("NUM_STEPS_MAX", 64)
    adapt_freq = config.get("ADAPT_FREQ", 50) 
    div_low = config.get("POLICY_CHANGE_LOW", 0.05) 
    div_high = config.get("POLICY_CHANGE_HIGH", 0.95)  
    ema_beta = config.get("EMA_BETA", 0.1)
    ref_batch_size = config.get("REF_BATCH_SIZE", 2048) 
    div_window = config.get("DIV_WINDOW", 10)
    burn_in_min_samples = div_window  

    num_steps = num_steps_min 

    exploration_steps = exploration_fraction * total_timesteps

    key, q_key = jax.random.split(key, 2)
    network = QNetwork(action_dim=action_dim) if config.get("PIXEL_BASED", True) else MLP_QNetwork(action_dim=action_dim)

    dummy_obs = jnp.zeros((1, *obs_shape))
    q_params = network.init(q_key, dummy_obs)

    anneal_lr = config.get("ANNEAL_LR", True)
    learning_rate = config.get("LEARNING_RATE", 2.5e-4)

    approx_num_iters = total_timesteps // (num_envs * num_steps_min)
    total_grad_steps = approx_num_iters * update_epochs_base * num_minibatches
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

    # ---- ABS jitted primitives ----
    @jax.jit
    def greedy_actions(params, obs):
        return jnp.argmax(network.apply(params, obs), axis=-1)

    def behavioral_divergence(old_params, new_params, ref_obs):
        """Fraction of ref-batch states where greedy action changed (eq. 3)."""
        old_a = greedy_actions(old_params, ref_obs)
        new_a = greedy_actions(new_params, ref_obs)
        return float(jnp.mean(old_a != new_a))

    def adapt_num_steps(smoothed_div, current_L):
        """Log-interp L_adapt then EMA (eqs. 4, 5, 9)."""
        d_clipped = float(np.clip(smoothed_div, div_low, div_high))
        # eq. 5: alpha
        alpha = np.log(d_clipped / div_low) / np.log(div_high / div_low)
        # eq. 4: L_target
        L_target = num_steps_max - alpha * (num_steps_max - num_steps_min)
        # eq. 9: EMA
        L_new = int(round((1 - ema_beta) * current_L + ema_beta * L_target))
        return max(num_steps_min, min(num_steps_max, L_new))

    def scale_epochs(L):
        """eq. 10: N_epochs scaled with L/L_min."""
        return max(update_epochs_base, int(round(update_epochs_base * L / num_steps_min)))

    def build_full_pqn_step(num_steps_static, update_epochs_static):
        batch_size = num_envs * num_steps_static
        minibatch_size = batch_size // num_minibatches

        def full_pqn_step(q_state, env_state, obs, last_done, rng, global_step):
            def step_once(carry, _):
                q_params, env_state, last_obs, last_done, key, global_step = carry
                epsilon = jnp.maximum(
                    end_e,
                    start_e + (end_e - start_e) * global_step.astype(jnp.float32) / exploration_steps,
                )
                q_vals = network.apply(q_params, last_obs)
                max_actions = jnp.argmax(q_vals, axis=-1)
                max_vals = q_vals[jnp.arange(num_envs), max_actions]
                key, act_key, exp_key = jax.random.split(key, 3)
                rnd = jax.random.randint(act_key, (num_envs,), 0, action_dim)
                explore = jax.random.uniform(exp_key, (num_envs,)) < epsilon
                actions = jnp.where(explore, rnd, max_actions)
                next_obs, new_states, rewards, next_done, info = vmap_step(env_state, actions)
                done = next_done.astype(jnp.float32)
                storage = Storage(
                    obs=last_obs, actions=actions, rewards=rewards,
                    dones=last_done, values=max_vals,
                    returns=jnp.zeros_like(rewards),
                )
                new_carry = (q_params, new_states, next_obs, done, key, global_step + num_envs)
                return new_carry, (storage, info)

            (q_params, env_state, next_obs, next_done, rng, global_step), (storage, infos) = jax.lax.scan(
                step_once,
                (q_state.params, env_state, obs, last_done, rng, global_step),
                None,
                length=num_steps_static,
            )

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
                (storage.rewards, next_values_t, next_dones_t), reverse=True,
            )
            storage = storage.replace(returns=returns)

            def update_epoch(carry, _):
                q_state, key = carry
                key, subkey = jax.random.split(key)
                def flatten(x):
                    return x.reshape((-1,) + x.shape[2:])
                def convert_data(x):
                    x = jax.random.permutation(subkey, x)
                    return jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
                flat = jax.tree_util.tree_map(flatten, storage)
                shuffled = jax.tree_util.tree_map(convert_data, flat)
                def update_minibatch(q_state, mb):
                    def loss_fn(params):
                        q_vals = network.apply(params, mb.obs)
                        q_sel = q_vals[jnp.arange(minibatch_size), mb.actions]
                        return jnp.mean((mb.returns - q_sel) ** 2), q_sel.mean()
                    (loss, q_val), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_state.params)
                    q_state = q_state.apply_gradients(grads=grads)
                    return q_state, (loss, q_val)
                q_state, (loss, q_val) = jax.lax.scan(update_minibatch, q_state, shuffled)
                return (q_state, key), (loss, q_val)

            (q_state, rng), (loss, q_val) = jax.lax.scan(update_epoch, (q_state, rng), (), length=update_epochs_static)

            return (q_state, env_state, next_obs, next_done, rng, global_step), (infos, loss[-1, -1], q_val[-1, -1], storage.obs)

        return jax.jit(full_pqn_step)

    step_cache = {}
    def get_step_fn(L, Nep):
        k = (L, Nep)
        if k not in step_cache:
            print(f"[pqn_ada] compiling full_pqn_step(L={L}, N_epochs={Nep}) ...")
            t0 = time.perf_counter()
            fn = build_full_pqn_step(L, Nep)
            # trigger compile
            _ = jax.block_until_ready(fn(q_state, _state, _obs, _done, key, jnp.array(0, dtype=jnp.int32)))
            print(f"[pqn_ada] compile time: {time.perf_counter() - t0:.2f}s")
            step_cache[k] = fn
        return step_cache[k]

    def save_and_eval(step_count):
        if config.get("SAVE_PATH", "./models") is not None:
            model_path = f'{config.get("SAVE_PATH", "./models")}/{run_name}/{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes([config, q_state_ref[0].params]))
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
                Model=QNetwork if config["PIXEL_BASED"] else MLP_QNetwork,
                seed=config["SEED"] + 42,
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

    print(f"[pqn_ada] starting training for {total_timesteps} steps, L in [{num_steps_min}, {num_steps_max}], K={adapt_freq}")
    global_step = jnp.array(0, dtype=jnp.int32)
    env_state_cur = _state
    obs_cur = _obs
    done_cur = _done
    rng_cur = key

    q_state_ref = [q_state] 
    old_params = jax.tree_util.tree_map(lambda x: x.copy(), q_state.params)  
    div_history = deque(maxlen=div_window)

    rtpt_max = total_timesteps // (num_envs * num_steps_min)
    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=rtpt_max)
    rtpt.start()

    run_time = time.perf_counter()
    iteration = 0
    current_Nep = update_epochs_base

    while int(global_step) < total_timesteps:
        iteration += 1
        rtpt.step()

        if config["EVAL_DURING_TRAIN"] and iteration > 0 and iteration % config["EVAL_EVERY"] == 0:
            save_and_eval(int(global_step))

        iteration_time_start = time.perf_counter()
        step_fn = get_step_fn(num_steps, current_Nep)
        (q_state, env_state_cur, obs_cur, done_cur, rng_cur, global_step), (infos, loss, q_val, rollout_obs) = step_fn(
            q_state_ref[0], env_state_cur, obs_cur, done_cur, rng_cur, global_step
        )
        q_state_ref[0] = q_state
        iteration_time = time.perf_counter() - iteration_time_start
        gs = int(global_step)

        if iteration % adapt_freq == 0:
            flat_obs = np.asarray(rollout_obs).reshape((-1,) + tuple(rollout_obs.shape[2:]))
            M = min(ref_batch_size, len(flat_obs))
            idx = np.random.choice(len(flat_obs), M, replace=False)
            ref_obs = jnp.asarray(flat_obs[idx])
            delta = behavioral_divergence(old_params, q_state.params, ref_obs)
            div_history.append(delta)
            old_params = jax.tree_util.tree_map(lambda x: x.copy(), q_state.params)

            if len(div_history) >= burn_in_min_samples:
                smoothed = float(np.mean(div_history))
                L_new = adapt_num_steps(smoothed, num_steps)
                if L_new != num_steps:
                    Nep_new = scale_epochs(L_new)
                    print(f"[pqn_ada] iter {iteration}: div={delta:.3f} smoothed={smoothed:.3f} L {num_steps}->{L_new} Nep {current_Nep}->{Nep_new}")
                    num_steps = L_new
                    current_Nep = Nep_new
                wandb.log({
                    "abs/behavioral_divergence": delta,
                    "abs/smoothed_divergence": smoothed,
                    "abs/rollout_length": num_steps,
                    "abs/update_epochs": current_Nep,
                    "abs/batch_size": num_envs * num_steps,
                }, step=gs)

        steps_per_iter = num_envs * num_steps
        print(f"[pqn_ada] it {iteration} | gs {gs} | L={num_steps} Nep={current_Nep} | avg_ret {infos['returned_episode_returns'][-1].mean():.2f} | td_loss {loss:.4f} | q_val {q_val:.4f} | SPS {int(gs / (time.perf_counter() - run_time))}")
        wandb.log({
            "charts/avg_episodic_return": infos["returned_episode_returns"][-1].mean(),
            "charts/avg_episodic_length": infos["returned_episode_lengths"][-1].mean(),
            "losses/td_loss": float(loss),
            "losses/q_values": float(q_val),
            "charts/SPS": int(gs / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(steps_per_iter / iteration_time),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": gs,
            "charts/rollout_length": num_steps,
        }, step=gs)

    eval_metrics = save_and_eval(int(global_step) + 1)
    wandb.finish()
    return eval_metrics
