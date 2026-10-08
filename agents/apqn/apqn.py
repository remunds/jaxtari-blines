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
import jaxtari
from jaxtari.wrappers import (
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    AtariWrapper,
    LogWrapper,
    FlattenObservationWrapper,
)
from agents.apqn.apqn_eval import evaluate
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
    env_id = config["ENV_ID"]
    pixel_based = config["PIXEL_BASED"]

    if pixel_based and config["NUM_ENVS"] > 16:
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

    # do not modify the seeding
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    gamma = config.get("GAMMA", 0.99)
    q_lambda_val = config.get("Q_LAMBDA", 0.65)
    num_minibatches = config.get("NUM_MINIBATCHES", 4)
    update_epochs_base = config.get("UPDATE_EPOCHS", 2)
    max_grad_norm = config.get("MAX_GRAD_NORM", 10.0)
    total_timesteps = config["TOTAL_TIMESTEPS"]
    start_e = config.get("START_E", 1.0)
    end_e = config.get("END_E", 0.001)
    exploration_steps = config.get("EXPLORATION_FRACTION", 0.10) * total_timesteps
    learning_rate = config.get("LEARNING_RATE", 2.5e-4)
    anneal_lr = config.get("ANNEAL_LR", True)

    num_steps_min = config.get("NUM_STEPS_MIN", 16)
    num_steps_max = config.get("NUM_STEPS_MAX", 64)
    adapt_freq = config.get("ADAPT_FREQ", 50)
    div_low = config.get("POLICY_CHANGE_LOW", 0.05)
    div_high = config.get("POLICY_CHANGE_HIGH", 0.95)
    ema_beta = config.get("EMA_BETA", 0.1)
    ref_batch_size = config.get("REF_BATCH_SIZE", 2048)
    div_window = config.get("DIV_WINDOW", 10)
    burn_in_min_samples = div_window

    env = make_env(env_id, list(config.get("TRAIN_MODS") or []), pixel_based, config.get("NATIVE_DOWNSCALING", True))()
    action_dim = env.action_space().n
    obs_shape = env.observation_space().shape
    if pixel_based:
        obs_shape = obs_shape[:-1]  # drop the trailing grayscale channel

    num_envs = config["NUM_ENVS"]

    # one chunk = scan_steps PQN updates. The rollout length L adapts only every adapt_freq updates,
    # so the scan length is rounded to a multiple of adapt_freq: adaptation happens exactly at chunk ends.
    scan_len = max(adapt_freq, (config.get("SCAN_STEPS", 1000) // adapt_freq) * adapt_freq)
    scan_steps, num_chunks, steps_per_chunk = plan_chunks(
        total_timesteps,
        num_envs * num_steps_min,  # smallest rollout; the planned budget assumes L = NUM_STEPS_MIN
        scan_len,
    )

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
    network = QNetwork(action_dim=action_dim) if pixel_based else MLP_QNetwork(action_dim=action_dim)
    q_params = network.init(q_key, jnp.zeros((1, *obs_shape)))

    approx_num_iters = total_timesteps // (num_envs * num_steps_min)
    total_grad_steps = approx_num_iters * update_epochs_base * num_minibatches
    lr = optax.linear_schedule(learning_rate, 0.0, total_grad_steps) if anneal_lr else learning_rate
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.radam(lr))
    q_state = TrainState.create(apply_fn=network.apply, params=q_params, tx=tx)

    obs, env_state = vmap_reset(jax.random.split(key, num_envs))
    num_steps = num_steps_min
    current_Nep = update_epochs_base
    carry = (
        q_state,
        env_state,
        obs,
        jnp.zeros(num_envs, dtype=jnp.float32),  # done flags of the last step
        key,
        jnp.array(0, dtype=jnp.int32),  # env steps taken
        jnp.zeros((num_steps * num_envs,) + tuple(obs_shape), dtype=obs.dtype),  # rollout obs for the divergence check
    )

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

    def build_chunk(num_steps_static, update_epochs_static):
        batch_size = num_envs * num_steps_static
        minibatch_size = batch_size // num_minibatches

        def pqn_update(carry):
            """One rollout of num_steps_static per env, Q(lambda) targets, and update_epochs_static of minibatch SGD."""
            q_state, env_state, obs, last_done, key, global_step, _ = carry

            def step_once(c, _):
                q_params, env_state, last_obs, last_done, key, global_step = c
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
                new_c = (q_params, new_states, next_obs, done, key, global_step + num_envs)
                return new_c, (storage, info)

            (q_params, env_state, next_obs, next_done, key, global_step), (storage, infos) = jax.lax.scan(
                step_once,
                (q_state.params, env_state, obs, last_done, key, global_step),
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

            (q_state, key), (loss, q_val) = jax.lax.scan(update_epoch, (q_state, key), (), length=update_epochs_static)

            # rollout obs of this update, kept in the carry for the divergence check at the chunk end
            ref_obs = storage.obs.reshape((batch_size,) + storage.obs.shape[2:])
            new_carry = (q_state, env_state, next_obs, next_done, key, global_step, ref_obs)
            return new_carry, (infos, loss[-1, -1], q_val[-1, -1])

        def train_chunk(carry):
            return jax.lax.scan(lambda c, _: pqn_update(c), carry, None, length=scan_steps)

        return train_chunk

    def fit_ref(carry, L):
        """Give the rollout-obs slot of the carry the shape of rollout length L (only matters after an adaptation)."""
        ref = carry[6]
        want = (L * num_envs,) + tuple(obs_shape)
        if ref.shape == want:
            return carry
        return carry[:6] + (jnp.zeros(want, dtype=ref.dtype),)

    chunk_fns = {}

    def get_chunk_fn(carry, L, Nep):
        # one compiled chunk per (rollout length, epochs); lowered AOT so nothing is donated before the loop
        if (L, Nep) not in chunk_fns:
            print(f"[apqn] compiling chunk(L={L}, N_epochs={Nep}) ...")
            t0 = time.perf_counter()
            chunk_fns[(L, Nep)] = jax.jit(build_chunk(L, Nep), donate_argnums=(0,)).lower(carry).compile()
            print(f"[apqn] compilation time: {time.perf_counter() - t0:.2f}s")
        return chunk_fns[(L, Nep)]

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
                    pixel_based=pixel_based,
                    native_downscaling=config.get("NATIVE_DOWNSCALING", True),
                    eval=True,
                ),
                env_id,
                eval_episodes=10,
                Model=QNetwork if pixel_based else MLP_QNetwork,
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

    print(f"[apqn] start training: {num_chunks} chunks x {scan_steps} updates, L in [{num_steps_min}, {num_steps_max}], adapt every {adapt_freq} updates")
    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=num_chunks)
    rtpt.start()
    run_time = time.perf_counter()
    old_params = jax.tree_util.tree_map(lambda x: x.copy(), q_state.params)
    div_history = deque(maxlen=div_window)

    for chunk in range(num_chunks):
        rtpt.step()
        if config["EVAL_DURING_TRAIN"] and chunk > 0 and chunk % config["EVAL_EVERY"] == 0:
            save_and_eval(carry[0].params, int(carry[5]))

        carry = fit_ref(carry, num_steps)
        compiled = get_chunk_fn(carry, num_steps, current_Nep)
        chunk_start = time.perf_counter()
        carry, (infos, loss, q_val) = compiled(carry)
        global_step = int(carry[5])
        chunk_time = time.perf_counter() - chunk_start
        steps_this_chunk = scan_steps * num_envs * num_steps

        metrics = {
            "charts/avg_episodic_return": float(infos["returned_episode_returns"][-1].mean()),
            "charts/avg_episodic_length": float(infos["returned_episode_lengths"][-1].mean()),
            "losses/td_loss": loss[-1].item(),
            "losses/q_values": q_val[-1].item(),
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(steps_this_chunk / chunk_time),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }

        # adaptive rollout length (eqs. 3-10), checked every adapt_freq updates = at chunk ends
        if (chunk + 1) * scan_steps % adapt_freq == 0:
            flat_obs = np.asarray(carry[6])
            M = min(ref_batch_size, len(flat_obs))
            idx = np.random.choice(len(flat_obs), M, replace=False)
            delta = behavioral_divergence(old_params, carry[0].params, jnp.asarray(flat_obs[idx]))
            div_history.append(delta)
            old_params = jax.tree_util.tree_map(lambda x: x.copy(), carry[0].params)

            if len(div_history) >= burn_in_min_samples:
                smoothed = float(np.mean(div_history))
                L_new = adapt_num_steps(smoothed, num_steps)
                if L_new != num_steps:
                    Nep_new = scale_epochs(L_new)
                    print(f"[apqn] chunk {chunk + 1}: div={delta:.3f} smoothed={smoothed:.3f} L {num_steps}->{L_new} Nep {current_Nep}->{Nep_new}")
                    num_steps = L_new
                    current_Nep = Nep_new
                metrics.update({
                    "abs/behavioral_divergence": delta,
                    "abs/smoothed_divergence": smoothed,
                    "abs/rollout_length": num_steps,
                    "abs/update_epochs": current_Nep,
                    "abs/batch_size": num_envs * num_steps,
                })

        wandb.log(metrics, step=global_step)
        print(
            f"[apqn] chunk {chunk + 1}/{num_chunks} | global_step {global_step} | "
            f"avg_return {metrics['charts/avg_episodic_return']:.2f} | "
            f"avg_length {metrics['charts/avg_episodic_length']:.2f} | "
            f"td_loss {metrics['losses/td_loss']:.4f} | q_val {metrics['losses/q_values']:.4f} | "
            f"L {steps_this_chunk // (scan_steps * num_envs)} | SPS {metrics['charts/SPS']}"
        )

    eval_metrics = save_and_eval(carry[0].params, global_step + 1)
    wandb.finish()
    return eval_metrics
