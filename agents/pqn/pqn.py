# Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/pqn_atari_envpool.py
import os
import time
from collections import deque

import flax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

import jaxatari
from jaxatari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
)
import wandb


@flax.struct.dataclass
class Storage:
    obs:     jnp.array
    actions: jnp.array
    rewards: jnp.array
    dones:   jnp.array
    values:  jnp.array
    returns: jnp.array


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


class MLP_QNetwork(nn.Module):
    action_dim:  int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.Dense(461, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        return x


def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False):
    def thunk():
        active_mods = mods
        if not eval and isinstance(active_mods, (list, tuple)) and len(active_mods) > 1:
            active_mods = []

        if isinstance(active_mods, (list, tuple)) and len(active_mods) == 0:
            mods_arg = None
        else:
            mods_arg = active_mods

        env = jaxatari.make(env_id, mods=mods_arg)

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


def build_eval_fn(env, apply_fn, eval_episodes, max_steps, action_dim):
    def wrapped_reset(key):
        next_obs, state = env.reset(key)
        return next_obs.squeeze()[None, ...], state

    def wrapped_step(state, action):
        next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze()[None, ...], next_state, reward, done, info

    def get_action(params, obs, key, epsilon):
        q_values = apply_fn(params, obs)
        greedy_action = jnp.argmax(q_values, axis=1)

        key, subkey = jax.random.split(key)
        random_action = jax.random.randint(subkey, greedy_action.shape, 0, action_dim)
        explore = jax.random.uniform(key, greedy_action.shape) < epsilon
        action = jnp.where(explore, random_action, greedy_action)
        return action, key

    def step_fn(carry, _):
        obs, env_state, keys, params, epsilon = carry

        actions, keys = jax.vmap(get_action, in_axes=(None, 0, 0, None))(params, obs, keys, epsilon)
        next_obs, next_env_state, reward, done, info = jax.vmap(wrapped_step)(env_state, actions)
        first_state = jax.tree.map(lambda x: x[0], next_env_state)

        return (next_obs, next_env_state, keys, params, epsilon), (first_state, done, reward)

    @jax.jit
    def eval_fn(params, reset_keys, epsilon):
        obs, env_state = jax.vmap(wrapped_reset)(reset_keys)

        _, (first_states_history, dones, rewards) = jax.lax.scan(
            step_fn, (obs, env_state, reset_keys, params, epsilon), None, length=max_steps)
        has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
        mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
        masked_rewards = rewards * (1 - mask_after_first_done)
        episodic_returns = jnp.sum(masked_rewards, axis=0)

        first_done = jnp.argmax(dones, axis=0)
        return episodic_returns, first_states_history, first_done

    return eval_fn


def single_run(config: dict) -> dict:
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    if isinstance(config.get("TRAIN_MODS"), list):
        config["TRAIN_MODS"] = tuple(config.get("TRAIN_MODS", []))
    if isinstance(config.get("EVAL_MODS"), list):
        config["EVAL_MODS"] = tuple(config.get("EVAL_MODS", []))

    if config.get("PIXEL_BASED", True) and config.get("NUM_ENVS", 1) > 16:
        config["NUM_ENVS"] = 8

    run_name = f"{config.get("ENV_ID", "pong")}_{config.get("EXP_NAME", "pqn")}_{"oc" if not config.get("PIXEL_BASED", True) else "pixel"}_{config.get("SEED", 0)}"

    batch_size        = config.get("NUM_ENVS", 8) * config.get("NUM_STEPS", 32)
    minibatch_size    = batch_size // config.get("NUM_MINIBATCHES", 4)
    num_iterations    = config.get("TOTAL_TIMESTEPS", 10_000_000) // batch_size
    exploration_steps = float(config.get("EXPLORATION_FRACTION", 0.10) * config.get("TOTAL_TIMESTEPS", 10_000_000))

    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
    )

    np.random.seed(config.get("SEED", 0))
    key = jax.random.PRNGKey(config.get("SEED", 0))
    key, q_key = jax.random.split(key)

    env = make_env(
        config.get("ENV_ID", "pong"),
        list(config.get("TRAIN_MODS", [])),
        config.get("PIXEL_BASED", True),
        config.get("NATIVE_DOWNSCALING", True),
        False
    )()

    n_actions = env.action_space().n
    obs_shape = env.observation_space().shape
    if config.get("PIXEL_BASED", True):
        obs_shape = obs_shape[:-1]

    q_network = QNetwork(n_actions) if config.get("PIXEL_BASED", True) else MLP_QNetwork(n_actions)

    total_grad_steps = num_iterations * config.get("UPDATE_EPOCHS", 2) * config.get("NUM_MINIBATCHES", 4)
    tx = optax.chain(
            optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 10.0)),
            optax.inject_hyperparams(optax.radam)(
                learning_rate=(optax.linear_schedule(config.get("LEARNING_RATE", 2.5e-4), 0.0, total_grad_steps)
                if config.get("ANNEAL_LR", False) else config.get("LEARNING_RATE", 2.5e-4))
                ),
        )

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, jnp.zeros((1, *obs_shape))),
        tx=tx,
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

    num_envs = config.get("NUM_ENVS", 8)
    key, *env_keys = jax.random.split(key, num_envs + 1)
    next_obs, env_states = vmap_reset(jnp.array(env_keys))
    next_done = jnp.zeros(num_envs, dtype=jnp.float32)

    def step_once(carry, _):
        q_params, env_states, last_obs, last_done, key, global_step = carry

        epsilon = jnp.maximum(
            config.get("END_E", 0.001),
            config.get("START_E", 1.0) + (config.get("END_E", 0.001) - config.get("START_E", 1.0)) * global_step.astype(jnp.float32) / exploration_steps,
        )

        q_vals      = q_network.apply(q_params, last_obs)
        max_actions = jnp.argmax(q_vals, axis=-1)
        max_vals    = q_vals[jnp.arange(num_envs), max_actions]

        key, act_key, exp_key = jax.random.split(key, 3)
        rnd     = jax.random.randint(act_key, (num_envs,), 0, n_actions)
        explore = jax.random.uniform(exp_key, (num_envs,)) < epsilon
        actions = jnp.where(explore, rnd, max_actions)

        next_obs, new_states, rewards, next_done, infos = vmap_step(env_states, actions)
        done = next_done.astype(jnp.float32)

        storage = Storage(
            obs=last_obs, actions=actions, rewards=rewards,
            dones=last_done, values=max_vals,
            returns=jnp.zeros_like(rewards),
        )
        new_carry = (q_params, new_states, next_obs, done, key, global_step + num_envs)
        return new_carry, (storage, infos)

    @jax.jit
    def rollout(q_params, env_states, last_obs, last_done, key, global_step):
        init_carry = (q_params, env_states, last_obs, last_done, key, global_step)
        final_carry, (storage, infos) = jax.lax.scan(
            step_once, init_carry, None, length=config.get("NUM_STEPS", 32)
        )
        return final_carry, storage, infos

    gamma_val   = config.get("GAMMA", 0.99)
    qlambda_val = config.get("Q_LAMBDA", 0.65)

    def compute_q_lambda_once(carry, inp):
        next_return = carry
        reward, next_val, next_done = inp
        ret = reward + gamma_val * (qlambda_val * next_return + (1.0 - qlambda_val) * next_val) * (1.0 - next_done)
        return ret, ret

    @jax.jit
    def compute_q_lambda(agent_state: TrainState, next_obs, next_done, storage: Storage):
        next_q   = q_network.apply(agent_state.params, next_obs)
        next_val = jnp.max(next_q, axis=-1)

        next_values_t = jnp.concatenate([storage.values[1:], next_val[None]], axis=0)
        next_dones_t  = jnp.concatenate([storage.dones[1:],  next_done[None]], axis=0)

        _, returns = jax.lax.scan(
            compute_q_lambda_once,
            next_val,
            (storage.rewards, next_values_t, next_dones_t),
            reverse=True,
        )
        return storage.replace(returns=returns)

    @jax.jit
    def update_pqn(q_state: TrainState, storage: Storage, key):
        def update_epoch(carry, _):
            q_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x):
                x = jax.random.permutation(subkey, x)
                return jnp.reshape(x, (config.get("NUM_MINIBATCHES", 4), -1) + x.shape[1:])

            flat     = jax.tree_util.tree_map(flatten, storage)
            shuffled = jax.tree_util.tree_map(convert_data, flat)

            def update_minibatch(q_state, mb: Storage):
                def loss_fn(params):
                    q_vals = q_network.apply(params, mb.obs)
                    q_sel  = q_vals[jnp.arange(minibatch_size), mb.actions]
                    loss   = jnp.mean((mb.returns - q_sel) ** 2)
                    return loss, q_sel
                (loss, q_sel), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_state.params)
                q_state = q_state.apply_gradients(grads=grads)
                return q_state, (loss, q_sel.mean())

            q_state, (loss, q_val) = jax.lax.scan(update_minibatch, q_state, shuffled)
            return (q_state, key), (loss, q_val)

        (q_state, key), (loss, q_val) = jax.lax.scan(
            update_epoch, (q_state, key), (), length=config.get("UPDATE_EPOCHS", 2)
        )
        return q_state, loss, q_val, key

    eval_mods_list = list(config.get("EVAL_MODS", [])) or list(config.get("TRAIN_MODS", []))
    eval_configs = [([], "default")]
    for mod in eval_mods_list:
        mods_cfg = list(mod) if isinstance(mod, (list, tuple)) else [mod]
        mod_label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods_cfg)
        eval_configs.append((mods_cfg, mod_label))

    eval_episodes = 10
    eval_max_steps = 10000

    eval_fns = {}
    for mods_cfg, mod_label in eval_configs:
        eval_env = make_env(
            config.get("ENV_ID", "pong"),
            mods=mods_cfg,
            pixel_based=config.get("PIXEL_BASED", True),
            native_downscaling=config.get("NATIVE_DOWNSCALING", True),
            eval=True,
        )()
        eval_fns[mod_label] = build_eval_fn(
            env=eval_env,
            apply_fn=q_network.apply,
            eval_episodes=eval_episodes,
            max_steps=eval_max_steps,
            action_dim=n_actions,
        )

    def save_and_eval(step_count, agent_state):
        model_path = ""
        if config.get("SAVE_PATH", "./models") is not None:
            model_path = f'{config.get("SAVE_PATH", "./models")}/{run_name}/{config.get("EXP_NAME", "pqn")}_{step_count}_{int(time.time())}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            with open(model_path, "wb") as f:
                f.write(
                    flax.serialization.to_bytes(
                        (None, agent_state.params)
                    )
                )
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")

        metrics = {}
        for mods_cfg, mod_label in eval_configs:
            reset_keys = jax.random.split(jax.random.PRNGKey(config.get("SEED", 0)), eval_episodes)

            episodic_returns, first_states_history, first_done = eval_fns[mod_label](
                agent_state.params, reset_keys, 0.05
            )

            avg_eval_return = float(jnp.mean(episodic_returns))
            return_key = f"eval/episodic_return_{mod_label}"
            metrics[return_key] = avg_eval_return
            print(f"evaluation at step {step_count} ({mod_label}): average return = {avg_eval_return}")

            wandb.log({return_key: avg_eval_return}, step=step_count)

            if config.get("CAPTURE_VIDEO", False):
                clean_renderer = jaxatari.make(config.get("ENV_ID", "pong"), mods=mods_cfg).renderer
                env_states_until_done = jax.tree.map(
                    lambda x: x[: first_done[0] + 1],
                    first_states_history.atari_state.atari_state.env_state,
                )
                frames = jax.vmap(clean_renderer.render)(env_states_until_done)
                frames = jnp.transpose(frames, (0, 3, 1, 2))
                video = wandb.Video(np.array(frames), fps=30, format="mp4")

                video_key = f"eval/video_{mod_label}"
                wandb.log({video_key: video}, step=step_count)
                print(f"video (eval) logged to wandb with {frames.shape} frames ({mod_label}).")

        return metrics

    avg_returns = deque(maxlen=20)
    global_step = jnp.int32(0)
    start_time  = time.time()
    total_eval_time = 0.0

    for iteration in range(1, num_iterations + 1):

        (_, env_states, next_obs, next_done, key, global_step), storage, infos = rollout(
            q_state.params, env_states, next_obs, next_done, key, global_step
        )

        gs = int(global_step)
        if "returned_episode" in infos:
            finished = np.array(infos["returned_episode"])
            ep_rets  = np.array(infos["returned_episode_returns"])
            for ret in ep_rets[finished]:
                avg_returns.append(float(ret))
                wandb.log({
                    "charts/episodic_return":     float(ret),
                    "charts/avg_episodic_return": float(np.mean(avg_returns)),
                }, step=gs)

        storage = compute_q_lambda(q_state, next_obs, next_done, storage)

        update_t0 = time.time()
        q_state, loss, q_val, key = update_pqn(q_state, storage, key)
        update_time = time.time() - update_t0

        if config.get("EVAL_DURING_TRAIN", True) and (iteration % config.get("EVAL_EVERY", 10) == 0):
            eval_t0 = time.time()
            save_and_eval(gs, q_state)
            total_eval_time += time.time() - eval_t0

        sps        = int(gs / (time.time() - start_time - total_eval_time))
        sps_update = int(batch_size / update_time)
        epsilon    = float(jnp.maximum(
            config.get("END_E", 0.001),
            config.get("START_E", 1.0) + (config.get("END_E", 0.001) - config.get("START_E", 1.0)) * gs / exploration_steps
        ))
        wandb.log({
            "charts/global_step": gs,
            "charts/epsilon":     epsilon,
            "charts/SPS":         sps,
            "charts/SPS_update":  sps_update,
            "losses/td_loss":     float(loss[-1, -1]),
            "losses/q_values":    float(q_val[-1, -1]),
        }, step=gs)

        if iteration % max(1, num_iterations // 20) == 0:
            print(f"step: {gs}/{config.get("TOTAL_TIMESTEPS", 10_000_000)} | SPS: {sps} | "
                  f"avg_return: {np.mean(avg_returns) if avg_returns else 0:.2f}")

    eval_metrics = save_and_eval(config.get("TOTAL_TIMESTEPS", 10_000_000), q_state)

    wandb.finish()
    return eval_metrics
