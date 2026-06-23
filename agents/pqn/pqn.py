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

from agents.pqn.pqn_eval import evaluate

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



def single_run(config: dict) -> dict:
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    if isinstance(config.get("TRAIN_MODS"), list):
        config["TRAIN_MODS"] = tuple(config["TRAIN_MODS"])
    if isinstance(config.get("EVAL_MODS"), list):
        config["EVAL_MODS"] = tuple(config["EVAL_MODS"])

    if config.get("PIXEL_BASED", True) and config.get("NUM_ENVS", 1) > 16:
        config["NUM_ENVS"] = 8

    run_name = f"{config["ENV_ID"]}_{config["EXP_NAME"]}_{"oc" if not config["PIXEL_BASED"] else "pixel"}_{config["SEED"]}"

    batch_size        = config["NUM_ENVS"] * config["NUM_STEPS"]
    minibatch_size    = batch_size // config["NUM_MINIBATCHES"]
    num_iterations    = config["TOTAL_TIMESTEPS"] // batch_size
    exploration_steps = float(config.get("EXPLORATION_FRACTION", 0.10) * config["TOTAL_TIMESTEPS"])

    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
    )

    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])
    key, q_key = jax.random.split(key)

    env = make_env(
        config.get("ENV_ID"),
        list(config.get("TRAIN_MODS", [])),
        config.get("PIXEL_BASED", True),
        config.get("NATIVE_DOWNSCALING", True),
        False
    )()


    n_actions = env.action_space().n
    obs_shape = env.observation_space().shape
    if config.get("PIXEL_BASED", True):
        obs_shape = obs_shape[:-1]  


    print(f"[PQN] run_name    : {run_name}")
    print(f"      obs_shape   : {obs_shape}")
    print(f"      n_actions   : {n_actions}")
    print(f"      num_envs    : {config['NUM_ENVS']}")
    print(f"      batch_size  : {batch_size}")
    print(f"      minibatch   : {minibatch_size}")
    print(f"      iterations  : {num_iterations}")

    q_network = QNetwork(n_actions) if config["PIXEL_BASED"] else MLP_QNetwork(n_actions)

    total_grad_steps = num_iterations * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
    tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.inject_hyperparams(optax.radam)(
                learning_rate=(optax.linear_schedule(config["LEARNING_RATE"], 0.0, total_grad_steps)
                if config["ANNEAL_LR"] else config["LEARNING_RATE"])
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


    key, *env_keys = jax.random.split(key, config["NUM_ENVS"] + 1)
    next_obs, env_states = vmap_reset(jnp.array(env_keys))
    next_done = jnp.zeros(config["NUM_ENVS"], dtype=jnp.float32)


    def step_once(carry, _):
        q_params, env_states, last_obs, last_done, key, global_step = carry

        epsilon = jnp.maximum(
            config["END_E"],
            config["START_E"] + (config["END_E"] - config["START_E"]) * global_step.astype(jnp.float32) / exploration_steps,
        )

        q_vals      = q_network.apply(q_params, last_obs)
        max_actions = jnp.argmax(q_vals, axis=-1)
        max_vals    = q_vals[jnp.arange(config["NUM_ENVS"]), max_actions]

        key, act_key, exp_key = jax.random.split(key, 3)
        rnd     = jax.random.randint(act_key, (config["NUM_ENVS"],), 0, n_actions)
        explore = jax.random.uniform(exp_key, (config["NUM_ENVS"],)) < epsilon
        actions = jnp.where(explore, rnd, max_actions)

        next_obs, new_states, rewards, next_done, infos = vmap_step(env_states, actions)
        done = next_done.astype(jnp.float32)

        storage = Storage(
            obs=last_obs, actions=actions, rewards=rewards,
            dones=last_done, values=max_vals,
            returns=jnp.zeros_like(rewards),
        )
        new_carry = (q_params, new_states, next_obs, done, key, global_step + config["NUM_ENVS"])
        return new_carry, (storage, infos)

    @jax.jit
    def rollout(q_params, env_states, last_obs, last_done, key, global_step):
        init_carry = (q_params, env_states, last_obs, last_done, key, global_step)
        final_carry, (storage, infos) = jax.lax.scan(
            step_once, init_carry, None, length=config["NUM_STEPS"]
        )
        return final_carry, storage, infos


    def compute_q_lambda_once(carry, inp):
        next_return = carry
        reward, next_val, next_done = inp
        ret = reward + config["GAMMA"] * (config["Q_LAMBDA"] * next_return + (1.0 - config["Q_LAMBDA"]) * next_val) * (1.0 - next_done)
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
                return jnp.reshape(x, (config["NUM_MINIBATCHES"], -1) + x.shape[1:])

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
            update_epoch, (q_state, key), (), length=config["UPDATE_EPOCHS"]
        )
        return q_state, loss, q_val, key

    avg_returns = deque(maxlen=20)
    global_step = jnp.int32(0)
    start_time  = time.time()

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

        sps        = int(gs / (time.time() - start_time))
        sps_update = int(batch_size / update_time)
        epsilon    = float(jnp.maximum(
            config["END_E"], config["START_E"] + (config["END_E"] - config["START_E"]) * gs / exploration_steps
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
            print(f"step: {gs}/{config["TOTAL_TIMESTEPS"]} | SPS: {sps} | "
                  f"avg_return: {np.mean(avg_returns) if avg_returns else 0:.2f}")

    model_path = None
    if config.get("SAVE_PATH", "./models") is not None:
        model_path = f"{config.get("SAVE_PATH", "./models")}/{run_name}_{int(time.time())}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes((None, q_state.params)))
        print(f"Model saved to {model_path}")

    eval_episodes = 10
    eval_metrics  = {}

    if model_path is not None:
        QNetwork = QNetwork if config["PIXEL_BASED"] else MLP_QNetwork
        episodic_returns, _ = evaluate(
            model_path, make_env, config.get("ENV_ID"), eval_episodes,
            QNetwork, pixel_based=config["PIXEL_BASED"], epsilon=0.05, seed=config["SEED"],
        )
        avg_eval_return = float(jnp.mean(episodic_returns))
        eval_metrics["eval/episodic_return_default"] = avg_eval_return
        wandb.log({"eval/episodic_return_default": avg_eval_return}, step=config["TOTAL_TIMESTEPS"])
        print(f"Eval return (greedy): {avg_eval_return:.2f}")

    wandb.finish()
    print("[PQN] Training complete.")
    return eval_metrics
