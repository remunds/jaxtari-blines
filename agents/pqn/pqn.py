import os
import time
from collections import deque

import flax
import flax.linen as nn
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

class QNetworkPixel(nn.Module):
    """CNN Q-network for pixel observations (4×84×84×1 input)."""
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.squeeze(-1)
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

class QNetworkOC(nn.Module):
    """MLP Q-network for object-centric observations (flat vector input)."""
    action_dim:  int
    hidden_size: int = 256

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)

def make_env(env_id: str, pixel_based: bool = True, eval: bool = False):
    """Returns a thunk (callable) that constructs the environment."""
    def thunk():
        env = jaxatari.make(env_id)
        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=not eval,
            first_fire=True,
            noop_max=30,
        )
        if pixel_based:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(84, 84),
                grayscale=True,
                frame_stack_size=4,
                frame_skip=4,
                max_pooling=True,
                clip_reward=not eval,
            )
        else:
            env = ObjectCentricWrapper(env, frame_stack_size=4, frame_skip=4, clip_reward=not eval)
            env = NormalizeObservationWrapper(env)
            env = FlattenObservationWrapper(env)
        return LogWrapper(env)
    return thunk



def single_run(config: dict) -> dict:
    config = {k.upper(): v for k, v in config.items() if k.lower() != "alg"}

    pixel_based         = bool(config.get("PIXEL_BASED", True))
    game                = str(config.get("ENV_ID", "pong")).lower()
    seed                = int(config.get("SEED", 42))
    exp_name            = str(config.get("EXP_NAME", "pqn"))

    total_timesteps     = int(config.get("TOTAL_TIMESTEPS", 10_000_000))
    num_envs            = int(config.get("NUM_ENVS", 8))
    num_steps           = int(config.get("NUM_STEPS", 128))
    learning_rate       = float(config.get("LEARNING_RATE", 2.5e-4))
    anneal_lr           = bool(config.get("ANNEAL_LR", True))
    gamma               = float(config.get("GAMMA", 0.99))
    q_lambda_val        = float(config.get("Q_LAMBDA", 0.65))
    num_minibatches     = int(config.get("NUM_MINIBATCHES", 4))
    update_epochs       = int(config.get("UPDATE_EPOCHS", 4))
    max_grad_norm       = float(config.get("MAX_GRAD_NORM", 10.0))
    start_e             = float(config.get("START_E", 1.0))
    end_e               = float(config.get("END_E", 0.01))
    exploration_fraction = float(config.get("EXPLORATION_FRACTION", 0.10))
    save_path           = config.get("SAVE_PATH", "./models")

    batch_size        = num_envs * num_steps
    minibatch_size    = batch_size // num_minibatches
    num_iterations    = total_timesteps // batch_size
    exploration_steps = float(exploration_fraction * total_timesteps)

    run_name = f"{game}_{exp_name}_{'pixel' if pixel_based else 'oc'}_{seed}"

    wandb.init(
        project=config.get("PROJECT", "jaxatari-pqn"),
        entity=config.get("ENTITY", None) or None,
        config=config,
        name=run_name,
        save_code=True,
    )

    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    key, q_key = jax.random.split(key)

    env = make_env(game, pixel_based=pixel_based, eval=False)()

    key, probe_key = jax.random.split(key)
    _obs, _ = env.reset(probe_key)
    obs_shape = _obs.shape
    n_actions = env.action_space().n

    print(f"[PQN] run_name    : {run_name}")
    print(f"      obs_shape   : {obs_shape}")
    print(f"      n_actions   : {n_actions}")
    print(f"      num_envs    : {num_envs}")
    print(f"      batch_size  : {batch_size}")
    print(f"      minibatch   : {minibatch_size}")
    print(f"      iterations  : {num_iterations}")

    q_network = QNetworkPixel(n_actions) if pixel_based else QNetworkOC(n_actions)

    total_grad_steps = num_iterations * update_epochs * num_minibatches
    lr = (optax.linear_schedule(learning_rate, 0.0, total_grad_steps)
          if anneal_lr else learning_rate)
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.radam(lr))

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, jnp.zeros((1, *obs_shape))),
        tx=tx,
    )

    vmap_reset = jax.jit(jax.vmap(env.reset))
    vmap_step  = jax.jit(jax.vmap(env.step))

    key, *env_keys = jax.random.split(key, num_envs + 1)
    next_obs, env_states = vmap_reset(jnp.array(env_keys))
    next_done = jnp.zeros(num_envs, dtype=jnp.float32)


    def step_once(carry, _):
        q_params, env_states, last_obs, last_done, key, global_step = carry

        epsilon = jnp.maximum(
            end_e,
            start_e + (end_e - start_e) * global_step.astype(jnp.float32) / exploration_steps,
        )

        q_vals      = q_network.apply(q_params, last_obs)
        max_actions = jnp.argmax(q_vals, axis=-1)
        max_vals    = q_vals[jnp.arange(num_envs), max_actions]

        key, act_key, exp_key = jax.random.split(key, 3)
        rnd     = jax.random.randint(act_key, (num_envs,), 0, n_actions)
        explore = jax.random.uniform(exp_key, (num_envs,)) < epsilon
        actions = jnp.where(explore, rnd, max_actions)

        next_obs, new_states, rewards, terminated, truncated, infos = vmap_step(env_states, actions)
        done = jnp.logical_or(terminated, truncated).astype(jnp.float32)

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
            step_once, init_carry, None, length=num_steps
        )
        return final_carry, storage, infos


    def compute_q_lambda_once(carry, inp):
        next_return = carry
        reward, next_val, next_done = inp
        ret = reward + gamma * (q_lambda_val * next_return + (1.0 - q_lambda_val) * next_val) * (1.0 - next_done)
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
                return jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])

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
            update_epoch, (q_state, key), (), length=update_epochs
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
            end_e, start_e + (end_e - start_e) * gs / exploration_steps
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
            print(f"step: {gs}/{total_timesteps} | SPS: {sps} | "
                  f"avg_return: {np.mean(avg_returns) if avg_returns else 0:.2f}")

    model_path = None
    if save_path is not None:
        model_path = f"{save_path}/{run_name}_{int(time.time())}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes((None, q_state.params)))
        print(f"Model saved to {model_path}")

    eval_episodes = 10
    eval_metrics  = {}

    if model_path is not None:
        QNetwork = QNetworkPixel if pixel_based else QNetworkOC
        episodic_returns, _ = evaluate(
            model_path, make_env, game, eval_episodes,
            QNetwork, pixel_based=pixel_based, epsilon=0.05, seed=seed,
        )
        avg_eval_return = float(jnp.mean(episodic_returns))
        eval_metrics["eval/episodic_return_default"] = avg_eval_return
        wandb.log({"eval/episodic_return_default": avg_eval_return}, step=total_timesteps)
        print(f"Eval return (greedy): {avg_eval_return:.2f}")

    wandb.finish()
    print("[PQN] Training complete.")
    return eval_metrics
