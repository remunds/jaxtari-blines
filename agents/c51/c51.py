# Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_atari_jax.py
import time
from collections import deque
from typing import NamedTuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flashbax as fbx
import wandb
from flax.training.train_state import TrainState

import jaxatari
from jaxatari.wrappers import (
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    AtariWrapper,
    LogWrapper,
    FlattenObservationWrapper,
)


def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True):
    def thunk():
        if isinstance(mods, (list, tuple)) and len(mods) == 0:
            mods_arg = None
        else:
            mods_arg = mods

        env = jaxatari.make(env_id, mods=mods_arg)
        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=True,
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
                clip_reward=True,
            )
        else:
            env = FlattenObservationWrapper(
                NormalizeObservationWrapper(
                    ObjectCentricWrapper(
                        env,
                        frame_stack_size=4,
                        frame_skip=4,
                        clip_reward=True,
                    )
                )
            )
        env = LogWrapper(env)
        return env
    return thunk


class QNetworkPixel(nn.Module):
    """CNN for pixel observations. Returns softmax distribution per action."""
    action_dim: int
    n_atoms: int

    @nn.compact
    def __call__(self, x):
        x = x.squeeze(-1)
        x = jnp.transpose(x, (0, 2, 3, 1))
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
        x = nn.Dense(self.action_dim * self.n_atoms)(x)
        x = x.reshape((x.shape[0], self.action_dim, self.n_atoms))
        return jax.nn.softmax(x, axis=-1)


class QNetworkOC(nn.Module):
    """MLP with LayerNorm for object-centric observations."""
    action_dim: int
    n_atoms: int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim * self.n_atoms)(x)
        x = x.reshape((x.shape[0], self.action_dim, self.n_atoms))
        return jax.nn.softmax(x, axis=-1)


class C51TrainState(TrainState):
    target_params: flax.core.FrozenDict
    atoms: jnp.ndarray


class Transition(NamedTuple):
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    if isinstance(config.get("TRAIN_MODS"), list):
        config["TRAIN_MODS"] = tuple(config["TRAIN_MODS"])

    pixel_based = config.get("PIXEL_BASED", True)
    env_id = config.get("ENV_ID", "pong")
    seed = config.get("SEED", 0)

    run_name = f"{env_id}_c51_{'pixel' if pixel_based else 'oc'}_{seed}"

    wandb.init(
        project=config.get("PROJECT", "jaxtari-c51"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
    )

    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    key, q_key = jax.random.split(key)

    env = make_env(
        env_id,
        list(config.get("TRAIN_MODS", [])),
        pixel_based,
        config.get("NATIVE_DOWNSCALING", True),
    )()

    key, probe_key = jax.random.split(key)
    _obs, _ = env.reset(probe_key)
    obs_shape = _obs.shape
    n_actions = env.action_space().n

    n_atoms = config.get("N_ATOMS", 51)
    v_min = config.get("V_MIN", -10.0)
    v_max = config.get("V_MAX", 10.0)
    gamma = config.get("GAMMA", 0.99)
    batch_size = config.get("BATCH_SIZE", 32)
    tgt_freq = config.get("TARGET_NETWORK_FREQUENCY", 10000)
    train_freq = config.get("TRAIN_FREQUENCY", 4)
    start_e = config.get("START_E", 1.0)
    end_e = config.get("END_E", 0.01)
    total_timesteps = config.get("TOTAL_TIMESTEPS", 10_000_000)
    exploration_steps = float(config.get("EXPLORATION_FRACTION", 0.10) * total_timesteps)
    chunk_size = config.get("CHUNK_SIZE", 1000)
    num_chunks = total_timesteps // chunk_size
    delta_z = (v_max - v_min) / (n_atoms - 1)

    q_network = QNetworkPixel(n_actions, n_atoms) if pixel_based else QNetworkOC(n_actions, n_atoms)
    atoms = jnp.linspace(v_min, v_max, n_atoms)

    tx = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adam(config.get("LEARNING_RATE", 2.5e-4), eps=0.01 / batch_size),
    )

    init_params = q_network.init(q_key, jnp.zeros((1, *obs_shape)))
    q_state = C51TrainState.create(
        apply_fn=q_network.apply,
        params=init_params,
        target_params=init_params,
        atoms=atoms,
        tx=tx,
    )

    key, reset_key = jax.random.split(key)
    obs, env_state = env.reset(reset_key)

    obs_dtype = jnp.float32 if not pixel_based else jnp.uint8
    buffer = fbx.make_flat_buffer(
        max_length=config.get("BUFFER_SIZE", 100_000),
        min_length=config.get("LEARNING_STARTS", 10_000),
        sample_batch_size=batch_size,
    )
    fake_transition = Transition(
        obs=jnp.zeros(obs_shape, dtype=obs_dtype),
        next_obs=jnp.zeros(obs_shape, dtype=obs_dtype),
        action=jnp.int32(0),
        reward=jnp.float32(0.0),
        done=jnp.float32(0.0),
    )
    buffer_state = buffer.init(fake_transition)

    def update(q_state, obs_b, actions_b, next_obs_b, rewards_b, dones_b):
        next_pmfs = q_network.apply(q_state.target_params, next_obs_b)
        next_vals = (next_pmfs * q_state.atoms[None, None, :]).sum(-1)
        next_act = jnp.argmax(next_vals, axis=-1)
        next_pmfs = next_pmfs[jnp.arange(batch_size), next_act]
        next_atoms = rewards_b[:, None] + gamma * q_state.atoms[None, :] * (1.0 - dones_b[:, None])
        tz = jnp.clip(next_atoms, v_min, v_max)
        b = (tz - v_min) / delta_z
        l = jnp.clip(jnp.floor(b).astype(jnp.int32), 0, n_atoms - 1)
        u = jnp.clip(jnp.ceil(b).astype(jnp.int32), 0, n_atoms - 1)
        d_l = (u.astype(jnp.float32) + (l == u).astype(jnp.float32) - b) * next_pmfs
        d_u = (b - l.astype(jnp.float32)) * next_pmfs
        target_pmfs = jnp.zeros((batch_size, n_atoms))
        def project_sample(i, val):
            val = val.at[i, l[i]].add(d_l[i])
            val = val.at[i, u[i]].add(d_u[i])
            return val
        target_pmfs = jax.lax.fori_loop(0, batch_size, project_sample, target_pmfs)
        target_pmfs = jax.lax.stop_gradient(target_pmfs)
        def loss_fn(params):
            pmfs = q_network.apply(params, obs_b)
            q_pred = pmfs[jnp.arange(batch_size), actions_b]
            q_pred_clipped = jnp.clip(q_pred, 1e-5, 1 - 1e-5)
            loss = -(target_pmfs * jnp.log(q_pred_clipped)).sum(-1).mean()
            q_vals = (q_pred * q_state.atoms[None, :]).sum(-1)
            return loss, q_vals
        (loss, q_vals), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=grads)
        return loss, q_vals, q_state

    def step_once(carry, _):
        q_state, env_state, obs, done, buffer_state, key, global_step = carry
        epsilon = jnp.maximum(end_e, start_e + (end_e - start_e) * global_step.astype(jnp.float32) / exploration_steps)
        q_pmfs = q_network.apply(q_state.params, obs[None])
        q_vals = (q_pmfs * q_state.atoms[None, None, :]).sum(-1)[0]
        greedy = jnp.argmax(q_vals)
        key, act_key, exp_key = jax.random.split(key, 3)
        rnd_act = jax.random.randint(act_key, (), 0, n_actions)
        action = jax.lax.cond(jax.random.uniform(exp_key) < epsilon, lambda: rnd_act, lambda: greedy)
        next_obs, new_state, reward, terminated, truncated, info = env.step(env_state, action)
        next_done = jnp.logical_or(terminated, truncated).astype(jnp.float32)
        transition = Transition(obs=obs.astype(obs_dtype), next_obs=next_obs.astype(obs_dtype), action=action, reward=reward.astype(jnp.float32), done=next_done)
        buffer_state = buffer.add(buffer_state, transition)
        key, sample_key = jax.random.split(key)
        def do_update(args):
            q_state, buffer_state, sample_key = args
            batch = buffer.sample(buffer_state, sample_key)
            e = batch.experience.first
            loss, q_pred, new_q_state = update(q_state, e.obs.astype(jnp.float32), e.action, e.next_obs.astype(jnp.float32), e.reward, e.done)
            return new_q_state, loss, q_pred.mean()
        def no_update(args):
            q_state, _, _ = args
            return q_state, jnp.float32(0.0), jnp.float32(0.0)
        new_q_state, loss, avg_q = jax.lax.cond(jnp.logical_and(buffer.can_sample(buffer_state), global_step % train_freq == 0), do_update, no_update, (q_state, buffer_state, sample_key))
        new_q_state = jax.lax.cond(global_step % tgt_freq == 0, lambda s: s.replace(target_params=s.params), lambda s: s, new_q_state)
        new_carry = (new_q_state, new_state, next_obs, next_done, buffer_state, key, global_step + 1)
        return new_carry, (info, loss, avg_q)

    @jax.jit
    def collect_and_update(q_state, env_state, obs, done, buffer_state, key, global_step):
        init_carry = (q_state, env_state, obs, done, buffer_state, key, global_step)
        final_carry, outputs = jax.lax.scan(step_once, init_carry, None, length=chunk_size)
        return final_carry, outputs

    avg_returns = deque(maxlen=20)
    global_step = jnp.int32(0)
    done = jnp.float32(0.0)
    start_time = time.time()

    for chunk_idx in range(1, num_chunks + 1):
        update_t0 = time.time()
        (q_state, env_state, obs, done, buffer_state, key, global_step), (infos, losses, avg_qs) = collect_and_update(q_state, env_state, obs, done, buffer_state, key, global_step)
        update_time = time.time() - update_t0
        gs = int(global_step)
        if "returned_episode" in infos:
            finished = np.array(infos["returned_episode"])
            ep_rets = np.array(infos["returned_episode_returns"])
            for ret in ep_rets[finished]:
                avg_returns.append(float(ret))
                wandb.log({"charts/episodic_return": float(ret), "charts/avg_episodic_return": float(np.mean(avg_returns))}, step=gs)
        sps = int(gs / (time.time() - start_time))
        sps_update = int(chunk_size / update_time)
        epsilon = float(jnp.maximum(end_e, start_e + (end_e - start_e) * gs / exploration_steps))
        td_loss = float(losses[losses != 0].mean()) if (losses != 0).any() else 0.0
        q_val = float(avg_qs[avg_qs != 0].mean()) if (avg_qs != 0).any() else 0.0
        wandb.log({"charts/global_step": gs, "charts/epsilon": epsilon, "charts/SPS": sps, "charts/SPS_update": sps_update, "losses/td_loss": td_loss, "losses/avg_q_values": q_val}, step=gs)
        if chunk_idx % max(1, num_chunks // 20) == 0:
            print(f"step: {gs} / {total_timesteps} | SPS: {sps} | return: {float(np.mean(avg_returns)) if avg_returns else 0:.2f}")

    wandb.finish()
    print("[C51] Training complete.")
    return {}
