"""Hierarchy of Reward Machines (Toro Icarte et al., ICML 2018), JAX version.
"""

import os
import random
import time
from functools import partial
from typing import Any

import flashbax as fbx
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxatari
import numpy as np
import optax
import wandb
from flax import struct
from flax.training.train_state import TrainState
from jaxatari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
)

from agents.double_dqn_hrm.hrm_eval import evaluate_hrm
from reward_machines.options import OptionSpec, build_options
from reward_machines.reward_machine import RewardMachine
from reward_machines.reward_machine_wrapper import RewardMachineWrapper
from reward_machines.games.game_rm import GameRM
from reward_machines.rm_registry import GAME_RM_REGISTRY

_NEG_INF = -1e9

def make_env(env_id, mods=[], game_rm: GameRM | None = None, options: OptionSpec = None,
             eval=False, use_shaping=False, gamma=0.99):

    assert mods is None or isinstance(mods, list), "mods must be None or a list of strings"
    if mods is not None and len(mods) == 0:
        mods = None
    if not eval and mods is not None and len(mods) > 0:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        env = jaxatari.make(env_id, mods=mods or None)

        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=not eval,
            first_fire=True,
            noop_max=30,
            full_action_space=False,
        )
        env = ObjectCentricWrapper(
            env,
            frame_stack_size=4,
            frame_skip=4,
            clip_reward=not eval
        )
        env = FlattenObservationWrapper(
            NormalizeObservationWrapper(env, dtype=jnp.float32)
        )
        # use_crm=False: HRM builds its own transitions, the CRM fan-out would
        # only be dead weight in the info dict.
        if game_rm is not None:
            rm = RewardMachine(game_rm)
            env = RewardMachineWrapper(
                env, reward_machine=rm, use_crm=False,
                use_shaping=use_shaping, gamma=gamma, options=options,
            )
        return LogWrapper(env)

    return thunk

@struct.dataclass
class OptionTimeStep:
    """One primitive step, usable to update every option head. """

    obs: jnp.ndarray
    action: jnp.ndarray
    option_rewards: jnp.ndarray
    next_obs: jnp.ndarray
    env_done: jnp.ndarray


@struct.dataclass
class MetaTimeStep:
    """One SMDP step: the whole execution of a single option."""

    obs: jnp.ndarray
    option: jnp.ndarray
    ret: jnp.ndarray
    next_obs: jnp.ndarray
    discount: jnp.ndarray
    done: jnp.ndarray


@struct.dataclass
class OptionExecState:
    """Per-env bookkeeping for the option currently in flight."""

    option: jnp.ndarray
    active: jnp.ndarray
    start_obs: jnp.ndarray
    ret: jnp.ndarray
    discount: jnp.ndarray
    length: jnp.ndarray


class OptionQNetwork(nn.Module):

    num_options: int
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        leading = x.shape[:-1]
        x = nn.Dense(512, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        q = nn.Dense(
            self.num_options * self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
        )(x)
        return q.reshape(*leading, self.num_options, self.action_dim)


class MetaQNetwork(nn.Module):

    num_options: int
    hidden: tuple = (512, 256)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(512, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        return nn.Dense(
            self.num_options, kernel_init=nn.initializers.orthogonal(0.01)
        )(x)


@struct.dataclass
class RingBufferState:
    data: Any
    ptr: jnp.ndarray


class MaskedRingBuffer:
    def __init__(self, capacity: int, sample_batch_size: int, min_length: int):
        self.capacity = capacity
        self.sample_batch_size = sample_batch_size
        self.min_length = min_length

    def init(self, item: Any) -> RingBufferState:
        """`item` is a pytree of per-row shapes (no leading batch dim)."""
        data = jax.tree.map(
            lambda x: jnp.zeros((self.capacity + 1,) + x.shape, x.dtype), item
        )
        return RingBufferState(data=data, ptr=jnp.int32(0))

    def add(self, state: RingBufferState, batch: Any, valid: jnp.ndarray) -> RingBufferState:
        """Append the rows of `batch` where `valid` is True."""
        offsets = jnp.cumsum(valid) - 1
        dest = jnp.where(valid, (state.ptr + offsets) % self.capacity, self.capacity)
        data = jax.tree.map(lambda buf, x: buf.at[dest].set(x), state.data, batch)
        return state.replace(data=data, ptr=state.ptr + jnp.sum(valid).astype(jnp.int32))

    def sample(self, state: RingBufferState, key) -> Any:
        size = jnp.minimum(state.ptr, self.capacity)
        idx = jax.random.randint(
            key, (self.sample_batch_size,), 0, jnp.maximum(size, 1)
        )
        return jax.tree.map(lambda buf: buf[idx], state.data)

    def can_sample(self, state: RingBufferState) -> jnp.ndarray:
        return state.ptr >= self.min_length

class HRMTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int


def _linear_eps(t, start, finish, anneal):
    return jnp.clip((finish - start) / anneal * t + start, finish)


def single_run(config: dict):
    # Keep TARGET_UPDATE_INTERVAL propotional to NUM_ENVS
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"])
    config["TARGET_UPDATE_INTERVAL"] = max(1, round(
        config["TARGET_UPDATE_GRAD_STEPS"] * config["TRAINING_INTERVAL"] / config["GRAD_STEPS_PER_ITER"]
    ))
    config["META_TARGET_UPDATE_INTERVAL"] = max(1, round(
        config["TARGET_UPDATE_GRAD_STEPS"] * config["TRAINING_INTERVAL"]
    ))

    gamma = config["GAMMA"]
    n_envs = config["NUM_ENVS"]
    k_max = config["OPTION_MAX_STEPS"]

    # Can't use pixexl_based for hrm
    assert not config.get("PIXEL_BASED", False), "HRM requires PIXEL_BASED: False"


    run_name = f'{config["ENV_ID"]}_hrm_{config["EXP_NAME"]}_{config["SEED"]}'
    wandb.init(project=config["PROJECT"], entity=config["ENTITY"], config=config,
               name=run_name, save_code=True)

    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    # --- reward machine + option structure (Python time, static) ---
    game_rm = GAME_RM_REGISTRY[config["GAME_RM"]]()
    options = build_options(game_rm)
    print(options.describe())
    num_options = options.num_options
    num_rm_states = game_rm.num_states()
    availability = options.availability  # (num_states, num_options)

    env = make_env(config["ENV_ID"], options=options, game_rm=game_rm,
                   mods=list(config.get("TRAIN_MODS", [])),
                   use_shaping=config.get("USE_SHAPING", False), gamma=gamma)()

    aug_dim = env.observation_space().shape[0]
    raw_dim = aug_dim - num_rm_states
    n_actions = env.action_space().n

    @jax.jit
    def vmap_reset(k):
        obs, state = jax.vmap(env.reset)(k)
        return obs.reshape(k.shape[0], aug_dim), state

    @jax.jit
    def vmap_step(state, action):
        obs, state, reward, term, trunc, info = jax.vmap(env.step)(state, action)
        return obs.reshape(action.shape[0], aug_dim), state, reward, term | trunc, info

    key, _rng = jax.random.split(key)
    init_obs, env_state = vmap_reset(jax.random.split(_rng, n_envs))

    # --- learning rates (mirrors the linear_schedule in dqn.py) ---
    warmup_iters = config["LEARNING_STARTS"] // n_envs
    num_grad_updates = max(
        1, (config["NUM_UPDATES"] - warmup_iters) // config["TRAINING_INTERVAL"]
    )

    def linear_schedule(base):
        return lambda count: base * (1.0 - (count / num_grad_updates))

    decay = config.get("LR_LINEAR_DECAY", False)
    option_lr = linear_schedule(config["LR"]) if decay else config["LR"]
    meta_lr = linear_schedule(config["META_LR"]) if decay else config["META_LR"]

    # --- networks ---
    option_net = OptionQNetwork(num_options=num_options, action_dim=n_actions)
    meta_net = MetaQNetwork(num_options=num_options)

    key, k_opt, k_meta = jax.random.split(key, 3)
    option_ts = HRMTrainState.create(
        apply_fn=option_net.apply,
        params=(p := option_net.init(k_opt, jnp.zeros(raw_dim))),
        target_network_params=jax.tree_util.tree_map(jnp.copy, p),
        tx=optax.chain(optax.clip_by_global_norm(10.0),
                       optax.adam(option_lr)),
        timesteps=0, n_updates=0,
    )
    meta_ts = HRMTrainState.create(
        apply_fn=meta_net.apply,
        params=(p := meta_net.init(k_meta, jnp.zeros(aug_dim))),
        target_network_params=jax.tree_util.tree_map(jnp.copy, p),
        tx=optax.chain(optax.clip_by_global_norm(10.0),
                       optax.adam(meta_lr)),
        timesteps=0, n_updates=0,
    )

    # --- buffers ---
    # Option transitions: exactly one per env per step, always valid -> flashbax.
    option_buffer = fbx.make_flat_buffer(
        max_length=config["BUFFER_SIZE"],
        min_length=config["BUFFER_BATCH_SIZE"],
        sample_batch_size=config["BUFFER_BATCH_SIZE"],
        add_sequences=False, add_batch_size=n_envs,
    )
    option_buffer = option_buffer.replace(
        init=jax.jit(option_buffer.init),
        add=jax.jit(option_buffer.add, donate_argnums=0),
        sample=jax.jit(option_buffer.sample),
        can_sample=jax.jit(option_buffer.can_sample),
    )
    option_bs = option_buffer.init(
        OptionTimeStep(
            obs=jnp.zeros(raw_dim, jnp.float32), action=jnp.int32(0),
            option_rewards=jnp.zeros(num_options, jnp.float32),
            next_obs=jnp.zeros(raw_dim, jnp.float32), env_done=jnp.bool_(False),
        )
    )

    # Meta transitions: only when an option ends -> masked ring buffer.
    meta_buffer = MaskedRingBuffer(
        capacity=config["META_BUFFER_SIZE"],
        sample_batch_size=config["META_BATCH_SIZE"],
        min_length=config["META_BATCH_SIZE"],
    )
    meta_bs = meta_buffer.init(
        MetaTimeStep(
            obs=jnp.zeros(aug_dim, jnp.float32), option=jnp.int32(0),
            ret=jnp.float32(0), next_obs=jnp.zeros(aug_dim, jnp.float32),
            discount=jnp.float32(0), done=jnp.bool_(False),
        )
    )

    exec_state = OptionExecState(
        option=jnp.zeros(n_envs, jnp.int32),
        active=jnp.zeros(n_envs, bool),  # forces a meta choice on step 0
        start_obs=jnp.zeros((n_envs, aug_dim), jnp.float32),
        ret=jnp.zeros(n_envs, jnp.float32),
        discount=jnp.ones(n_envs, jnp.float32),
        length=jnp.zeros(n_envs, jnp.int32),
    )

    def u_of(aug_obs):
        """Recover the RM state from the one-hot tail. Avoids depending on the
        wrapper nesting order of env_state."""
        return jnp.argmax(aug_obs[..., -num_rm_states:], axis=-1)

    def eps_greedy(rng, q, t, cfg_prefix):
        eps = _linear_eps(t, config[f"{cfg_prefix}EPSILON_START"],
                          config[f"{cfg_prefix}EPSILON_FINISH"],
                          config[f"{cfg_prefix}EPSILON_ANNEAL_TIME"])
        rng_a, rng_e = jax.random.split(rng)
        greedy = jnp.argmax(q, axis=-1)
        rand = jax.random.randint(rng_a, greedy.shape, 0, q.shape[-1])
        return jnp.where(jax.random.uniform(rng_e, greedy.shape) < eps, rand, greedy)

    def eps_greedy_masked(rng, q, mask, t):
        eps = _linear_eps(t, config["META_EPSILON_START"],
                          config["META_EPSILON_FINISH"],
                          config["META_EPSILON_ANNEAL_TIME"])
        rng_g, rng_e = jax.random.split(rng)
        greedy = jnp.argmax(jnp.where(mask, q, _NEG_INF), axis=-1)
        gumbel = jax.random.gumbel(rng_g, q.shape)
        rand = jnp.argmax(jnp.where(mask, gumbel, _NEG_INF), axis=-1)
        return jnp.where(jax.random.uniform(rng_e, greedy.shape) < eps, rand, greedy)

    def _option_learn(ts, buf_state, rng):
        batch = option_buffer.sample(buf_state, rng).experience.first

        q_next_online = option_net.apply(ts.params, batch.next_obs)  # (B,O,A)
        a_star = jnp.argmax(q_next_online, axis=-1)  # (B,O)
        q_next_target = option_net.apply(ts.target_network_params, batch.next_obs)
        q_next = jnp.take_along_axis(q_next_target, a_star[..., None], -1).squeeze(-1)

        # An option's own task terminates when its formula becomes true.
        done_o = (batch.option_rewards > 0) | batch.env_done[:, None]
        target = batch.option_rewards + (1.0 - done_o) * gamma * q_next  # (B,O)

        def _loss(params):
            q = option_net.apply(params, batch.obs)  # (B,O,A)
            act = jnp.broadcast_to(batch.action[:, None, None], q.shape[:-1] + (1,))
            chosen = jnp.take_along_axis(q, act, -1).squeeze(-1)  # (B,O)
            return jnp.mean(optax.huber_loss(chosen, target, delta=1.0))

        loss, grads = jax.value_and_grad(_loss)(ts.params)
        ts = ts.apply_gradients(grads=grads).replace(n_updates=ts.n_updates + 1)
        return ts, loss

    def _option_learn_k(ts, buf_state, rng):
        def _grad_step(carry, _):
            ts, rng = carry
            rng, step_rng = jax.random.split(rng)
            ts, loss = _option_learn(ts, buf_state, step_rng)
            return (ts, rng), loss

        (ts, _), losses = jax.lax.scan(
            _grad_step, (ts, rng), None, config["GRAD_STEPS_PER_ITER"]
        )
        return ts, losses.mean()

    def _meta_learn(ts, buf_state, rng):
        """SMDP Q-learning:
        y = sum_k gamma^k r_{t+k} + gamma^K max_{o' in avail(u')} Q^-(s', u', o')
        """
        batch = meta_buffer.sample(buf_state, rng)
        mask_next = availability[u_of(batch.next_obs)]  # (B,O)

        q_next_online = jnp.where(
            mask_next, meta_net.apply(ts.params, batch.next_obs), _NEG_INF
        )
        o_star = jnp.argmax(q_next_online, axis=-1)
        q_next_target = meta_net.apply(ts.target_network_params, batch.next_obs)
        q_next = jnp.take_along_axis(q_next_target, o_star[:, None], -1).squeeze(-1)

        target = batch.ret + (1.0 - batch.done) * batch.discount * q_next

        def _loss(params):
            q = meta_net.apply(params, batch.obs)
            chosen = jnp.take_along_axis(q, batch.option[:, None], -1).squeeze(-1)
            return jnp.mean(optax.huber_loss(chosen, target, delta=1.0))

        loss, grads = jax.value_and_grad(_loss)(ts.params)
        ts = ts.apply_gradients(grads=grads).replace(n_updates=ts.n_updates + 1)
        return ts, loss

    def _update_step(runner_state, unused):
        (option_ts, meta_ts, option_bs, meta_bs,
         env_state, last_obs, exec_state, rng) = runner_state

        raw_obs = last_obs[..., :raw_dim]
        rng, rng_m, rng_a, rng_lo, rng_lm = jax.random.split(rng, 5)

        # 1. meta-controller commits a new option wherever none is in flight.
        mask = availability[u_of(last_obs)]
        q_meta = meta_net.apply(meta_ts.params, last_obs)
        proposed = eps_greedy_masked(rng_m, q_meta, mask, meta_ts.timesteps)

        started = ~exec_state.active
        option = jnp.where(started, proposed, exec_state.option)
        start_obs = jnp.where(started[:, None], last_obs, exec_state.start_obs)
        ret = jnp.where(started, 0.0, exec_state.ret)
        discount = jnp.where(started, 1.0, exec_state.discount)
        length = jnp.where(started, 0, exec_state.length)

        # 2. the active option's policy picks a primitive action.
        q_all = option_net.apply(option_ts.params, raw_obs)  # (N,O,A)
        q_opt = jnp.take_along_axis(
            q_all, option[:, None, None], axis=1
        ).squeeze(1)  # (N,A)
        action = eps_greedy(rng_a, q_opt, option_ts.timesteps, "")

        obs, env_state, reward, done, info = vmap_step(env_state, action)
        option_ts = option_ts.replace(timesteps=option_ts.timesteps + n_envs)

        # 3. option replay: one row per env, all heads get a pseudo-reward.
        option_bs = option_buffer.add(
            option_bs,
            OptionTimeStep(
                obs=raw_obs, action=action,
                option_rewards=info["option_rewards"],
                next_obs=obs[..., :raw_dim], env_done=done,
            ),
        )

        # 4. accumulate the discounted SMDP return of the running option.
        ret = ret + discount * reward
        discount = discount * gamma
        length = length + 1

        # 5. terminate on subgoal, episode end, or timeout. The timeout is not
        #    in the paper but is required in practice: without it a stuck
        #    option never releases control and never produces a meta sample.
        u_changed = u_of(obs) != u_of(last_obs)
        terminate = info["option_terminate"] | done | (length >= k_max) | u_changed
        meta_bs = meta_buffer.add(
            meta_bs,
            MetaTimeStep(obs=start_obs, option=option, ret=ret, next_obs=obs,
                         discount=discount, done=done),
            valid=terminate,
        )
        meta_ts = meta_ts.replace(
            timesteps=meta_ts.timesteps + jnp.sum(terminate).astype(jnp.int32)
        )

        exec_state = OptionExecState(
            option=option, active=~terminate, start_obs=start_obs,
            ret=ret, discount=discount, length=length,
        )

        # 6. gradient steps
        env_iters = option_ts.timesteps // n_envs
        can_learn = (
            option_buffer.can_sample(option_bs)
            & (option_ts.timesteps > config["LEARNING_STARTS"])
            & (env_iters % config["TRAINING_INTERVAL"] == 0)
        )
        option_ts, option_loss = jax.lax.cond(
            can_learn, lambda ts, r: _option_learn_k(ts, option_bs, r),
            lambda ts, r: (ts, jnp.float32(0)), option_ts, rng_lo,
        )
        meta_ts, meta_loss = jax.lax.cond(
            can_learn & meta_buffer.can_sample(meta_bs),
            lambda ts, r: _meta_learn(ts, meta_bs, r),
            lambda ts, r: (ts, jnp.float32(0)), meta_ts, rng_lm,
        )

        def _sync(ts, interval):
            return jax.lax.cond(
                env_iters % interval == 0,
                lambda s: s.replace(target_network_params=optax.incremental_update(
                    s.params, s.target_network_params, config["TAU"])),
                lambda s: s, ts,
            )

        option_ts = _sync(option_ts, config["TARGET_UPDATE_INTERVAL"])
        meta_ts = _sync(meta_ts, config["META_TARGET_UPDATE_INTERVAL"])

        metrics = {
            "timesteps": option_ts.timesteps,
            "option_loss": option_loss,
            "meta_loss": meta_loss,
            "returns_sum": jnp.sum(info["returned_episode_returns"] * info["returned_episode"]),
            "returns_count": jnp.sum(info["returned_episode"]),
            "env_reward": info["env_reward"].mean(),
            "rm_reward": info["rm_reward"].mean(),
            "option_len": jnp.mean(jnp.where(terminate, length, 0).sum()
                                   / jnp.maximum(jnp.sum(terminate), 1)),
            "option_hist": jnp.sum(jax.nn.one_hot(option, num_options), axis=0),
            "subgoal_hits": info["option_rewards"].sum(axis=0),
        }
        runner_state = (option_ts, meta_ts, option_bs, meta_bs,
                        env_state, obs, exec_state, rng)
        return runner_state, metrics

    def save_model(option_params, meta_params, step):
        """Write a checkpoint. No-op when SAVE_PATH is not configured."""
        if config.get("SAVE_PATH") is None:
            return
        model_path = f'{config["SAVE_PATH"]}/{run_name}/step_{step}.hrm_model'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(
                [config, option_params, meta_params]))
        print(f"model saved to {model_path}")

    def eval_model(option_params, meta_params, step):
        """Evaluate on the base game and on each mod separately."""
        eval_mods = config.get("EVAL_MODS") or config.get("TRAIN_MODS") or []
        eval_configs = [([], "default")] + [([mod], mod) for mod in eval_mods]

        metrics = {}
        for mods_config, mod_label in eval_configs:
            print(f"Evaluating on {mod_label} ...")
            episodic_returns, env_states = evaluate_hrm(
                option_params,
                meta_params,
                partial(
                    make_env,
                    options=options,
                    game_rm=game_rm,
                    mods=mods_config,
                    eval=True,
                    gamma=gamma,
                ),
                config["ENV_ID"],
                eval_episodes=10,
                option_net=option_net,
                meta_net=meta_net,
                options=options,
                num_rm_states=num_rm_states,
                seed=config["SEED"],
                option_max_steps=k_max,
            )
            returns_np = jax.device_get(episodic_returns)
            mean_ret = float(np.mean(returns_np))
            metrics[mod_label] = mean_ret

            log = {
                f"eval/episodic_return_{mod_label}": mean_ret,
                f"eval/episodic_return_{mod_label}_std": float(np.std(returns_np)),
                f"eval/episodic_return_{mod_label}_min": float(np.min(returns_np)),
                f"eval/episodic_return_{mod_label}_max": float(np.max(returns_np)),
            }
            if config.get("CAPTURE_VIDEO", False):
                renderer = jaxatari.make(config["ENV_ID"], mods=mods_config or None).renderer
                frames = jnp.transpose(jax.vmap(renderer.render)(env_states), (0, 3, 1, 2))
                log[f"eval/video_{mod_label}"] = wandb.Video(
                    np.array(frames), fps=30, format="mp4")
                print(f"Video (eval) logged with {frames.shape[0]} frames.")
            wandb.log(log, step=step)
        return metrics

    updates_per_chunk = max(1, config["EVAL_EVERY"] // n_envs)
    num_chunks = config["NUM_UPDATES"] // updates_per_chunk
    assert num_chunks > 0, (
        f"num_chunks=0: EVAL_EVERY={config['EVAL_EVERY']} too large for "
        f"NUM_UPDATES={config['NUM_UPDATES']} (NUM_ENVS={n_envs})"
    )

    @partial(jax.jit, donate_argnums=(0,))
    def train_chunk(rs):
        return jax.lax.scan(_update_step, rs, None, updates_per_chunk)

    key, _rng = jax.random.split(key)
    runner_state = (option_ts, meta_ts, option_bs, meta_bs,
                    env_state, init_obs, exec_state, _rng)
    
    episode_returns = []
    start_time = time.time()
    for _ in range(1, num_chunks + 1):
        runner_state, m = train_chunk(runner_state)
        option_ts, meta_ts = runner_state[0], runner_state[1]
        global_step = int(option_ts.timesteps)
        elapsed = time.time() - start_time

        log = {
            "charts/env_reward_per_step": float(m["env_reward"].mean()),
            "charts/rm_reward_per_step": float(m["rm_reward"].mean()),
            "charts/mean_option_length": float(m["option_len"].mean()),
            "losses/option_loss": float(m["option_loss"].mean()),
            "losses/meta_loss": float(m["meta_loss"].mean()),
            "charts/SPS": int(global_step / elapsed),
        }
        count = float(m["returns_count"].sum())
        if count > 0:
            avg_return = float(m["returns_sum"].sum()) / count
            log["charts/avg_episodic_return"] = avg_return
            episode_returns.append(avg_return)

        hist = np.asarray(m["option_hist"].sum(axis=0))
        hits = np.asarray(m["subgoal_hits"].sum(axis=0))
        for i, name in enumerate(options.names):
            log[f"options/selected_{i}_{name}"] = float(hist[i])
            log[f"options/achieved_{i}_{name}"] = float(hits[i])
        wandb.log(log, step=global_step)

        if config.get("EVAL_DURING_TRAIN", False):
            eval_metrics = eval_model(option_ts.params, meta_ts.params, global_step)

    print(f"Total train time: {(time.time() - start_time) / 60:.2f} min.")
    final_step = int(option_ts.timesteps)
    save_model(option_ts.params, meta_ts.params, final_step)
    eval_metrics = eval_model(option_ts.params, meta_ts.params, final_step)
    if episode_returns:
        n_last = max(1, len(episode_returns) // 5)  # last ~20% of training
        wandb.run.summary["final/return_last_20pct"] = float(np.mean(episode_returns[-n_last:]))
        wandb.run.summary["final/return_mean_all"] = float(np.mean(episode_returns))
    wandb.finish()
    return eval_metrics
