# Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
import os

# Limit JAX GPU memory usage to avoid OOM / fragmentation issues.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

import random
import subprocess
import time
import tempfile
from functools import partial
from typing import Optional

import flashbax as fbx
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxatari.wrappers import (
    AtariWrapper,
    PixelObsWrapper,
    ObjectCentricWrapper,
    LogWrapper,
    FlattenObservationWrapper,
    NormalizeObservationWrapper,
    ContinuousActionWrapper,
)
from jaxatari import spaces

from rtpt import RTPT
from agents.sac_cale.sac_eval import evaluate

# Optional benchmark tool
def get_gpu_stats():
    """Return (memory_used_MB, memory_total_MB, utilization_percent) for the first GPU."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        line = out.stdout.strip().splitlines()[0]
        used, total, util = [int(x) for x in line.split(",")]
        return used, total, util
    except Exception:
        return -1, -1, -1


# ---------------------------------------------------------------------------
# Optional performance instrumentation for benchmarking. (opt-in via SAC_PROFILE=1).
#
# It wraps the training-loop sections in `jax.block_until_ready` so that wall
# time can be attributed to a specific section instead of to "whatever the async
# dispatch pipeline happened to overlap". Blocking *serialises* the pipeline, so
# numbers gathered in this mode are a critical-path breakdown and NOT the
# throughput of a normal run. The values that are computed are unchanged, so the
# training trajectory is identical; the mode is only a measuring instrument.
#
# When SAC_PROFILE is unset, `prof` is None and each instrumented site pays a
# single `if prof:` boolean test.
# ---------------------------------------------------------------------------
PROFILE_ENABLED = os.environ.get("SAC_PROFILE", "").strip().lower() in ("1", "true", "yes", "on")

# Populated by `single_run` when profiling is enabled, so benchmark scripts can
# read the breakdown programmatically.
LAST_PROFILE = None


class SectionProfiler:
    """Accumulates per-section wall time (one entry per instrumented call site)."""

    def __init__(self):
        self.times = {}
        self.calls = {}

    def add(self, name, t0, value=None):
        """Record the elapsed time since ``t0``; wait for ``value`` first when given."""
        if value is not None:
            leaves = jax.tree_util.tree_leaves(value)
            if leaves:
                jax.block_until_ready(leaves)
        dt = time.perf_counter() - t0
        self.times[name] = self.times.get(name, 0.0) + dt
        self.calls[name] = self.calls.get(name, 0) + 1

    @property
    def total(self):
        return sum(self.times.values())

    @property
    def training_total(self):
        """Sum of the sections that run once per environment step (excludes warmup-fill)."""
        return sum(v for k, v in self.times.items() if not k.startswith("warmup/"))

    def summary(self, steps=None, wall_s=None, title=None):
        lines = []
        if title:
            lines.append(title)
        lines.append(f"{'section':<24}{'total s':>10}{'calls':>9}{'ms/call':>10}{'% profiled':>12}")
        for name, secs in sorted(self.times.items(), key=lambda kv: -kv[1]):
            calls = self.calls[name]
            pct = 100.0 * secs / self.total if self.total else 0.0
            lines.append(f"{name:<24}{secs:>10.2f}{calls:>9}{1000.0 * secs / calls:>10.3f}{pct:>11.1f}%")
        lines.append(f"{'PROFILED TOTAL':<24}{self.total:>10.2f}")
        if steps and self.training_total:
            lines.append(f"{'training ms/env-step (profiled)':<24}"
                         f"{1000.0 * self.training_total / steps:>10.3f}")
        if wall_s and steps:
            lines.append(f"{'training window s (real)':<24}{wall_s:>10.2f}   "
                         f"-> {steps / wall_s:7.1f} SPS (async, unblocked)")
        return "\n".join(lines)


def make_env(
    env_id: str,
    mods: Optional[list] = None,
    pixel_based: bool = True,
    native_downscaling: bool = True,
    eval: bool = False,
    tau_cale: float = 0.5,
):
    if mods is None:
        mods = []
    if not eval and len(mods) > 0:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        import jaxatari
        env = jaxatari.make(env_id, mods=mods)
        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=not eval,
            first_fire=True,
            noop_max=30,
            full_action_space= True,
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
        env = ContinuousActionWrapper(env, tau=tau_cale)

        return env
    return thunk


# ---------- Networks ----------
class CNNEncoder(nn.Module):
    """CNN Encoder for pixel observations."""
    @nn.compact
    def __call__(self, x):
        if x.ndim == 5:
            b, stack, h, w, c = x.shape
            x = jnp.transpose(x, (0, 2, 3, 1, 4))
            x = x.reshape((b, h, w, stack * c))
        elif x.ndim == 4:
            x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID",
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID",
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID",
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class MLPEncoder(nn.Module):
    """MLP for object-centric observations."""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class Actor(nn.Module):
    """Gaussian policy for continuous actions."""
    action_dim: int
    log_std_min: float = -1.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        log_std = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        log_std = jnp.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        return mean, log_std


class SoftQNetwork(nn.Module):
    """Soft Q(s,a) network (Critic) """
    @nn.compact
    def __call__(self, x, a):
        r, theta, fire = a[..., 0], a[..., 1], a[..., 2]
        a_feat = jnp.stack(
            [r, jnp.sin(theta), jnp.cos(theta), fire, r * jnp.cos(theta), r * jnp.sin(theta)],
            axis=-1,
        )
        a_feat = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.1))(a_feat)
        a_feat = nn.relu(a_feat)
        x = jnp.concatenate([x, a_feat], axis=-1)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return nn.Dense(1, kernel_init=orthogonal(1.0))(x)


@flax.struct.dataclass
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray  # Stores true env termination (terminated)


def render_env_states(env_states, env_id, mods=None, chunk_size=256, max_frames=None):
    """Render a batch of env states into a (T, C, H, W) uint8 array for wandb.Video."""
    import jaxatari

    renderer = jaxatari.make(env_id, mods=mods).renderer
    # Every array leaf is stacked along the same frame axis; scalar leaves are ignored.
    lengths = [
        int(leaf.shape[0])
        for leaf in jax.tree_util.tree_leaves(env_states)
        if getattr(leaf, "ndim", 0) > 0
    ]
    if not lengths:
        return None

    num_frames = max(lengths)
    if max_frames is not None:
        num_frames = min(num_frames, int(max_frames))
    if num_frames <= 0:
        return None

    chunks = None
    offset = 0
    for start in range(0, num_frames, chunk_size):
        stop = min(start + chunk_size, num_frames)
        chunk_states = jax.tree_util.tree_map(
            lambda x: x[start:stop] if hasattr(x, "ndim") and x.ndim > 0 else x, env_states
        )
        frames = jax.vmap(renderer.render)(chunk_states)  # (t, H, W, C) on device
        frames = np.asarray(jax.device_get(frames))       # host copy; device chunk can be reused
        frames = frames.transpose(0, 3, 1, 2)             # (t, H, W, C) -> (t, C, H, W)
        if chunks is None:  # single C-contiguous output buffer, no second copy
            chunks = np.empty((num_frames,) + frames.shape[1:], dtype=frames.dtype)
        chunks[offset:offset + frames.shape[0]] = frames
        offset += frames.shape[0]
    return chunks


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    if isinstance(config.get("TRAIN_MODS"), list):
        config["TRAIN_MODS"] = tuple(config["TRAIN_MODS"])
    if isinstance(config.get("EVAL_MODS"), list):
        config["EVAL_MODS"] = tuple(config["EVAL_MODS"])

    run_name = f'{config["ENV_ID"]}_{config["EXP_NAME"]}_{"oc" if not config["PIXEL_BASED"] else "pixel"}_{config["SEED"]}'

    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
    )

    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])
    key, encoder_key, actor_key, qf1_key, qf2_key = jax.random.split(key, 5)

    env = make_env(
        config["ENV_ID"],
        list(config.get("TRAIN_MODS", [])),
        config["PIXEL_BASED"],
        config.get("NATIVE_DOWNSCALING", True),
        False,
        config.get("TAU_CALE",0.5)
    )()
    obs_space = env.observation_space()
    assert isinstance(obs_space, spaces.Box), "SAC requires Box observation space."
    action_space = env.action_space()
    assert isinstance(action_space, spaces.Box), "ContinuousActionWrapper should give Box action space."
    action_dim = action_space.shape[0]
    low = jnp.array(action_space.low)
    high = jnp.array(action_space.high)
    action_scale = (high - low) / 2.0
    action_bias = (high + low) / 2.0

    reward_scale = float(config.get("REWARD_SCALE_FACTOR", 10.0))
    # Soft penalty bounds for log-alpha; the farther it goes beyond a bound,
    # the stronger the force pulling it back (the penalty grows quadratically
    # as log-alpha moves beyond either bound).
    log_alpha_min = float(config.get("LOG_ALPHA_MIN", -5.0))
    log_alpha_max = float(config.get("LOG_ALPHA_MAX", 5.0))

    @jax.jit
    def vmap_reset(key):
        obs, state = jax.vmap(env.reset)(key)
        return obs, state

    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        next_done = jnp.logical_or(terminated, truncated)
        return next_obs, state, reward, terminated, next_done, info

    encoder_cls = CNNEncoder if config["PIXEL_BASED"] else MLPEncoder
    shared_encoder = encoder_cls()
    actor = Actor(action_dim=action_dim)
    qf1, qf2 = SoftQNetwork(), SoftQNetwork()

    sample_obs = jnp.zeros((1,) + obs_space.shape, dtype=jnp.uint8 if config["PIXEL_BASED"] else jnp.float32)
    dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

    encoder_params = shared_encoder.init(encoder_key, sample_obs)
    hidden = shared_encoder.apply(encoder_params, sample_obs)

    actor_params = actor.init(actor_key, hidden)
    qf1_params = qf1.init(qf1_key, hidden, dummy_action)
    qf2_params = qf2.init(qf2_key, hidden, dummy_action)

    encoder_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=config.get("ENCODER_LR", config["Q_LR"]), eps=1e-8),
    )
    q_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=config["Q_LR"], eps=1e-8),
    )
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=config["POLICY_LR"], eps=1e-8),
    )

    encoder_state = TrainState.create(apply_fn=None, params=encoder_params, tx=encoder_optimizer)
    qf1_state = TrainState.create(apply_fn=None, params=qf1_params, tx=q_optimizer)
    qf2_state = TrainState.create(apply_fn=None, params=qf2_params, tx=q_optimizer)
    actor_state = TrainState.create(apply_fn=None, params=actor_params, tx=actor_optimizer)

    encoder_target_params = encoder_params
    qf1_target_params = qf1_params
    qf2_target_params = qf2_params
    actor_target_params = actor_params

    # CleanRL default
    target_entropy = -float(action_dim)

    if config["AUTOTUNE"]:
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": jnp.zeros((1,), dtype=jnp.float32)},
            tx=optax.adam(learning_rate=config.get("ALPHA_LR", config["Q_LR"]), eps=1e-8),
        )
        alpha = jnp.exp(alpha_state.params["log_alpha"])
    else:
        alpha = jnp.asarray(config["ALPHA"], dtype=jnp.float32)
        alpha_state = None

    # ---------- CALE action transformation helpers ----------
    def gaussian_log_prob(x, mean, log_std):
        std = jnp.exp(log_std)
        return -0.5 * (((x - mean) / (std + 1e-8)) ** 2 + 2.0 * jnp.log(std + 1e-8) + jnp.log(2.0 * jnp.pi))

    def tanh_affine_log_jacobian(u):
        tanh_corr = 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
        return tanh_corr + jnp.log(action_scale)

    def sample_env_action(mean, log_std, key):
        noise = jax.random.normal(key, shape=mean.shape)
        u = mean + jnp.exp(log_std) * noise
        action_tanh = jnp.tanh(u)
        action_env = action_bias + action_scale * action_tanh
        log_prob = gaussian_log_prob(u, mean, log_std).sum(axis=-1)
        log_prob = log_prob - tanh_affine_log_jacobian(u).sum(axis=-1)
        return action_env, log_prob, u

    def deterministic_env_action(mean):
        return action_bias + action_scale * jnp.tanh(mean)

    # ---------- JIT functions ----------
    @jax.jit
    def sample_action(encoder_state, actor_state, obs, key):
        key, subkey = jax.random.split(key)
        hidden = shared_encoder.apply(encoder_state.params, obs)
        mean, log_std = actor.apply(actor_state.params, hidden)
        key, action_key = jax.random.split(subkey)
        action_env, _, _ = sample_env_action(mean, log_std, action_key)
        return action_env, key

    @jax.jit
    def get_det_action(encoder_state, actor_state, obs):
        hidden = shared_encoder.apply(encoder_state.params, obs)
        mean, _ = actor.apply(actor_state.params, hidden)
        return deterministic_env_action(mean)

    def remove_env_dim(x):
        if x.ndim >= 2 and x.shape[1] > 1:
            return x.reshape((-1,) + x.shape[2:])
        if x.ndim >= 2 and x.shape[1] == 1:
            return x.squeeze(1)
        return x

    def as_dtype(x, dtype):
        """Avoid an unnecessary device dispatch when ``x`` already has ``dtype``."""
        return x if x.dtype == dtype else x.astype(dtype)

    def _g_norm(*pytrees):
        flat, _ = jax.flatten_util.ravel_pytree(pytrees)
        return jnp.sqrt(jnp.sum(flat ** 2))

    def _to_float(x):
        return float(np.asarray(x).reshape(-1)[0])

    @jax.jit
    def update_qf(
        encoder_state, qf1_state, qf2_state, actor_state,
        encoder_target_params, qf1_target_params, qf2_target_params,
        alpha, batch, key,
    ):

        # Sample the next action from the online actor.
        next_hidden = shared_encoder.apply(encoder_state.params, batch.next_obs)
        next_mean, next_log_std = actor.apply(actor_state.params, next_hidden)
        key, subkey = jax.random.split(key)
        next_action, next_log_prob, _ = sample_env_action(next_mean, next_log_std, subkey)
        # Evaluate target Q-functions with the target encoder, reducing target drift
        z_next = shared_encoder.apply(encoder_target_params, batch.next_obs)
        qf1_next_target = qf1.apply(qf1_target_params, z_next, next_action).squeeze(-1)
        qf2_next_target = qf2.apply(qf2_target_params, z_next, next_action).squeeze(-1)
        min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target) - alpha * next_log_prob

        next_q_value = reward_scale * batch.reward + config["GAMMA"] * (1.0 - batch.done) * min_qf_next_target
        next_q_value = jax.lax.stop_gradient(next_q_value)

        def critic_loss_fn(enc_params, qf1_params, qf2_params):
            z = shared_encoder.apply(enc_params, batch.obs)
            qf1_a = qf1.apply(qf1_params, z, batch.action).squeeze(-1)
            qf2_a = qf2.apply(qf2_params, z, batch.action).squeeze(-1)
            qf1_loss = ((qf1_a - next_q_value) ** 2).mean()
            qf2_loss = ((qf2_a - next_q_value) ** 2).mean()
            return qf1_loss + qf2_loss, (qf1_loss, qf2_loss, qf1_a.mean(), qf2_a.mean())

        (qf_loss, (qf1_loss, qf2_loss, qf1_values, qf2_values)), \
            (enc_grads, qf1_grads, qf2_grads) = jax.value_and_grad(
                critic_loss_fn, argnums=(0, 1, 2), has_aux=True
            )(encoder_state.params, qf1_state.params, qf2_state.params)

        new_encoder_state = encoder_state.apply_gradients(grads=enc_grads)
        new_qf1_state = qf1_state.apply_gradients(grads=qf1_grads)
        new_qf2_state = qf2_state.apply_gradients(grads=qf2_grads)

        qf1_grad_norm = _g_norm(qf1_grads)
        qf2_grad_norm = _g_norm(qf2_grads)

        return (new_encoder_state, new_qf1_state, new_qf2_state, qf_loss, qf1_loss, qf2_loss,
                qf1_values, qf2_values, next_q_value.mean(),
                qf1_grad_norm, qf2_grad_norm,
                key)

    @jax.jit
    def update_actor_and_alpha(
        encoder_state, actor_state, qf1_state, qf2_state, alpha_state, alpha, batch, key,
    ):
        key, actor_key = jax.random.split(key)

        def actor_loss_fn(actor_params):
            # STOP GRADIENT on encoder: actor optimization must not destroy shared encoder features
            z_actor = jax.lax.stop_gradient(shared_encoder.apply(encoder_state.params, batch.obs))
            mean, log_std = actor.apply(actor_params, z_actor)
            action_env, log_prob, _ = sample_env_action(mean, log_std, actor_key)

            qf1_pi = qf1.apply(qf1_state.params, z_actor, action_env).squeeze(-1)
            qf2_pi = qf2.apply(qf2_state.params, z_actor, action_env).squeeze(-1)
            min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)

            alpha_for_actor = jax.lax.stop_gradient(alpha)
            actor_loss = (alpha_for_actor * log_prob - min_qf_pi).mean()
            return actor_loss

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)

        if config["AUTOTUNE"]:
            key, alpha_key = jax.random.split(key)

            def get_log_prob(actor_params):
                z_actor = jax.lax.stop_gradient(shared_encoder.apply(encoder_state.params, batch.obs))
                mean, log_std = actor.apply(actor_params, z_actor)
                _, log_prob, _ = sample_env_action(mean, log_std, alpha_key)
                return log_prob

            log_prob_alpha = get_log_prob(new_actor_state.params)
            log_prob_alpha = jax.lax.stop_gradient(log_prob_alpha)

            def alpha_loss_fn(alpha_params):
                log_alpha = alpha_params["log_alpha"]
                alpha_value = jnp.exp(log_alpha)
                below = jnp.maximum(log_alpha_min - log_alpha, 0.0)
                above = jnp.maximum(log_alpha - log_alpha_max, 0.0)
                range_penalty = jnp.mean(below ** 2) + jnp.mean(above ** 2)
                alpha_loss = -(alpha_value * (log_prob_alpha + target_entropy)).mean() + 0.1 * range_penalty
                return alpha_loss

            alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(alpha_state.params)
            new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
            new_alpha = jnp.exp(new_alpha_state.params["log_alpha"])
            log_prob_mean = log_prob_alpha.mean()
        else:
            new_alpha_state = alpha_state
            new_alpha = alpha
            alpha_loss = jnp.array(0.0)
            log_prob_mean = jnp.array(0.0)

        actor_grad_norm = _g_norm(actor_grads)

        return (
            encoder_state,  # Encoder untouched by actor step
            new_actor_state,
            new_alpha_state,
            new_alpha,
            actor_loss,
            log_prob_mean,
            alpha_loss,
            actor_grad_norm,
            key,
        )

    @jax.jit
    def target_update(
        encoder_state, qf1_state, qf2_state,
        encoder_target_params, qf1_target_params, qf2_target_params,
        actor_state, actor_target_params, tau,
    ):
        new_encoder_target_params = optax.incremental_update(encoder_state.params, encoder_target_params, tau)
        new_qf1_target_params = optax.incremental_update(qf1_state.params, qf1_target_params, tau)
        new_qf2_target_params = optax.incremental_update(qf2_state.params, qf2_target_params, tau)
        new_actor_target_params = optax.incremental_update(actor_state.params, actor_target_params, tau)
        return new_encoder_target_params, new_qf1_target_params, new_qf2_target_params, new_actor_target_params

    # ---------- Save and eval function ----------
    def save_and_eval(iteration, encoder_state, actor_state, qf1_state, qf2_state):
        if config.get("SAVE_PATH") is not None:
            model_path = f'{config["SAVE_PATH"]}/{run_name}/{config["EXP_NAME"]}_{iteration}_{time.time()}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(
                    flax.serialization.to_bytes(
                        [
                            config,
                            [
                                encoder_state.params,
                                actor_state.params,
                                qf1_state.params,
                                qf2_state.params,
                            ],
                        ]
                    )
                )
            print(f"model saved to {model_path}")
        else:
            model_path = None

        eval_mods = config["EVAL_MODS"] if len(config["EVAL_MODS"]) > 0 else config["TRAIN_MODS"]
        eval_configs = [([], "default")]
        if len(eval_mods) > 0:
            mods_list = list(eval_mods)
            for mod in mods_list:
                mods_config = [mod] if not isinstance(mod, (list, tuple)) else list(mod)
                mod_label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods_config)
                eval_configs.append((mods_config, mod_label))

        metrics = {}
        for mods_config, mod_label in eval_configs:
            print(f"Evaluating on {mod_label} ...")
            eval_model_path = model_path
            tmp_path = None
            try:
                if eval_model_path is None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".cleanrl_model") as tmp:
                        tmp_path = tmp.name
                        tmp.write(
                            flax.serialization.to_bytes(
                                [
                                    config,
                                    [
                                        encoder_state.params,
                                        actor_state.params,
                                        qf1_state.params,
                                        qf2_state.params,
                                    ],
                                ]
                            )
                        )
                    eval_model_path = tmp_path

                episodic_returns, env_states = evaluate(
                    eval_model_path,
                    partial(
                        make_env,
                        mods=mods_config,
                        pixel_based=config["PIXEL_BASED"],
                        native_downscaling=config["NATIVE_DOWNSCALING"],
                        eval=True,
                        tau_cale=config["TAU_CALE"],
                    ),
                    config["ENV_ID"],
                    eval_episodes=10,
                    Model=(CNNEncoder, Actor, SoftQNetwork) if config["PIXEL_BASED"] else (MLPEncoder, Actor, SoftQNetwork),
                    seed=config["SEED"] + 42,
                )
                mean_return = np.mean(jax.device_get(episodic_returns))
                metrics[mod_label] = mean_return
                wandb.log({f"eval/episodic_return_{mod_label}": mean_return}, step=iteration)

                if config.get("CAPTURE_VIDEO", False):
                    frames = render_env_states(
                        env_states,
                        config["ENV_ID"],
                        mods=mods_config,
                        max_frames=config.get("VIDEO_MAX_FRAMES", 1800),
                    )
                    if frames is not None:
                        video = wandb.Video(frames, fps=30, format="mp4")
                        wandb.log({f"eval/video_{mod_label}": video}, step=iteration)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        return metrics

    # ---------- Training loop ----------
    num_envs = config["NUM_ENVS"]
    total_timesteps = config["TOTAL_TIMESTEPS"]
    learning_starts = config["LEARNING_STARTS"]

    key, reset_key = jax.random.split(key)
    obs, env_state = vmap_reset(jax.random.split(reset_key, num_envs))

    buffer_on_cpu = config.get("BUFFER_ON_CPU", True)
    cpu_device = jax.devices("cpu")[0] if buffer_on_cpu else None

    try:
        compute_device = jax.devices("gpu")[0]
    except RuntimeError:
        compute_device = jax.devices()[0]

    dummy_transition = Transition(
        obs=obs,
        action=jnp.zeros((config["NUM_ENVS"], action_dim), dtype=jnp.float32),
        reward=jnp.zeros((config["NUM_ENVS"],), dtype=jnp.float32),
        next_obs=obs,
        done=jnp.zeros((config["NUM_ENVS"],), dtype=jnp.bool_),
    )
    if buffer_on_cpu:
        dummy_transition = jax.tree.map(lambda x: jax.device_put(x, cpu_device), dummy_transition)

    buffer = fbx.make_item_buffer(
        max_length=config["BUFFER_SIZE"],
        min_length=config["BATCH_SIZE"],
        sample_batch_size=config["BATCH_SIZE"],
        add_batches=True,
        add_sequences=False,
    )
    buffer_state = buffer.init(dummy_transition)

    # ---------------------------------------------------------------------------
    # Replay buffer access.
    #
    # Donate the old state on add so XLA can reuse replay storage in place.
    # The old state must not be read after donation; sampling is dispatched
    # before the next add, so this ordering is safe.
    # ---------------------------------------------------------------------------
    max_buffer_length = config["BUFFER_SIZE"]
    min_buffer_length = config["BATCH_SIZE"]

    if buffer_on_cpu:
        @jax.jit(device=cpu_device, donate_argnums=(0,))
        def buffer_add(state, transition):
            return buffer.add(state, transition)

        @jax.jit(device=cpu_device)
        def buffer_sample(state, key):
            return buffer.sample(state, key)
    else:
        buffer_add = jax.jit(buffer.add, donate_argnums=(0,))
        buffer_sample = jax.jit(buffer.sample)

    def can_sample_count(n_added):
        return n_added >= max_buffer_length or (n_added % max_buffer_length) >= min_buffer_length

    buffer_num_added = 0

    # Optional benchmark instrumentation (SAC_PROFILE=1); disabled in normal runs.
    # Profiling synchronizes measured sections, so timings are diagnostic only
    # and may differ from normal asynchronous execution.
    global LAST_PROFILE
    prof = SectionProfiler() if PROFILE_ENABLED else None
    if prof is not None:
        print("[SAC_PROFILE] section timing enabled; runs are serialised per section, "
              "the training trajectory is unchanged.")

    print("Filling replay buffer with random actions...")
    steps_to_fill = int(learning_starts // num_envs) + 1
    fill_batch_size = config.get("FILL_BATCH_SIZE", 256)
    pending = []

    def flush_pending(pending, buffer_state):
        nonlocal buffer_num_added
        if not pending:
            return buffer_state
        batch = Transition(
            obs=jnp.stack([t.obs for t in pending], axis=0),
            action=jnp.stack([t.action for t in pending], axis=0),
            reward=jnp.stack([t.reward for t in pending], axis=0),
            next_obs=jnp.stack([t.next_obs for t in pending], axis=0),
            done=jnp.stack([t.done for t in pending], axis=0),
        )
        batch = jax.tree.map(lambda x: jax.device_put(x, cpu_device), batch)
        buffer_num_added += len(pending) * num_envs
        return buffer_add(buffer_state, batch)

    for _ in range(steps_to_fill):
        if prof:
            t_sec = time.perf_counter()
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(subkey, (num_envs, action_dim), minval=low, maxval=high)
        next_obs, env_state, reward, terminated, next_done, info = vmap_step(env_state, action)
        if prof:
            prof.add("warmup/env_step", t_sec, next_obs)
        obs_dtype = jnp.uint8 if config["PIXEL_BASED"] else jnp.float32
        transition = Transition(
            as_dtype(obs, obs_dtype),
            action,
            as_dtype(reward, jnp.float32),
            as_dtype(next_obs, obs_dtype),
            as_dtype(terminated, jnp.bool_),
        )
        pending.append(transition)
        if len(pending) >= fill_batch_size:
            if prof:
                t_sec = time.perf_counter()
            buffer_state = flush_pending(pending, buffer_state)
            if prof:
                prof.add("warmup/replay_add", t_sec, buffer_state)
            pending = []
        obs = next_obs

    buffer_state = flush_pending(pending, buffer_state)

    print("Starting training...")
    global_step = 0
    start_time = time.time()

    train_add_batch_size = config.get("TRAIN_ADD_BATCH_SIZE", 8)
    train_pending = []

    def flush_train_pending(pending, buffer_state):
        nonlocal buffer_num_added
        if not pending:
            return buffer_state
        batch = Transition(
            obs=jnp.stack([t.obs for t in pending], axis=0),
            action=jnp.stack([t.action for t in pending], axis=0),
            reward=jnp.stack([t.reward for t in pending], axis=0),
            next_obs=jnp.stack([t.next_obs for t in pending], axis=0),
            done=jnp.stack([t.done for t in pending], axis=0),
        )
        batch = jax.tree.map(lambda x: jax.device_put(x, cpu_device), batch)
        buffer_num_added += len(pending) * num_envs
        return buffer_add(buffer_state, batch)

    total_iterations = total_timesteps // (num_envs * config["SCAN_STEPS"]) + 1
    rtpt = RTPT(name_initials=config.get("NAME_INITIALS", "SA"), experiment_name=run_name, max_iterations=total_iterations)
    rtpt.start()

    steps_per_iteration = config["SCAN_STEPS"]
    num_iterations = total_timesteps // (num_envs * steps_per_iteration) + 1

    for iteration in range(num_iterations):
        rtpt.step()

        for local_step in range(steps_per_iteration):
            if prof:
                t_sec = time.perf_counter()
            action, key = sample_action(encoder_state, actor_state, obs, key)
            if prof:
                prof.add("action_sample", t_sec, action)

            if prof:
                t_sec = time.perf_counter()
            next_obs, env_state, reward, terminated, next_done, info = vmap_step(env_state, action)
            if prof:
                prof.add("env_step", t_sec, next_obs)

            if prof:
                t_sec = time.perf_counter()
            obs_dtype = jnp.uint8 if config["PIXEL_BASED"] else jnp.float32
            transition = Transition(
                as_dtype(obs, obs_dtype),
                as_dtype(action, jnp.float32),
                as_dtype(reward, jnp.float32),
                as_dtype(next_obs, obs_dtype),
                as_dtype(terminated, jnp.bool_),
            )
            if prof:
                prof.add("transition_build", t_sec, transition)

            train_pending.append(transition)
            if len(train_pending) >= train_add_batch_size:
                if prof:
                    t_sec = time.perf_counter()
                buffer_state = flush_train_pending(train_pending, buffer_state)
                if prof:
                    prof.add("replay_add", t_sec, buffer_state)
                train_pending = []

            obs = next_obs
            global_step += num_envs

            if can_sample_count(buffer_num_added):
                if prof:
                    t_sec = time.perf_counter()
                key, sample_key = jax.random.split(key)
                batch = buffer_sample(buffer_state, sample_key).experience
                batch = Transition(
                    remove_env_dim(batch.obs),
                    remove_env_dim(batch.action),
                    remove_env_dim(batch.reward),
                    remove_env_dim(batch.next_obs),
                    remove_env_dim(batch.done),
                )
                if prof:
                    prof.add("replay_sample", t_sec, batch)

                if prof:
                    t_sec = time.perf_counter()
                if buffer_on_cpu:
                    batch = jax.device_put(batch, compute_device)
                batch = Transition(
                    batch.obs.astype(jnp.float32),
                    batch.action.astype(jnp.float32),
                    batch.reward.astype(jnp.float32),
                    batch.next_obs.astype(jnp.float32),
                    batch.done.astype(jnp.float32),
                )
                if prof:
                    prof.add("h2d_and_float_cast", t_sec, batch)

                if config["AUTOTUNE"]:
                    current_alpha = jnp.exp(alpha_state.params["log_alpha"]).squeeze()
                else:
                    current_alpha = config["ALPHA"]

                if prof:
                    t_sec = time.perf_counter()
                (encoder_state, qf1_state, qf2_state, qf_loss, qf1_loss, qf2_loss,
                 qf1_values, qf2_values, next_q_values,
                 qf1_grad_norm, qf2_grad_norm,
                 key) = update_qf(
                    encoder_state, qf1_state, qf2_state, actor_state,
                    encoder_target_params, qf1_target_params, qf2_target_params,
                    current_alpha, batch, key,
                )
                if prof:
                    prof.add("critic_update", t_sec,
                             (encoder_state, qf1_state, qf2_state, qf_loss, qf1_values, qf2_values))

                # Single actor update on delay
                if global_step % config["POLICY_FREQUENCY"] == 0:
                    if prof:
                        t_sec = time.perf_counter()
                    (
                        encoder_state,
                        actor_state,
                        alpha_state,
                        current_alpha,
                        actor_loss,
                        log_prob_mean,
                        alpha_loss,
                        actor_grad_norm,
                        key,
                    ) = update_actor_and_alpha(
                        encoder_state, actor_state, qf1_state, qf2_state,
                        alpha_state, current_alpha, batch, key,
                    )
                    if prof:
                        prof.add("actor_update", t_sec,
                                 (encoder_state, actor_state, alpha_state, actor_loss))
                else:
                    actor_loss = jnp.array(0.0)
                    log_prob_mean = jnp.array(0.0)
                    alpha_loss = jnp.array(0.0)
                    actor_grad_norm = jnp.array(0.0)

                if global_step % config["TARGET_NETWORK_FREQUENCY"] == 0:
                    if prof:
                        t_sec = time.perf_counter()
                    encoder_target_params, qf1_target_params, qf2_target_params, actor_target_params = target_update(
                        encoder_state, qf1_state, qf2_state,
                        encoder_target_params, qf1_target_params, qf2_target_params,
                        actor_state, actor_target_params, config["TAU"],
                    )
                    if prof:
                        prof.add("target_update", t_sec,
                                 (encoder_target_params, qf1_target_params, qf2_target_params))

        buffer_state = flush_train_pending(train_pending, buffer_state)
        train_pending = []

        # Logging inside safe sample check context
        if iteration % 1 == 0 and can_sample_count(buffer_num_added):
            avg_return = info["returned_episode_returns"].mean() if "returned_episode_returns" in info else 0.0
            avg_length = info["returned_episode_lengths"].mean() if "returned_episode_lengths" in info else 0.0

            metrics = jax.device_get({
                "charts/avg_episodic_return": avg_return,
                "charts/avg_episodic_length": avg_length,
                "losses/qf1_loss": qf1_loss,
                "losses/qf2_loss": qf2_loss,
                "losses/qf_loss": qf_loss / 2.0,
                "losses/qf1_values": qf1_values,
                "losses/qf2_values": qf2_values,
                "losses/actor_loss": actor_loss,
                "losses/log_prob_mean": log_prob_mean,
                "losses/alpha_loss": alpha_loss,
                "losses/alpha": current_alpha,
                "reward": jnp.mean(batch.reward),
                "charts/SPS": global_step / (time.time() - start_time + 1e-8),
                "charts/global_step": global_step,
                "charts/iteration": iteration,
            })
            wandb.log(metrics, step=global_step)

            print(
                f"[ITER {iteration:3d}] step={global_step:6d} ret={_to_float(avg_return):6.1f} "
                f"len={_to_float(avg_length):5.0f} alpha={_to_float(current_alpha):.4f} "
                f"logp={_to_float(log_prob_mean):6.2f} q1={_to_float(qf1_values):7.2f} "
                f"q2={_to_float(qf2_values):7.2f} qf_loss={_to_float(qf_loss):.4f} "
                f"act_loss={_to_float(actor_loss):6.3f} "
                f"|g_act|={_to_float(actor_grad_norm):6.2f} "
                f"|g_q1|={_to_float(qf1_grad_norm):6.2f} |g_q2|={_to_float(qf2_grad_norm):6.2f}"
            )

        if config.get("EVAL_DURING_TRAIN", False) and iteration > 0 and iteration % config.get("EVAL_EVERY", 50) == 0:
            save_and_eval(iteration, encoder_state, actor_state, qf1_state, qf2_state)

    if prof is not None:
        LAST_PROFILE = prof
        print(
            prof.summary(
                steps=global_step,
                wall_s=time.time() - start_time,
                title=(
                    f"[SAC_PROFILE] TRANSITIONS={global_step} BATCH_SIZE={config['BATCH_SIZE']} "
                    f"TRAIN_ADD_BATCH_SIZE={train_add_batch_size} "
                    f"POLICY_FREQUENCY={config['POLICY_FREQUENCY']} "
                    f"TARGET_NETWORK_FREQUENCY={config['TARGET_NETWORK_FREQUENCY']} "
                    f"BUFFER_SIZE={config['BUFFER_SIZE']} BUFFER_ON_CPU={buffer_on_cpu} "
                    f"PIXEL_BASED={config['PIXEL_BASED']} NUM_ENVS={num_envs} "
                    f"compute_device={compute_device}"
                ),
            )
        )

    print("Evaluating final model ...")
    metrics = save_and_eval(iteration + 1, encoder_state, actor_state, qf1_state, qf2_state)
    wandb.finish()
    print("Training finished.")
    return metrics