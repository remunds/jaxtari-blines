"""Evaluation metrics for C-JEPA world model.

- Masked-slot MSE: loss on held-out validation trajectories.
- Rollout error: multi-step autoregressive prediction MSE (via JIT scan).
"""

from typing import Dict, Optional
from functools import partial

import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents.cjepa.cjepa import CJEPA
from agents.cjepa.data import TrajectoryDataset


def evaluate_masked_mse(
    model: CJEPA,
    params: flax.core.FrozenDict,
    dataset: TrajectoryDataset,
    num_batches: int = 10,
    batch_size: int = 64,
    rng: jax.Array = None,
) -> Dict[str, float]:
    """Compute masked-slot MSE on validation data."""
    if rng is None:
        rng = jax.random.PRNGKey(0)

    losses = []
    for _ in range(num_batches):
        rng, batch_rng = jax.random.split(rng)
        batch = dataset.get_batch(batch_size, batch_rng)
        full_obs = jnp.concatenate([batch["obs"], batch["target_obs"]], axis=1)
        full_actions = jnp.concatenate([batch["actions"], batch["target_actions"]], axis=1)
        loss, _ = model.apply(params, full_obs, full_actions, rng=rng, training=False,
                                rngs={'dropout': rng})
        losses.append(float(loss))

    return {"val_loss": float(np.mean(losses))}


# ── JIT-compiled scan for multi-step rollout ──────────────────────────

@partial(jax.jit, static_argnums=(0,))
def _rollout_mse(
    model: CJEPA,
    params: flax.core.FrozenDict,
    context_obs: jnp.ndarray,
    context_actions: jnp.ndarray,
    future_obs: jnp.ndarray,
    future_actions: jnp.ndarray,
    rng: jax.Array,
) -> jnp.ndarray:
    """True autoregressive rollout MSE.

    Calls predict_future once (which does the full autoregressive loop
    internally in JIT-compiled code), then compares each predicted slot
    against the ground truth slot.

    Args:
        model: CJEPA model.
        params: Model parameters.
        context_obs: [T_hist, obs_dim] initial context.
        context_actions: [T_hist] initial context actions.
        future_obs: [num_steps, obs_dim] GT observations for comparison.
        future_actions: [num_steps] actions for each step.
        rng: PRNG key.

    Returns:
        mses: [num_steps] MSE at each rollout step.
    """
    num_steps = future_obs.shape[0]
    total_actions = jnp.concatenate([context_actions, future_actions], axis=0)

    # True autoregressive prediction: predict_future feeds its own
    # predictions back as context for subsequent steps
    pred = model.apply(
        params,
        context_obs[None, :, :],         # [1, T_hist, obs_dim]
        total_actions[None, :],           # [1, T_hist + num_steps - 1]
        method=model.predict_future,
        num_future=num_steps,
        rngs={'dropout': rng},
    )
    pred_slots = pred[0, -num_steps:, :, :]  # [num_steps, S, D]

    # Encode ground truth observations to slots
    gt_slots = model.apply(
        params, future_obs[None, :, :],
        method=model.encode_slots,
    )[0]  # [num_steps, S, D]

    # MSE per step
    mses = jnp.mean((pred_slots - gt_slots) ** 2, axis=(1, 2))  # [num_steps]
    return mses


# ── Public evaluation function ────────────────────────────────────────

def evaluate_rollout(
    model: CJEPA,
    params: flax.core.FrozenDict,
    dataset: TrajectoryDataset,
    rollout_steps: int = 30,
    history_frames: int = 3,
    pred_frames: int = 1,
    num_samples: int = 32,
    rng: jax.Array = None,
) -> Dict[str, float]:
    """Compute multi-step rollout error using JIT-compiled scan.

    Samples trajectories from the validation dataset, runs autoregressive
    prediction for rollout_steps, and reports MSE at each step.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    if dataset.num_windows < num_samples:
        num_samples = dataset.num_windows

    max_rollout = min(rollout_steps, int(dataset.traj_length) - history_frames)

    # Sample trajectory indices
    rng, idx_rng = jax.random.split(rng)
    sample_indices = jax.random.randint(idx_rng, (num_samples,), 0, dataset.num_trajectories)
    sample_indices = np.array(jax.device_get(sample_indices))

    mses_per_step = [[] for _ in range(max_rollout)]

    for traj_idx in sample_indices:
        traj_obs = dataset.obs[int(traj_idx)]
        traj_actions = dataset.actions[int(traj_idx)]

        # Find a valid starting position (after dones)
        done_positions = np.where(np.array(dataset.dones[int(traj_idx)]))[0]
        start = 0
        for d in done_positions:
            if d >= history_frames + max_rollout:
                start = 0
                break
            start = max(start, d + 1)

        if start + history_frames + max_rollout > len(traj_obs):
            continue

        # Extract context and future
        context_o = jnp.array(traj_obs[start:start + history_frames])
        context_a = jnp.array(traj_actions[start:start + history_frames])
        future_o = jnp.array(traj_obs[start + history_frames:start + history_frames + max_rollout])
        future_a = jnp.array(traj_actions[start + history_frames:start + history_frames + max_rollout])

        # Single model.apply call — true autoregressive rollout
        rng, step_rng = jax.random.split(rng)
        mses = _rollout_mse(
            model, params,
            context_o, context_a,
            future_o, future_a, step_rng,
        )
        mses_np = np.array(mses)
        for step_idx in range(max_rollout):
            mses_per_step[step_idx].append(float(mses_np[step_idx]))

    results = {}
    for step_idx, mses in enumerate(mses_per_step):
        if len(mses) > 0:
            results[f"rollout_mse_step_{step_idx + 1}"] = float(np.mean(mses))

    if results:
        results["rollout_mse_avg"] = float(np.mean(list(results.values())))

    return results
