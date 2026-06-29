#!/usr/bin/env python3
"""Visualize C-JEPA world model rollouts on Pong using JAXtari's renderer.

Produces:
- ``{run_name}_trajectories.png`` — predicted vs actual x,y positions over time
- ``{run_name}_trajectories_2d.png`` — 2D screen-space trajectories
- ``{run_name}_rollout.gif`` — side-by-side rendered frames (ground truth | prediction)

Usage::

    uv run python agents/cjepa/viz_rollout.py \\
        --checkpoint ./models/pong_cjepa_pong_oc_0/step_8000.ckpt \\
        --run-name pong_viz_step8000 \\
        --num-frames 300
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
from functools import partial

import flax
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on sys.path for imports when run as script
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from agents.cjepa.cjepa import CJEPA, create_train_state
from agents.cjepa.data import get_env_info, make_oc_env_single_frame


# ── Position extraction from OC observations ──────────────────────────

def extract_positions(oc_obs: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Extract (x, y) positions for each object from an OC observation.

    OC layout [T, 24]: player(8), enemy(8), ball(8).
    Each object: x, y, width, height, active, visual_id, state, orientation.
    """
    was_1d = oc_obs.ndim == 1
    if oc_obs.ndim == 1:
        oc_obs = oc_obs[None, :]
    elif oc_obs.ndim == 2:
        oc_obs = oc_obs[None, :, :]

    player_x = np.array(oc_obs[..., 0])
    player_y = np.array(oc_obs[..., 1])
    enemy_x = np.array(oc_obs[..., 8])
    enemy_y = np.array(oc_obs[..., 9])
    ball_x = np.array(oc_obs[..., 16])
    ball_y = np.array(oc_obs[..., 17])

    player_x, player_y = player_x[0], player_y[0]
    enemy_x, enemy_y = enemy_x[0], enemy_y[0]
    ball_x, ball_y = ball_x[0], ball_y[0]

    if was_1d:
        return {
            "player": (float(player_x), float(player_y)),
            "enemy": (float(enemy_x), float(enemy_y)),
            "ball": (float(ball_x), float(ball_y)),
        }
    return {
        "player": (player_x, player_y),
        "enemy": (enemy_x, enemy_y),
        "ball": (ball_x, ball_y),
    }


# ── Trajectory collection ─────────────────────────────────────────────

def collect_trajectory(
    env_id: str = "pong",
    num_frames: int = 310,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Collect a trajectory from the OC env, tracking raw states for rendering.

    Returns:
        gt_obs: [num_frames, 24] OC observations (object attrs only).
        gt_actions: [num_frames] discrete actions.
        gt_frames: list of [210, 160, 3] uint8 rendered frames.
    """
    import jaxatari
    oc_env = make_oc_env_single_frame(env_id)
    raw_env = jaxatari.make(env_id)

    rng = jax.random.PRNGKey(seed)
    obs_list = []
    action_list = []
    frame_list = []

    # Reset
    rng, reset_rng = jax.random.split(rng)
    obs, log_state = oc_env.reset(reset_rng)
    pong_state = log_state.atari_state.atari_state.env_state

    # Collect frames using jax.lax.scan for performance
    # Also collect raw PongState for re-rendering with predicted positions
    def step_fn(carry, _):
        obs, log_state, rng = carry
        rng, a_rng = jax.random.split(rng)
        action = jax.random.randint(a_rng, (), 0, raw_env.action_space().n)
        next_obs, next_log_state, reward, term, trunc, info = oc_env.step(log_state, action)
        pong_state = next_log_state.atari_state.atari_state.env_state
        frame = raw_env.render(pong_state)
        return (next_obs, next_log_state, rng), (obs, action, frame, pong_state)

    (_, _, _), (obs_seq, actions_seq, frames_seq, states_seq) = jax.lax.scan(
        step_fn, (obs, log_state, rng), None, length=num_frames
    )

    # Convert to numpy
    gt_obs = np.array(obs_seq)        # [num_frames, obs_dim]
    if gt_obs.ndim == 3 and gt_obs.shape[1] == 1:
        gt_obs = gt_obs[:, 0, :]
    gt_obs = gt_obs[..., :24]          # Keep only object attrs
    gt_actions = np.array(actions_seq)  # [num_frames]
    gt_frames = [np.array(f) for f in frames_seq]  # list of [210, 160, 3]

    # Unbatch the PongState pytree into a list of individual states
    gt_states = []
    for i in range(num_frames):
        s = jax.tree.map(lambda x: x[i], states_seq)
        gt_states.append(s)

    return gt_obs, gt_actions, gt_frames, gt_states


# ── Model rollout ─────────────────────────────────────────────────────

def rollout_model(
    model: CJEPA,
    params: flax.core.FrozenDict,
    obs_sequence: jnp.ndarray,
    action_sequence: jnp.ndarray,
    history_frames: int = 3,
    num_predict: int = 300,
    rng: jax.Array = None,
) -> jnp.ndarray:
    """True autoregressive rollout.

    Calls predict_future with num_future=N, which:
    1. Encodes initial T_hist frames to slots
    2. Predicts the next frame via the transformer
    3. Feeds predicted slots back as context for the next step
    4. Injects actions at each step

    This is a SINGLE model.apply call — the autoregressive loop
    runs inside the JIT-compiled predict_future method, not in Python.

    Returns:
        pred_slots: [num_predict, S, D] — predicted slot vectors.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    context_obs = obs_sequence[:history_frames]          # [T_hist, obs_dim]
    # predict_future needs T_hist + num_predict - 1 actions
    total_actions = history_frames + num_predict - 1
    rollout_actions = action_sequence[:total_actions]    # [T_hist + num_predict - 1]

    result = model.apply(
        params,
        context_obs[None, :, :],          # [1, T_hist, obs_dim]
        rollout_actions[None, :],          # [1, T_hist + num_predict - 1]
        method=model.predict_future,
        num_future=num_predict,
        rngs={'dropout': rng},
    )
    # result: [1, T_hist + num_predict, S, D]
    return result[0, history_frames:, :, :]  # [num_predict, S, D]


# ── Trajectory plots ──────────────────────────────────────────────────

def plot_trajectories(
    gt_oc: np.ndarray,
    pred_slots: np.ndarray,
    save_path: str,
    max_frames: int = 300,
    decoded_oc: Optional[np.ndarray] = None,
):
    """Plot predicted vs actual object positions over time (X/Y and 2D).

    If decoded_oc is provided (from SlotDecoder), use its x,y dims directly
    (they're in screen coordinates). Otherwise fall back to normalizing
    raw slot dims 0-1 as a proxy.
    """
    T = min(gt_oc.shape[0], max_frames)
    gt_oc = gt_oc[:T]
    pred_slots = pred_slots[:T]
    if decoded_oc is not None:
        decoded_oc = decoded_oc[:T]
    frames = np.arange(T)

    gt_pos = extract_positions(gt_oc)

    # Predicted positions: use decoder output if available, else raw slot proxy
    pred = {}
    for i, obj in enumerate(["player", "enemy", "ball"]):
        if decoded_oc is not None:
            # Direct screen coordinates from decoder (no normalization needed)
            pred[obj] = (np.array(decoded_oc[:, i, 0]), np.array(decoded_oc[:, i, 1]))
        else:
            # Fallback: normalize raw slot dims 0-1 as position proxy
            px = np.array(pred_slots[:, i, 0])
            py = np.array(pred_slots[:, i, 1])
            def norm(arr, lo, hi):
                a_min, a_max = arr.min(), arr.max()
                if a_max - a_min < 1e-6:
                    return np.full_like(arr, (lo + hi) / 2)
                return (arr - a_min) / (a_max - a_min) * (hi - lo) + lo
            pred[obj] = (norm(px, 0, 160), norm(py, 0, 210))

    colors = {"player": "#2196F3", "enemy": "#F44336", "ball": "#4CAF50"}

    # 1. X/Y over time (3×2 grid)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for i, (obj, label) in enumerate([("player", "Player"), ("enemy", "Enemy"), ("ball", "Ball")]):
        gt_x, gt_y = gt_pos[obj]
        pd_x, pd_y = pred[obj]

        axes[i, 0].plot(frames, gt_x, color=colors[obj], alpha=0.7, label=f"{label} GT", linewidth=1)
        axes[i, 0].plot(frames, pd_x, color=colors[obj], linestyle="--", alpha=0.7, label=f"{label} Pred", linewidth=1)
        axes[i, 0].set_ylabel("X"); axes[i, 0].legend(fontsize=8); axes[i, 0].set_title(f"{label} X", fontsize=10)

        axes[i, 1].plot(frames, gt_y, color=colors[obj], alpha=0.7, label=f"{label} GT", linewidth=1)
        axes[i, 1].plot(frames, pd_y, color=colors[obj], linestyle="--", alpha=0.7, label=f"{label} Pred", linewidth=1)
        axes[i, 1].set_ylabel("Y"); axes[i, 1].legend(fontsize=8); axes[i, 1].set_title(f"{label} Y", fontsize=10)

    axes[-1, 0].set_xlabel("Frame")
    axes[-1, 1].set_xlabel("Frame")
    fig.suptitle("C-JEPA: Predicted vs Ground Truth Object Positions", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Trajectory plot: {save_path}")

    # 2. 2D trajectories
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    for i, (obj, label) in enumerate([("player", "Player"), ("enemy", "Enemy"), ("ball", "Ball")]):
        gt_x, gt_y = gt_pos[obj]
        pd_x, pd_y = pred[obj]
        ax = axes2[i]
        ax.plot(gt_x, gt_y, color=colors[obj], alpha=0.7, label=f"{label} GT", linewidth=1)
        ax.plot(pd_x, pd_y, color=colors[obj], linestyle="--", alpha=0.5, label=f"{label} Pred", linewidth=1)
        ax.scatter(gt_x[0], gt_y[0], color=colors[obj], marker="o", s=50, zorder=5)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_title(f"{label} 2D Trajectory", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 160); ax.set_ylim(210, 0); ax.set_aspect("equal")

    fig2.suptitle("C-JEPA: 2D Trajectory (Predicted vs Ground Truth)", fontsize=13)
    plt.tight_layout()
    p2d = save_path.replace(".png", "_2d.png")
    plt.savefig(p2d, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  2D trajectory: {p2d}")


# ── Rollout video ─────────────────────────────────────────────────────

def render_rollout_gif(
    gt_states: list,
    gt_frames: List[np.ndarray],
    pred_slots: np.ndarray,
    save_path: str,
    raw_env=None,
    max_frames: int = 300,
    decoded_oc: Optional[np.ndarray] = None,
):
    """Create a split-screen GIF: ground truth | predicted rendering.

    Left: ground truth rendered frame.
    Right: renders the same frame but with object positions replaced
           by the model's predicted positions.

    If decoded_oc is provided (from SlotDecoder), uses its x,y dims
    directly (screen coords). Otherwise falls back to normalizing
    raw slot dims 0-1.

    Args:
        gt_states: list of PongState objects (one per frame).
        gt_frames: list of [210, 160, 3] uint8 ground truth frames.
        pred_slots: [num_predict, S, D] predicted slot vectors.
        save_path: Output GIF path.
        raw_env: A JAXtari env for rendering. Created if None.
        max_frames: Max frames to include.
        decoded_oc: [T, S, 8] decoded OC attributes (x=0, y=1 in screen coords).
    """
    import jaxatari
    if raw_env is None:
        raw_env = jaxatari.make('pong')

    T = min(len(gt_frames), pred_slots.shape[0], max_frames)

    if decoded_oc is not None:
        # Use decoded positions directly (screen coords)
        pred_x = {
            "player": np.array(decoded_oc[:T, 0, 0]),
            "enemy":  np.array(decoded_oc[:T, 1, 0]),
            "ball":   np.array(decoded_oc[:T, 2, 0]),
        }
        pred_y = {
            "player": np.array(decoded_oc[:T, 0, 1]),
            "enemy":  np.array(decoded_oc[:T, 1, 1]),
            "ball":   np.array(decoded_oc[:T, 2, 1]),
        }
    else:
        # Fallback: normalize raw slot dims 0-1
        def norm(arr, lo, hi):
            a_min, a_max = arr.min(), arr.max()
            if a_max - a_min < 1e-6:
                return np.full_like(arr, (lo + hi) / 2)
            return (arr - a_min) / (a_max - a_min) * (hi - lo) + lo
        pred_x = {
            "player": norm(np.array(pred_slots[:T, 0, 0]), 0, 160),
            "enemy":  norm(np.array(pred_slots[:T, 1, 0]), 0, 160),
            "ball":   norm(np.array(pred_slots[:T, 2, 0]), 0, 160),
        }
        pred_y = {
            "player": norm(np.array(pred_slots[:T, 0, 1]), 0, 210),
            "enemy":  norm(np.array(pred_slots[:T, 1, 1]), 0, 210),
            "ball":   norm(np.array(pred_slots[:T, 2, 1]), 0, 210),
        }

    side_frames = []
    skip = max(1, T // max_frames)

    for t in range(0, T, skip):
        # Left: ground truth
        left = gt_frames[t]

        # Right: render with predicted positions
        state = gt_states[t]
        pred_state = state.replace(
            player_y=np.float32(pred_y["player"][t]),
            enemy_y=np.int32(np.round(pred_y["enemy"][t])),
            ball_x=np.int32(np.round(pred_x["ball"][t])),
            ball_y=np.int32(np.round(pred_y["ball"][t])),
        )
        right = np.array(raw_env.render(pred_state))

        # Side-by-side
        combined = np.concatenate([left, right], axis=1)
        side_frames.append(combined)

    if not side_frames:
        print("  No frames to render.")
        return

    from PIL import Image
    pil_frames = [Image.fromarray(f) for f in side_frames]
    duration_ms = max(50, int(1000 / 60 * skip))
    pil_frames[0].save(
        save_path, save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=0, optimize=False,
    )
    print(f"  Rollout GIF: {save_path} ({len(side_frames)} frames @ {duration_ms}ms)")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="C-JEPA Rollout Visualization")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--run-name", type=str, default="cjepa_rollout",
                        help="Run name for output files")
    parser.add_argument("--num-frames", type=int, default=300,
                        help="Number of frames to rollout (~5s at 60fps)")
    parser.add_argument("--start-frame", type=int, default=50,
                        help="Skip initial frames; use previous 3 as context")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Setup ──
    print("Setting up model...")
    rng = jax.random.PRNGKey(args.seed)
    env_id = "pong"
    env_info = get_env_info(env_id)
    num_slots, obj_attr_dim, num_actions, obs_dim = (
        env_info["num_slots"], env_info["obj_attr_dim"],
        env_info["num_actions"], env_info["obs_dim"],
    )
    print(f"  Slots={num_slots}  ObjAttr={obj_attr_dim}  Actions={num_actions}  ObsDim={obs_dim}")

    model = CJEPA(
        num_slots=num_slots, obj_attr_dim=obj_attr_dim, num_actions=num_actions,
        slot_dim=64, history_frames=3, pred_frames=1, num_masked_slots=1,
        action_emb_dim=16, transformer_depth=6, transformer_heads=8,
        transformer_dim_head=64, transformer_mlp_dim=2048, dropout=0.1,
    )
    dummy_obs = jnp.zeros((1, 4, obs_dim))
    dummy_actions = jnp.zeros((1, 4), dtype=jnp.int32)
    train_state, init_params = create_train_state(model, rng, dummy_obs, dummy_actions)

    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        checkpoint_bytes = f.read()
    try:
        # Try full load (exact match)
        params = flax.serialization.from_bytes(init_params, checkpoint_bytes)
    except ValueError:
        # Partial load: checkpoint may be missing new keys (e.g. decoder)
        print("  Partial load — some params initialized fresh (e.g. decoder)")
        def _partial_load(target, state):
            if isinstance(target, dict):
                result = {}
                for k, v in target.items():
                    if k in state:
                        result[k] = _partial_load(v, state[k])
                    else:
                        result[k] = v  # keep init value for new keys
                return result
            else:
                return state  # leaf: use checkpoint value
        ckpt = flax.serialization.msgpack_restore(checkpoint_bytes)
        merged = _partial_load(flax.core.unfreeze(init_params), ckpt)
        params = flax.core.freeze(merged)

    # ── Collect trajectory (OC obs + rendered frames + raw states) ──
    n_collect = args.num_frames + max(args.start_frame, 10)
    print(f"Collecting {n_collect} frames...")
    gt_obs, gt_actions, gt_frames, gt_states = collect_trajectory(
        env_id=env_id, num_frames=n_collect, seed=args.seed + 1,
    )
    print(f"  OC obs: {gt_obs.shape}  Frames: {len(gt_frames)}")

    # ── Run model rollout ──
    start_frame = args.start_frame
    n_pred = min(args.num_frames, len(gt_frames) - start_frame)
    if start_frame < 3:
        print(f"  Warning: start_frame={start_frame} < 3, need at least 3 history frames")
    print(f"Running autoregressive rollout for {n_pred} steps from frame {start_frame}...")
    # Slice trajectory so rollout_model gets context [start_frame-3..] + prediction target
    traj_from_start = jnp.array(gt_obs[start_frame - 3:])
    acts_from_start = jnp.array(gt_actions[start_frame - 3:])
    pred_slots = rollout_model(
        model, params, traj_from_start, acts_from_start,
        history_frames=3, num_predict=n_pred, rng=rng,
    )
    print(f"  Predicted slots: {pred_slots.shape}")

    # Decode predicted slots back to OC attributes
    decoded_oc = np.array(model.apply(
        params, pred_slots[None, :, :, :],
        method=model.decode_slots,
    )[0])  # [T, S, 8] — dims 0=x, 1=y in screen coords

    # Trim GT to match rollout (start from start_frame)
    gt_obs_viz = gt_obs[start_frame:start_frame + n_pred]
    gt_frames_viz = gt_frames[start_frame:start_frame + n_pred]
    gt_states_viz = gt_states[start_frame:start_frame + n_pred]
    pred_slots_viz = np.array(pred_slots[:n_pred])

    # ── Generate outputs ──
    print(f"\nGenerating visualizations ({n_pred} frames)...")

    plot_trajectories(gt_obs_viz, pred_slots_viz,
                      os.path.join(args.output_dir, f"{args.run_name}_trajectories.png"),
                      max_frames=n_pred, decoded_oc=decoded_oc)

    import jaxatari
    raw_env = jaxatari.make('pong')
    render_rollout_gif(gt_states_viz, gt_frames_viz, pred_slots_viz,
                       os.path.join(args.output_dir, f"{args.run_name}_rollout.gif"),
                       raw_env=raw_env, max_frames=n_pred, decoded_oc=decoded_oc)

    print(f"\nDone → {args.output_dir}/")


# ── Programmatic entry point for training integration ────────────────

def generate_eval_visualizations(
    model: CJEPA,
    params: flax.core.FrozenDict,
    rng: jax.Array,
    step: int,
    output_dir: str = "./eval_viz",
    num_frames: int = 100,
    start_frame: int = 50,
    env_id: str = "pong",
    seed: int = 0,
) -> Dict[str, str]:
    """Generate rollout visualizations for wandb logging during training.

    Collects a trajectory, runs the model autoregressively,
    and produces trajectory plots + a side-by-side rollout GIF.

    Args:
        model: CJEPA model instance.
        params: Trained parameters.
        rng: PRNG key.
        step: Current training step (used in filenames).
        output_dir: Where to save temp viz files.
        num_frames: Frames to collect and predict.
        start_frame: Skip initial frames; use previous 3 as context.
        env_id: Atari env ID.
        seed: Random seed for trajectory collection.

    Returns:
        Dict with keys 'trajectory_plot' and 'rollout_gif' mapping to file paths.
    """
    import os, jaxatari
    os.makedirs(output_dir, exist_ok=True)

    # Collect trajectory (OC obs + raw states for rendering)
    gt_obs, gt_actions, gt_frames, gt_states = collect_trajectory(
        env_id=env_id, num_frames=num_frames + start_frame, seed=seed,
    )

    # Run model rollout from start_frame onwards
    n_pred = min(num_frames, len(gt_frames) - start_frame)
    traj_from_start = jnp.array(gt_obs[start_frame - 3:])
    acts_from_start = jnp.array(gt_actions[start_frame - 3:])
    pred_slots = rollout_model(
        model, params, traj_from_start, acts_from_start,
        history_frames=3, num_predict=n_pred, rng=rng,
    )

    # Decode predicted slots back to OC attributes
    decoded_oc = np.array(model.apply(
        params, pred_slots[None, :, :, :],
        method=model.decode_slots,
    )[0])  # [T, S, 8] — dims 0=x, 1=y in screen coords

    gt_obs_viz = gt_obs[start_frame:start_frame + n_pred]
    gt_frames_viz = gt_frames[start_frame:start_frame + n_pred]
    gt_states_viz = gt_states[start_frame:start_frame + n_pred]
    pred_slots_viz = np.array(pred_slots[:n_pred])

    # Generate plots (pass decoded positions — they're already in screen coords)
    plot_path = os.path.join(output_dir, f"eval_step{step}_trajectories.png")
    plot_trajectories(gt_obs_viz, pred_slots_viz, plot_path, max_frames=n_pred,
                      decoded_oc=decoded_oc)

    # Generate GIF — render predicted positions as actual game frames
    import jaxatari
    raw_env = jaxatari.make('pong')
    gif_path = os.path.join(output_dir, f"eval_step{step}_rollout.gif")
    render_rollout_gif(gt_states_viz, gt_frames_viz, pred_slots_viz, gif_path,
                       raw_env=raw_env, max_frames=n_pred, decoded_oc=decoded_oc)

    return {"trajectory_plot": plot_path, "rollout_gif": gif_path}


if __name__ == "__main__":
    main()
