"""C-JEPA training loop.

JEPA-style training with encoder/predictor updates, wandb logging,
and periodic evaluation.
"""

import os
import time
from functools import partial
from typing import Dict, Tuple, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from agents.cjepa.cjepa import CJEPA, create_train_state
from agents.cjepa.data import collect_dataset, get_env_info, TrajectoryDataset
from agents.cjepa.eval import evaluate_rollout
from agents.cjepa.viz_rollout import generate_eval_visualizations


@partial(jax.jit, static_argnums=(0,))
def train_step(
    model: CJEPA,
    train_state: flax.training.train_state.TrainState,
    batch_obs: jnp.ndarray,
    batch_actions: jnp.ndarray,
    rng: jax.Array,
    dropout_rng: jax.Array,
) -> Tuple[flax.training.train_state.TrainState, jnp.ndarray, Dict]:
    """Single training step: compute loss, gradients, update parameters.

    Args:
        model: CJEPA model.
        train_state: Flax TrainState with params and optimizer.
        batch_obs: [B, T_total, obs_dim]
        batch_actions: [B, T_total]
        rng: PRNG key for masking.
        dropout_rng: PRNG key for transformer dropout.

    Returns:
        train_state: Updated TrainState.
        loss: Training loss scalar.
        info: Debug info dict.
    """
    def loss_fn(params):
        loss, info = model.apply(
            params, batch_obs, batch_actions, rng=rng, training=True,
            rngs={'dropout': dropout_rng},
        )
        return loss, info

    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss, info


@partial(jax.jit, static_argnums=(0,))
def train_scan(
    model: CJEPA,
    train_state: flax.training.train_state.TrainState,
    all_obs: jnp.ndarray,
    all_actions: jnp.ndarray,
    rng: jax.Array,
    dropout_rng: jax.Array,
) -> Tuple[flax.training.train_state.TrainState, jnp.ndarray, Dict]:
    """Run multiple train steps in a single JIT-compiled scan.

    Each step consumes one slice from all_obs/all_actions along axis 0.
    This eliminates Python-to-GPU roundtrips between steps.

    Args:
        model: CJEPA model.
        train_state: Flax TrainState.
        all_obs: [scan_steps, B, T_total, obs_dim]
        all_actions: [scan_steps, B, T_total]
        rng: PRNG key for masking (split per step).
        dropout_rng: PRNG key for dropout (split per step).

    Returns:
        train_state: Updated TrainState after all steps.
        losses: [scan_steps] array of losses.
        infos: Dict with [scan_steps] arrays for each scalar info field.
    """
    def step_fn(carry, xs):
        ts, rng, d_rng = carry
        obs, actions = xs

        rng, step_rng = jax.random.split(rng)
        d_rng, step_d_rng = jax.random.split(d_rng)

        new_ts, loss, info = train_step(model, ts, obs, actions, step_rng, step_d_rng)
        return (new_ts, rng, d_rng), (loss, info)

    carry = (train_state, rng, dropout_rng)
    (final_ts, _, _), (losses, infos) = jax.lax.scan(step_fn, carry, (all_obs, all_actions))

    return final_ts, losses, infos


def single_run(config: dict):
    """Run C-JEPA training.

    Args:
        config: Configuration dictionary with all hyperparameters.
    """
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    # Setup run name and wandb
    run_name = f'{config["ENV_ID"]}_{config["EXP_NAME"]}_oc_{config["SEED"]}'
    wandb.init(
        project=config["PROJECT"],
        entity=config["ENTITY"],
        config=config,
        name=run_name,
        save_code=True,
        mode=config.get("WANDB_MODE", "online"),
    )

    # Seeding
    rng = jax.random.PRNGKey(config["SEED"])

    # Get environment info
    env_id = config["ENV_ID"]
    frame_stack_size = config.get("FRAME_STACK_SIZE", 1)
    env_info = get_env_info(env_id, frame_stack_size=frame_stack_size)
    num_slots = env_info["num_slots"]
    obj_attr_dim = env_info["obj_attr_dim"]
    num_actions = env_info["num_actions"]
    obs_dim = env_info["obs_dim"]
    stacked_obj_attr_dim = env_info["stacked_obj_attr_dim"]
    print(f"Environment: {env_id}")
    print(f"  num_slots={num_slots}, obj_attr_dim={obj_attr_dim}, "
          f"num_actions={num_actions}, obs_dim={obs_dim}, "
          f"frame_stack_size={frame_stack_size}, "
          f"stacked_obj_attr_dim={stacked_obj_attr_dim}")

    # Hyperparameters
    slot_dim = config.get("SLOT_DIM", 128)
    history_frames = config.get("HISTORY_FRAMES", 3)
    pred_frames = config.get("PRED_FRAMES", 1)
    max_masked_slots = config.get("MAX_MASKED_SLOTS", 2)
    action_emb_dim = config.get("ACTION_EMB_DIM", 16)
    transformer_depth = config.get("TRANSFORMER_DEPTH", 6)
    transformer_heads = config.get("TRANSFORMER_HEADS", 8)
    transformer_dim_head = config.get("TRANSFORMER_DIM_HEAD", 64)
    transformer_mlp_dim = config.get("TRANSFORMER_MLP_DIM", 2048)
    dropout = config.get("TRANSFORMER_DROPOUT", 0.1)

    batch_size = config.get("BATCH_SIZE", 64)
    learning_rate = config.get("LEARNING_RATE", 5e-4)
    max_grad_norm = config.get("MAX_GRAD_NORM", 1.0)
    num_train_steps = int(config.get("NUM_TRAIN_STEPS", 50000))
    eval_every = config.get("EVAL_EVERY", 500)
    save_path = config.get("SAVE_PATH", "./models")
    rollout_steps = config.get("ROLLOUT_STEPS", 30)
    eval_viz = config.get("EVAL_VIZ", False)
    viz_frames = config.get("VIZ_FRAMES", 100)
    viz_start_frame = config.get("VIZ_START_FRAME", 50)

    # Data collection config
    num_train_trajs = int(config.get("NUM_TRAIN_TRAJECTORIES", 1000))
    num_val_trajs = int(config.get("NUM_VAL_TRAJECTORIES", 100))
    traj_length = int(config.get("TRAJECTORY_LENGTH", 50))
    num_envs = config.get("NUM_ENVS", 64)
    frameskip = config.get("FRAMESKIP", 4)

    # Create model
    model = CJEPA(
        num_slots=num_slots,
        obj_attr_dim=obj_attr_dim,
        num_actions=num_actions,
        slot_dim=slot_dim,
        history_frames=history_frames,
        pred_frames=pred_frames,
        max_masked_slots=max_masked_slots,
        action_emb_dim=action_emb_dim,
        transformer_depth=transformer_depth,
        transformer_heads=transformer_heads,
        transformer_dim_head=transformer_dim_head,
        transformer_mlp_dim=transformer_mlp_dim,
        dropout=dropout,
        frame_stack_size=frame_stack_size,
    )

    T_total = history_frames + pred_frames

    # Initialize with dummy data
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, T_total, obs_dim))
    dummy_actions = jnp.zeros((1, T_total), dtype=jnp.int32)
    train_state, model_params = create_train_state(
        model, rng, dummy_obs, dummy_actions,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
    )
    print(f"Model initialized with {sum(p.size for p in jax.tree.leaves(model_params))} parameters")

    # Collect training data
    print(f"Collecting {num_train_trajs} training trajectories...")
    rng, data_rng = jax.random.split(rng)
    train_dataset = collect_dataset(
        env_id=env_id,
        num_trajectories=num_train_trajs,
        traj_length=traj_length,
        num_envs=num_envs,
        seed=int(jax.random.randint(data_rng, (), 0, 2**31 - 1)),
        frame_skip=frameskip,
        frame_stack_size=frame_stack_size,
    )
    print(f"  Training windows: {train_dataset.num_windows}")

    # Collect validation data
    print(f"Collecting {num_val_trajs} validation trajectories...")
    rng, val_data_rng = jax.random.split(rng)
    val_dataset = collect_dataset(
        env_id=env_id,
        num_trajectories=num_val_trajs,
        traj_length=traj_length,
        num_envs=num_envs,
        seed=int(jax.random.randint(val_data_rng, (), 0, 2**31 - 1)),
        frame_skip=frameskip,
        frame_stack_size=frame_stack_size,
    )
    print(f"  Validation windows: {val_dataset.num_windows}")

    # Training loop
    print(f"\nStarting training for {num_train_steps} steps...")
    start_time = time.time()
    compile_time = None

    # Number of steps per scan call (amortizes Python overhead)
    scan_steps = config.get("SCAN_STEPS", 8)
    print(f"  Using scan over {scan_steps} steps per outer iteration")

    for outer_step in range(1, num_train_steps + 1, scan_steps):
        steps_remaining = min(scan_steps, num_train_steps - outer_step + 1)
        outer_start = time.time()

        # Pre-load batches for this scan window
        obs_batches = []
        action_batches = []
        for _ in range(steps_remaining):
            rng, batch_rng = jax.random.split(rng)
            batch = train_dataset.get_batch(batch_size, batch_rng)
            full_obs = jnp.concatenate([batch["obs"], batch["target_obs"]], axis=1)
            full_actions = jnp.concatenate([batch["actions"], batch["target_actions"]], axis=1)
            obs_batches.append(full_obs)
            action_batches.append(full_actions)

        all_obs = jnp.stack(obs_batches, axis=0)       # [S, B, T_total, obs_dim]
        all_actions = jnp.stack(action_batches, axis=0)  # [S, B, T_total]

        # Run all steps in a single JIT-compiled scan
        rng, scan_rng, scan_drng = jax.random.split(rng, 3)
        train_state, losses, infos = train_scan(
            model, train_state, all_obs, all_actions, scan_rng, scan_drng,
        )

        if compile_time is None:
            compile_time = time.time()
            print(f"  Compile + first scan: {compile_time - start_time:.2f}s")

        outer_time = time.time() - outer_start

        # Process results for each inner step
        for i in range(steps_remaining):
            step = outer_step + i
            loss = float(losses[i])

            # Extract per-step info dict from the batched scan outputs
            # Skip non-scalar entries (e.g. masked_indices which has trailing dims)
            step_info = {}
            for k, v in infos.items():
                if v.ndim == 1:
                    step_info[k] = float(v[i])
                elif v.ndim == 0:
                    step_info[k] = float(v)
                # Skip arrays with ndim > 1 (per-step arrays like masked_indices)

            step_time = outer_time / steps_remaining

            # Logging every 10 steps
            if step % 10 == 0 or step == 1:
                wandb.log({
                    "train/loss": loss,
                    "train/loss_future": float(step_info.get("loss_future", 0)),
                    "train/loss_masked_history": float(step_info.get("loss_masked_history", 0)),
                    "train/predicted_norm": float(step_info.get("predicted_norm", 0)),
                    "train/target_norm": float(step_info.get("target_norm", 0)),
                    "train/step_time": step_time,
                    "train/step": step,
                }, step=step)

                # Log per-masked-slot MSE
                for key in infos:
                    if key.startswith("slot_"):
                        wandb.log({f"train/{key}": float(step_info[key])}, step=step)
                # Log which actual slot index was masked (scalar survives scan)
                if 'masked_slot_0_idx' in step_info:
                    wandb.log({"train/masked_actual_slot": float(step_info['masked_slot_0_idx'])}, step=step)

            # Print progress every 100 steps
            if step % 100 == 0 or step == 1:
                actual_slot = int(step_info.get('masked_slot_0_idx', -1))
                mse_val = step_info.get('slot_0_mse', 0)
                fut = step_info.get('loss_future', 0)
                hist = step_info.get('loss_masked_history', 0)
                print(f"  Step {step}/{num_train_steps} | loss={loss:.4f} "
                      f"(future={fut:.4f}, hist_masked={hist:.4f}) | "
                      f"masked slot {actual_slot} mse={mse_val:.4f} | "
                      f"{step_time*1000:.1f}ms/step")

            # Evaluation
            if step % eval_every == 0:
                print(f"\n  Evaluating at step {step}...")

                # Validation loss
                rng, val_rng = jax.random.split(rng)
                val_batch = val_dataset.get_batch(batch_size, val_rng)
                val_obs = jnp.concatenate([val_batch["obs"], val_batch["target_obs"]], axis=1)
                val_actions = jnp.concatenate([val_batch["actions"], val_batch["target_actions"]], axis=1)
                val_loss, val_info = model.apply(
                    train_state.params, val_obs, val_actions, rng=rng, training=False,
                    rngs={'dropout': rng},
                )
                wandb.log({
                    "eval/val_loss": float(val_loss),
                }, step=step)

                # Rollout error
                try:
                    rollout_mse = evaluate_rollout(
                        model, train_state.params, val_dataset,
                        rollout_steps=rollout_steps,
                        history_frames=history_frames,
                        pred_frames=pred_frames,
                        rng=rng,
                    )
                    wandb.log({
                        "eval/rollout_mse_avg": float(rollout_mse.get("rollout_mse_avg", 0)),
                    }, step=step)
                    print(f"  Rollout MSE avg: {rollout_mse.get('rollout_mse_avg', 0):.4f}")
                except Exception as e:
                    print(f"  Rollout evaluation failed: {e}")

                # Rollout visualizations (trajectory plots + GIF) logged to wandb
                if eval_viz:
                    try:
                        viz_dir = f"{save_path}/{run_name}/viz" if save_path else "./outputs"
                        viz_paths = generate_eval_visualizations(
                            model, train_state.params, rng=rng,
                            step=step, output_dir=viz_dir,
                            num_frames=viz_frames, start_frame=viz_start_frame,
                            env_id=env_id, seed=step + 42,
                            frame_stack_size=frame_stack_size,
                        )
                        wandb.log({
                            "eval/trajectories": wandb.Image(viz_paths["trajectory_plot"]),
                            "eval/rollout_gif": wandb.Video(viz_paths["rollout_gif"], format="gif"),
                        }, step=step)
                        print(f"  Visualizations logged to wandb")
                    except Exception as e:
                        print(f"  Visualizations failed: {e}")

                # Save checkpoint
                if save_path:
                    checkpoint_dir = f"{save_path}/{run_name}"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = f"{checkpoint_dir}/step_{step}.ckpt"
                    with open(checkpoint_path, "wb") as f:
                        f.write(flax.serialization.to_bytes(train_state.params))
                    print(f"  Checkpoint saved to {checkpoint_path}")

                print()

    # Final evaluation
    print("Training complete. Running final evaluation...")
    rng, val_rng = jax.random.split(rng)
    val_batch = val_dataset.get_batch(batch_size * 4, val_rng)
    val_obs = jnp.concatenate([val_batch["obs"], val_batch["target_obs"]], axis=1)
    val_actions = jnp.concatenate([val_batch["actions"], val_batch["target_actions"]], axis=1)
    final_val_loss, _ = model.apply(
        train_state.params, val_obs, val_actions, rng=rng, training=False,
        rngs={'dropout': rng},
    )
    wandb.log({"eval/final_val_loss": float(final_val_loss)}, step=num_train_steps)

    # Final rollout
    try:
        rollout_mse = evaluate_rollout(
            model, train_state.params, val_dataset,
            rollout_steps=rollout_steps,
            history_frames=history_frames,
            pred_frames=pred_frames,
            rng=rng,
        )
        wandb.log({
            "eval/final_rollout_mse_avg": float(rollout_mse.get("rollout_mse_avg", 0)),
        }, step=num_train_steps)
    except Exception as e:
        print(f"Final rollout evaluation failed: {e}")

    # Save final model
    if save_path:
        checkpoint_dir = f"{save_path}/{run_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_path = f"{checkpoint_dir}/final.ckpt"
        with open(final_path, "wb") as f:
            f.write(flax.serialization.to_bytes(train_state.params))
        print(f"Final model saved to {final_path}")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f}min)")

    wandb.finish()
    return {"final_val_loss": float(final_val_loss), "total_time": total_time}
