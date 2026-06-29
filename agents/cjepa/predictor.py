"""Masked Slot Predictor — the core C-JEPA mechanism.

Port of `src/cjepa_predictor.py:MaskedSlotPredictor` from PyTorch to Flax/JAX.

Applies object-level masking: entire object slots are masked across future
timesteps, forcing the model to infer an object's state from interactions
with other objects via the non-causal transformer.
"""

from typing import Tuple, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import orthogonal, constant, zeros, truncated_normal

from agents.cjepa.transformer import NonCausalTransformer


class MaskedSlotPredictor(nn.Module):
    """Predicts masked/future slots given a history of slots and actions.

    Architecture:
        mask_token + time_pos_embed + anchor_query → NonCausalTransformer → to_out

    Args:
        num_slots: Total number of slots per frame.
        slot_dim: Dimension of each slot.
        history_frames: Number of input (history) frames.
        pred_frames: Number of future frames to predict.
        num_masked_slots: Number of object slots to mask during training.
        depth: Transformer depth.
        num_heads: Number of attention heads.
        dim_head: Dimension per head.
        mlp_dim: MLP hidden dimension.
        dropout: Dropout rate.
    """
    num_slots: int
    slot_dim: int = 64
    history_frames: int = 3
    pred_frames: int = 1
    num_masked_slots: int = 1
    depth: int = 6
    num_heads: int = 8
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.1

    def setup(self):
        self.total_frames = self.history_frames + self.pred_frames

        # 1. Learnable Mask Token — represents missing data
        self.mask_token = self.param(
            'mask_token',
            truncated_normal(stddev=0.02),
            (1, 1, self.slot_dim),
        )

        # 2. Time Positional Embedding — shared across slots, per timestep
        self.time_pos_embed = self.param(
            'time_pos_embed',
            truncated_normal(stddev=0.02),
            (1, self.total_frames, 1, self.slot_dim),
        )

        # 3. ID Projector (Anchor mechanism)
        # Projects t=0 slot into a query: "predict what this object does"
        self.id_projector = nn.Dense(
            self.slot_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )

        # 4. Transformer backbone
        self.transformer = NonCausalTransformer(
            dim=self.slot_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
        )

        # 5. Output head
        self.to_out = nn.Dense(
            self.slot_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )

    def get_mask_indices(
        self,
        rng: jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Select N slots to mask.

        Returns:
            is_slot_masked: boolean [num_slots] — True means masked (target).
            masked_indices: int [num_masked_slots] — indices of masked slots.
        """
        if self.num_masked_slots <= 0:
            masked_indices = jnp.array([], dtype=jnp.int32)
            is_slot_masked = jnp.zeros(self.num_slots, dtype=jnp.bool_)
            return is_slot_masked, masked_indices

        # Randomly select N distinct slots to mask
        # Using permutation + first N (handles case when num_masked_slots > num_slots)
        perm = jax.random.permutation(rng, self.num_slots)
        masked_indices = perm[:self.num_masked_slots]

        is_slot_masked = jnp.zeros(self.num_slots, dtype=jnp.bool_)
        is_slot_masked = is_slot_masked.at[masked_indices].set(True)
        return is_slot_masked, masked_indices

    def prepare_input(
        self,
        x: jnp.ndarray,
        rng: jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Construct the transformer input by mixing real data with query tokens.

        Logic:
        - t=0: ALWAYS visible (real data) for ALL slots.
        - Masked (target) slots: visible at t=0, query tokens at t>=1.
        - Unmasked (context) slots: visible at t=0..T_hist-1, query at future.
        - Future frames (t >= T_hist): query tokens for all slots.

        Args:
            x: Ground truth slots [B, T_hist, S, D]
            rng: PRNG key for masking randomness.

        Returns:
            full_input: [B, T_total, S, D] — mixed real + query tokens
            masked_indices: [num_masked_slots] — indices of masked slots
        """
        B, T_hist, S, D = x.shape
        T_total = self.total_frames

        # 1. Get mask indices
        is_slot_masked, masked_indices = self.get_mask_indices(rng)

        # 2. Compute anchor queries from t=0
        # anchors: [B, S, D]
        anchors = x[:, 0, :, :]
        anchor_queries = self.id_projector(anchors)  # [B, S, D]

        # 3. Construct query grid (default for everything)
        # mask_token: [1, 1, 1, D] -> broadcast to [B, T_total, S, D]
        # time_pos_embed: [1, T_total, 1, D] -> broadcast to [B, T_total, S, D]
        # anchor_queries: [B, 1, S, D] -> broadcast to [B, T_total, S, D]
        query_input = (
            self.mask_token +
            self.time_pos_embed +
            anchor_queries[:, None, :, :]
        )

        # 4. Overwrite visible positions with real data
        final_input = query_input

        # (A) t=0: ALL slots get real data + time_pos_embed[0]
        final_input = final_input.at[:, 0, :, :].set(
            x[:, 0, :, :] + self.time_pos_embed[:, 0, :, :]
        )

        # (B) Unmasked slots at history frames (t=1 to T_hist-1) get real data
        num_unmasked = S - self.num_masked_slots
        if num_unmasked > 0 and T_hist > 1:
            # Get unmasked indices with static size
            unmasked_indices = jnp.where(~is_slot_masked, size=num_unmasked)[0]
            # real_history: [B, T_hist-1, num_unmasked, D]
            real_history = x[:, 1:T_hist, :, :][:, :, unmasked_indices, :]
            # time_pos: [1, T_hist-1, 1, D] -> broadcast
            history_pos = self.time_pos_embed[:, 1:T_hist, :, :]
            # Overwrite
            final_input = final_input.at[:, 1:T_hist, unmasked_indices, :].set(
                real_history + history_pos
            )

        return final_input, masked_indices

    def __call__(
        self,
        x: jnp.ndarray,
        rng: Optional[jax.Array] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Training forward pass with masking.

        Args:
            x: Ground truth slots [B, T_hist, S, D]
            rng: PRNG key for masking (required if training=True).
            training: Whether to apply dropout / masking.

        Returns:
            out: Predicted slots [B, T_total, S, D]
            masked_indices: [num_masked_slots] indices of masked slots
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)

        B, T_hist, S, D = x.shape
        T_total = self.total_frames

        # 1. Prepare mixed input
        x_input, masked_indices = self.prepare_input(x, rng)

        # 2. Flatten (T, S) -> sequence for transformer
        # [B, T_total, S, D] -> [B, T_total * S, D]
        x_flat = x_input.reshape(B, T_total * S, D)

        # 3. Run non-causal transformer
        out_flat = self.transformer(x_flat, training=training)

        # 4. Unflatten back
        out = out_flat.reshape(B, T_total, S, D)

        # 5. Output projection
        out = self.to_out(out)

        return out, masked_indices

    def inference(
        self,
        x: jnp.ndarray,
        training: bool = False,
        rng: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        """Inference forward pass (no masking).

        Takes full history and predicts future frames without any masking.

        Args:
            x: Full history slots [B, T_hist, S, D]

        Returns:
            future_prediction: [B, pred_frames, S, D]
        """
        B, T_hist, S, D = x.shape
        T_pred = self.pred_frames
        T_total = T_hist + T_pred

        # Use only the relevant time positions
        # time_pos_embed is [1, total_frames, 1, D] where total_frames = history_frames + pred_frames
        # For inference, total input = T_hist + T_pred may differ from self.total_frames
        # We use the last T_total positions from time_pos_embed
        inf_time_pos = self.time_pos_embed[:, -T_total:, :, :]

        # 1. Anchor queries from t=0
        anchors = x[:, 0, :, :]
        anchor_queries = self.id_projector(anchors)  # [B, S, D]

        # 2. History part — real data + time pos
        input_history = x + inf_time_pos[:, :T_hist, :, :]

        # 3. Future part — query tokens
        tokens_grid = jnp.broadcast_to(
            self.mask_token, (B, T_pred, S, D)
        )
        pos_grid = jnp.broadcast_to(
            inf_time_pos[:, T_hist:T_total, :, :], (B, T_pred, S, D)
        )
        anchor_grid = jnp.broadcast_to(
            anchor_queries[:, None, :, :], (B, T_pred, S, D)
        )
        input_future = tokens_grid + pos_grid + anchor_grid

        # 4. Concatenate and flatten
        full_input = jnp.concatenate([input_history, input_future], axis=1)
        x_flat = full_input.reshape(B, T_total * S, D)

        # 5. Run transformer
        out_flat = self.transformer(x_flat, training=training)

        # 6. Unflatten + output projection
        out = out_flat.reshape(B, T_total, S, D)
        out = self.to_out(out)

        # Return only predicted future frames
        return out[:, -T_pred:, :, :]
