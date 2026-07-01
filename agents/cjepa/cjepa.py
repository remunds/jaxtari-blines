"""C-JEPA world model — combines all components into a single model.

Architecture:
    OC observation [B, T, obj_attr_dim * num_objects]
        │
        ▼
    SlotEncoder → slots [B, T, S, slot_dim]
        │
        ├──► Target branch (stop-gradient): target_slots [B, T, S, slot_dim]
        │
        └──► Context branch → MaskedSlotPredictor → predicted [B, T_total, S, slot_dim]
                                                              │
                                                              ▼
                                                    JEPA Loss (masked slots only)
"""

from typing import Dict, Tuple, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import flax.training.train_state
import optax

from agents.cjepa.slot_encoder import SlotEncoder
from agents.cjepa.action_encoder import ActionEncoder
from agents.cjepa.predictor import MaskedSlotPredictor
from agents.cjepa.decoder import SlotDecoder


class CJEPA(nn.Module):
    """C-JEPA world model.

    Maps OC observations to slot latents and predicts masked/future slots.

    Args:
        num_slots: Number of object slots (inferred from env).
        obj_attr_dim: Dimension of each object's attributes.
        num_actions: Number of discrete actions.
        slot_dim: Dimension of slot latent.
        history_frames: Number of history frames.
        pred_frames: Number of future frames to predict.
        max_masked_slots: Max number of slots to mask (actual k in [0, max]).
        action_emb_dim: Dimension of action embedding.
        transformer_depth: Depth of non-causal transformer.
        transformer_heads: Number of attention heads.
        transformer_dim_head: Dimension per head.
        transformer_mlp_dim: MLP hidden dimension.
        dropout: Dropout rate.
    """
    num_slots: int
    obj_attr_dim: int
    num_actions: int
    slot_dim: int = 128
    history_frames: int = 3
    pred_frames: int = 1
    max_masked_slots: int = 2
    action_emb_dim: int = 128
    transformer_depth: int = 6
    transformer_heads: int = 16
    transformer_dim_head: int = 64
    transformer_mlp_dim: int = 2048
    dropout: float = 0.1
    decoder_loss_weight: float = 1.0

    def setup(self):
        self.slot_encoder = SlotEncoder(
            num_slots=self.num_slots,
            obj_attr_dim=self.obj_attr_dim,
            slot_dim=self.slot_dim,
        )
        self.action_encoder = ActionEncoder(
            num_actions=self.num_actions,
            emb_dim=self.action_emb_dim,
        )
        self.predictor = MaskedSlotPredictor(
            num_slots=self.num_slots,
            slot_dim=self.slot_dim,
            history_frames=self.history_frames,
            pred_frames=self.pred_frames,
            max_masked_slots=self.max_masked_slots,
            depth=self.transformer_depth,
            num_heads=self.transformer_heads,
            dim_head=self.transformer_dim_head,
            mlp_dim=self.transformer_mlp_dim,
            dropout=self.dropout,
        )
        # Action injection: project action_emb to slot_dim and add to slots
        # In the original Push-T setup, action_emb_dim == slot_dim == 128,
        # so this is an identity-like learned projection.
        self.action_proj = nn.Dense(
            self.slot_dim,
            kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=jax.nn.initializers.constant(0.0),
        )
        # Slot → OC decoder (trained with stop_gradient; CJEPA stays frozen)
        self.decoder = SlotDecoder(
            slot_dim=self.slot_dim,
            obj_attr_dim=self.obj_attr_dim,
        )

    def decode_slots(
        self,
        slots: jnp.ndarray,
    ) -> jnp.ndarray:
        """Decode slot vectors back to OC attributes.

        Args:
            slots: [B, T, S, D] slot vectors.

        Returns:
            oc_attrs: [B, T, S, obj_attr_dim] decoded attributes.
        """
        return self.decoder(slots)

    def encode_slots(
        self,
        obs: jnp.ndarray,
        params: Optional[Dict] = None,
    ) -> jnp.ndarray:
        """Encode OC observations to slot latents.

        Args:
            obs: OC observations [B, T, obj_attr_dim * num_slots]

        Returns:
            slots: [B, T, S, slot_dim]
        """
        return self.slot_encoder(obs)

    def masked_history_loss(
        self,
        predicted: jnp.ndarray,
        targets: jnp.ndarray,
        T_hist: int,
        masked_indices: jnp.ndarray,
        active_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """MSE on masked slots at history timesteps only.

        This is the C-JEPA core loss: the model must infer masked object
        states at t=1..T_hist-1 using visible context from other objects.

        Args:
            predicted: [B, T_total, S, D]
            targets: [B, T_total, S, D] (with stop-gradient)
            T_hist: Number of history frames.
            masked_indices: [max_masked_slots]
            active_mask: [max_masked_slots]

        Returns:
            loss: scalar MSE on active masked slots at history timesteps
        """
        # Select candidate masked slots at history timesteps
        pred_masked = predicted[:, :T_hist, masked_indices, :]  # [B, T_hist, max_k, D]
        targ_masked = targets[:, :T_hist, masked_indices, :]    # [B, T_hist, max_k, D]

        # Per-slot MSE at each masked position: [max_k]
        per_slot_mse = jnp.mean((pred_masked - targ_masked) ** 2, axis=(0, 1, 3))

        # Average over active slots only (inactive=padding, ignored)
        num_active = jnp.maximum(jnp.sum(active_mask), 1)
        loss = jnp.sum(per_slot_mse * active_mask) / num_active
        return loss

    def future_loss(
        self,
        predicted: jnp.ndarray,
        targets: jnp.ndarray,
        T_hist: int,
    ) -> jnp.ndarray:
        """MSE on ALL slots at future timesteps.

        Always computed regardless of masking — the model must predict
        future states for all objects.

        Args:
            predicted: [B, T_total, S, D]
            targets: [B, T_total, S, D] (with stop-gradient)
            T_hist: Number of history frames.

        Returns:
            loss: scalar MSE on all slots at future timesteps
        """
        return jnp.mean((predicted[:, T_hist:, :, :] - targets[:, T_hist:, :, :]) ** 2)

    def __call__(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        rng: jax.Array,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Alias for compute_loss — default forward pass.

        Args:
            obs: [B, T_total, obj_attr_dim * num_slots]
            actions: [B, T_total]
        """
        return self.compute_loss(obs, actions, rng=rng, training=training)

    def compute_loss(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        rng: jax.Array,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Full JEPA loss computation.

        The observations include BOTH history and future frames.
        First T_hist frames are used as context; all T_total frames
        serve as targets for the loss on masked slots.

        Args:
            obs: OC observations [B, T_total, obj_attr_dim * num_slots]
                 where T_total = T_hist + T_pred.
            actions: Discrete action indices [B, T_total]
            rng: PRNG key.
            training: Whether to apply dropout / masking.

        Returns:
            loss: JEPA loss scalar.
            info: Dict with debug information.
        """
        _, T_total, _ = obs.shape
        T_hist = self.history_frames

        # 1. Encode ALL observations to slots (shared encoder)
        all_slots = self.encode_slots(obs)  # [B, T_total, S, D]

        # 2. Encode actions
        action_emb = self.action_encoder(actions)  # [B, T_total, action_emb_dim]
        action_proj = self.action_proj(action_emb)  # [B, T_total, D]

        # 3. Target branch (stop-gradient): slots + action projection
        # Shape: [B, T_total, S, D]
        target_slots = all_slots + action_proj[:, :, None, :]
        target_slots = jax.lax.stop_gradient(target_slots)

        # 4. Context branch: only use first T_hist frames
        context_slots = all_slots[:, :T_hist, :, :] + action_proj[:, :T_hist, None, :]

        # 5. Predictor: context → predicted for all T_total frames
        predicted, masked_indices, active_mask = self.predictor(
            context_slots, rng=rng, training=training
        )
        # predicted: [B, T_total, S, D]
        # active_mask: [max_masked_slots] — which candidate slots are actually masked

        # 6a. Future loss: MSE on ALL slots at future timesteps (always present)
        loss_future = self.future_loss(predicted, target_slots, T_hist)

        # 6b. Masked history loss: MSE on masked slots at history timesteps
        #   Only computed when slots are actually masked (k > 0).
        #   This is the core C-JEPA interaction reasoning objective.
        loss_masked_history = self.masked_history_loss(
            predicted, target_slots, T_hist, masked_indices, active_mask,
        )

        # Total loss
        loss = loss_future + loss_masked_history

        # Number of actually-masked slots (for logging)
        num_active = jnp.sum(active_mask)

        info = {
            'loss': loss,
            'loss_future': loss_future,
            'loss_masked_history': loss_masked_history,
            'masked_indices': masked_indices,
            'num_masked': num_active,
            'predicted_norm': jnp.mean(jnp.abs(predicted)),
            'target_norm': jnp.mean(jnp.abs(target_slots)),
        }

        # Log the first active masked slot index (or -1 if none masked)
        first_active = jnp.where(
            jnp.any(active_mask),
            masked_indices[0],
            -1,
        )
        info['masked_slot_0_idx'] = first_active

        # Per-slot MSE at history timesteps for debugging.
        # NOTE: keys must be STATIC across scan steps, so we index by masked-slot-position
        # (0, 1, 2), not by actual slot index.
        for i in range(min(int(self.max_masked_slots), 3)):
            slot_idx = masked_indices[i]
            slot_mse = jnp.mean(
                (predicted[:, :T_hist, slot_idx, :] - target_slots[:, :T_hist, slot_idx, :]) ** 2
            )
            # Zero out MSE for inactive (padding) masked slots
            info[f'slot_{int(i)}_mse'] = jnp.where(active_mask[i], slot_mse, 0.0)

        # 7. Auxiliary decoder loss: decode predicted slots back to OC
        #    stop_gradient prevents gradients from flowing to encoder/predictor
        decoded = self.decoder(jax.lax.stop_gradient(predicted))  # [B, T, S, obj_attr_dim]
        # Reshape GT obs to match: [B, T, S, obj_attr_dim]
        gt_oc = obs.reshape(*obs.shape[:2], self.num_slots, self.obj_attr_dim)
        decoder_loss = jnp.mean((decoded - gt_oc) ** 2)
        info['decoder_loss'] = decoder_loss
        # Decoder loss only updates decoder params (source inputs are stop_gradient'd)
        loss = loss + self.decoder_loss_weight * decoder_loss

        return loss, info

    def predict_future(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        num_future: int = 1,
    ) -> jnp.ndarray:
        """Predict future slots autoregressively (for evaluation).

        Args:
            obs: OC observations [B, T_hist, obj_attr_dim * num_slots]
            actions: Actions [B, T_hist + num_future - 1]
            num_future: Number of future steps to predict.

        Returns:
            predicted_slots: [B, T_hist + num_future, S, D]
        """
        B, T_hist, _ = obs.shape

        # Encode all history
        slots = self.encode_slots(obs)  # [B, T_hist, S, D]
        action_emb = self.action_encoder(actions)  # [B, T_hist+num_future-1, act_dim]
        action_proj = self.action_proj(action_emb)

        # Add actions to slots for history
        context_slots = slots + action_proj[:, :T_hist, None, :]

        # Predict first batch
        future_pred = self.predictor.inference(context_slots)  # [B, pred_frames, S, D]

        # For multi-step, use the predicted as part of next context
        all_slots = jnp.concatenate([context_slots, future_pred], axis=1)

        if num_future > self.predictor.pred_frames:
            remaining = num_future - self.predictor.pred_frames
            for t in range(remaining):
                # Use last T_hist frames as context
                context = all_slots[:, -(self.predictor.history_frames):, :, :]
                next_pred = self.predictor.inference(context)  # [B, pred_frames, S, D]
                all_slots = jnp.concatenate([all_slots, next_pred], axis=1)

                # Inject action for the next step
                act_idx = T_hist + self.predictor.pred_frames + t
                if act_idx < actions.shape[1]:
                    act_emb = self.action_proj(
                        self.action_encoder(actions[:, act_idx:act_idx+1])
                    )  # [B, 1, D]
                    all_slots = all_slots.at[:, -1, :, :].add(act_emb[:, 0, None, :])

        return all_slots[:, :T_hist + num_future, :, :]


def create_train_state(
    model: nn.Module,
    rng: jax.Array,
    sample_obs: jnp.ndarray,
    sample_actions: jnp.ndarray,
    learning_rate: float = 5e-4,
    max_grad_norm: float = 1.0,
) -> Tuple[flax.training.train_state.TrainState, Dict]:
    """Create initial training state with optimizer.

    Args:
        model: CJEPA model instance.
        rng: PRNG key.
        sample_obs: Sample observation [1, T_hist, obj_attr_dim * num_slots].
        sample_actions: Sample actions [1, T_hist].
        learning_rate: Learning rate.
        max_grad_norm: Maximum gradient norm for clipping.

    Returns:
        train_state: Flax TrainState.
        model_params: Model parameters (for reference).
    """
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    params = model.init(
        {'params': init_rng, 'dropout': dropout_rng},
        sample_obs, sample_actions, rng=rng, training=True,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=learning_rate, eps=1e-5, weight_decay=1e-4),
    )

    train_state = flax.training.train_state.TrainState.create(
        apply_fn=None,
        params=params,
        tx=tx,
    )
    return train_state, params
