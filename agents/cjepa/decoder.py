"""Slot-to-OC decoder for C-JEPA visualization.

Maps predicted slots (64-dim) back to object-centric attributes (8-dim)
so the model's predictions can be rendered as actual game frames.

Gradients are stopped at the input during training — the decoder
trains independently without affecting the CJEPA model.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import orthogonal, constant


class SlotDecoder(nn.Module):
    """Per-slot MLP decoder: slot vector → stacked OC attributes.

    Decodes to stacked_obj_attr_dim = obj_attr_dim * frame_stack_size
    per object, reconstructing all frame-stacked attributes.
    Applied via vmap over the slot dimension.
    """

    slot_dim: int = 128
    obj_attr_dim: int = 8
    hidden_dim: int = 256
    frame_stack_size: int = 1

    @nn.compact
    def __call__(self, slots: jnp.ndarray) -> jnp.ndarray:
        """Decode slots to OC observations.

        Args:
            slots: [B, T, S, D] — predicted slot vectors.

        Returns:
            oc: [B, T, S, stacked_obj_attr_dim] — decoded OC attributes.
        """
        B, T, S, D = slots.shape
        out_dim = self.obj_attr_dim * self.frame_stack_size

        # Define per-slot MLP
        def mlp(z):
            z = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(z)
            z = nn.relu(z)
            z = nn.Dense(
                out_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(z)
            return z

        # vmap over slot dimension
        vmap_mlp = jax.vmap(mlp, in_axes=-2, out_axes=-2)
        oc = vmap_mlp(slots)  # [B, T, S, stacked_obj_attr_dim]
        return oc
