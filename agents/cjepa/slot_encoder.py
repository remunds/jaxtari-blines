"""Per-object MLP slot encoder for C-JEPA.

Maps object-centric (OC) observations to slot latents using a shared MLP
applied via jax.vmap over the slot/object dimension.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import orthogonal, constant


class SlotEncoder(nn.Module):
    """A per-object MLP slot encoder.

    Input:  flattened OC observation [B, T, obj_attr_dim * num_objects]
    Reshapes to [B, T, S, obj_attr_dim] where S = num_slots (= num_objects).
    Applies a shared MLP via jax.vmap over the slot dimension.
    Output: [B, T, S, slot_dim]
    """

    num_slots: int
    obj_attr_dim: int
    slot_dim: int = 64
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, T, obj_attr_dim * num_slots]
        B, T, _ = x.shape

        # Reshape to [B, T, S, obj_attr_dim]
        x = x.reshape(B, T, self.num_slots, self.obj_attr_dim)

        # Define the per-object MLP
        def mlp(z):
            z = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(z)
            z = nn.relu(z)
            z = nn.Dense(
                self.slot_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(z)
            return z

        # vmap over slot dimension: apply mlp to each object's attributes
        # Input: [B, T, S, obj_attr_dim] → vmap over S → [B, T, S, slot_dim]
        vmap_mlp = jax.vmap(mlp, in_axes=-2, out_axes=-2)
        slots = vmap_mlp(x)

        return slots
