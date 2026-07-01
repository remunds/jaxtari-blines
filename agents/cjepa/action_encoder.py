"""Discrete action encoder for C-JEPA.

Embeds discrete action indices into continuous vectors and projects
them to the slot dimension for injection into the predictor.
"""

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import orthogonal, constant


class ActionEncoder(nn.Module):
    """Encodes discrete actions into continuous embeddings.

    Input:  discrete action indices [B, T] (integers in [0, num_actions))
    Output: action embeddings [B, T, emb_dim]

    The embedding is later tiled across all slots and concatenated
    to the slot features.
    """

    num_actions: int
    emb_dim: int = 128
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, T] — integer actions
        # Embedding lookup
        x = nn.Embed(
            num_embeddings=self.num_actions,
            features=self.emb_dim,
        )(x)
        # Small MLP projection
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.emb_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        return x
