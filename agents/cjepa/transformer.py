"""Non-causal transformer encoder for C-JEPA.

Standard transformer encoder with full (non-causal) self-attention.
Processes flattened token sequences.
"""

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import orthogonal, constant, zeros


class FeedForward(nn.Module):
    """MLP feed-forward block with pre-norm."""
    dim: int
    mlp_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = nn.LayerNorm()(x)
        x = nn.Dense(
            self.mlp_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        x = nn.Dense(
            self.dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        return x


class SelfAttention(nn.Module):
    """Multi-head self-attention with pre-norm (no causal mask)."""
    dim: int
    num_heads: int = 8
    dim_head: int = 64
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Pre-norm
        x = nn.LayerNorm()(x)
        # Multi-head attention (no causal mask → full attention)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.num_heads * self.dim_head,
            out_features=self.dim,
            dropout_rate=self.dropout,
            deterministic=not training,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x, x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer encoder block: self-attention + feed-forward with residual."""
    dim: int
    num_heads: int = 8
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Self-attention + residual
        attn_out = SelfAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            dim_head=self.dim_head,
            dropout=self.dropout,
        )(x, training=training)
        x = x + attn_out

        # Feed-forward + residual
        ff_out = FeedForward(
            dim=self.dim,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
        )(x, training=training)
        x = x + ff_out
        return x


class NonCausalTransformer(nn.Module):
    """Stack of transformer encoder blocks with non-causal (full) attention.

    Args:
        dim: Token dimension (slot_dim).
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        dim_head: Dimension per head.
        mlp_dim: Hidden dimension of MLP.
        dropout: Dropout rate.
    """
    dim: int
    depth: int = 6
    num_heads: int = 8
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # x: [B, seq_len, D]
        for _ in range(self.depth):
            x = TransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                dim_head=self.dim_head,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
            )(x, training=training)
        x = nn.LayerNorm()(x)
        return x
