from abc import ABC, abstractmethod
import functools

import jax
import jax.numpy as jnp

class GameRM(ABC):
    @abstractmethod
    def num_states(self) -> int:
        """Number of RM states (including the terminal state)."""
        ...

    @abstractmethod
    def init_state(self) -> int:
        """Index of the initial RM state."""
        ...

    @abstractmethod
    def terminal_state(self) -> int:
        """Index of the single terminal state."""
        ...

    @abstractmethod
    def from_states(self) -> jnp.ndarray:
        """Array which saves starting state of Transistion i"""
        ...

    @abstractmethod
    def to_states(self) -> jnp.ndarray:
        """Array which saves destination state of Transistion i"""
        ...

    @abstractmethod
    def require_false(self) -> jnp.ndarray:
        """Saves which props need to be false in a Transistion"""
        ...

    @abstractmethod
    def require_true(self) -> jnp.ndarray:
        """Saves which props need to be true in a Transistion"""
        ...

    @abstractmethod
    def rewards(self) -> jnp.ndarray:
        """Saves reward of a Transistion"""
        ...

    @abstractmethod
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, obs) -> jnp.ndarray:
        """Game-specific labelling: obs -> boolean proposition vector."""
        ...
