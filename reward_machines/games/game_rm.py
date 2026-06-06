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
    def delta_u(self) -> jnp.ndarray:
        """Transition matrix, shape (num_states, num_prop_combos)."""
        ...

    @abstractmethod
    def delta_r(self) -> jnp.ndarray:
        """Reward matrix, shape (num_states, num_states)."""
        ...

    @abstractmethod
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, obs) -> jnp.ndarray:
        """Game-specific labelling: obs -> boolean proposition vector."""
        ...
