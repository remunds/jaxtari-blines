import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class AsteroidsRm(GameRM):

    NUM_FEATURES = 162
    WIDTH_START, WIDTH_END = -104, -87  # asteroids.width of the newest frame
    SCREEN_WIDTH = 160.0                # normalization bound of the width field
    W_LARGE, W_MEDIUM, W_SMALL = 16.0, 8.0, 4.0

    PROP_INDEX = {
        "hit_small": 0,
        "hit_medium": 1,
        "hit_large": 2,
    }

    TRANSITIONS = [
        {"from": 0, "true": ["hit_small"],  "to": 0, "reward": 1.0, "option": True},
        {"from": 0, "true": ["hit_medium"], "to": 0, "reward": 0.5, "option": True},
        {"from": 0, "true": ["hit_large"],  "to": 0, "reward": 0.2, "option": True},
    ]

    def __init__(self):
        (self._from, self._rt, self._rf, self._to, self._rew) = build_transitions(
            len(self.PROP_INDEX), self.PROP_INDEX, self.TRANSITIONS
        )

    def num_states(self):     return 1
    def init_state(self):     return 0
    def terminal_state(self): return -99

    def from_states(self):    return self._from
    def require_true(self):   return self._rt
    def require_false(self):  return self._rf
    def to_states(self):      return self._to
    def rewards(self):        return self._rew

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, obs):
        F = self.NUM_FEATURES
        w_now = jnp.round(obs[self.WIDTH_START:self.WIDTH_END] * self.SCREEN_WIDTH)
        w_prev = jnp.round(
            obs[self.WIDTH_START - F:self.WIDTH_END - F] * self.SCREEN_WIDTH
        )
        # A hit downgrades the asteroid in its own slot: L -> M, M -> S, S -> gone.
        # Split-off mediums spawn into empty slots (0 -> 8) and are not counted.
        hit_large = jnp.any((w_prev == self.W_LARGE) & (w_now == self.W_MEDIUM))
        hit_medium = jnp.any((w_prev == self.W_MEDIUM) & (w_now == self.W_SMALL))
        hit_small = jnp.any((w_prev == self.W_SMALL) & (w_now == 0.0))
        return jnp.array([hit_small, hit_medium, hit_large]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, obs):
        return jnp.zeros(())
