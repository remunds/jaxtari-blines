import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class EnduroRm(GameRM):
    NUM_FEATURES = 17
    # Opponent window layout (JAXAtari master): base y and y-range of each slot
    SLOT_BASE_Y = (135.0, 115.0, 95.0, 85.0, 75.0, 69.0, 64.0)
    SLOT_HEIGHT = (20.0, 20.0, 20.0, 10.0, 10.0, 6.0, 5.0)
    # Normalization bounds of enemy positions: x in [-1, 160], y in [-1, 210]
    X_RANGE, Y_RANGE = 161.0, 211.0

    PROP_INDEX = {"overtook": 0, "got_overtaken": 1}

    # Single state: mirrors the game's scoring (+1 per overtake, -1 per being overtaken)
    TRANSITIONS = [
        {"from": 0, "true": ["overtook"], "to": 0, "reward": 1.0},
        {"from": 0, "true": ["got_overtaken"], "to": 0, "reward": -1.0},
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


    # ... __init__ and accessors unchanged ...

    def _window_phase(self, frame):
        """Fractional position of the opponent window in [0, 1), or -1 if no car is visible.

        All visible cars share the same fractional window index:
        y = base_y[slot] + floor(phase * slot_height[slot]).
        """
        xs = frame[0:14:2] * self.X_RANGE - 1.0
        ys = frame[1:14:2] * self.Y_RANGE - 1.0
        present = xs > -0.5  # empty slots are encoded as x = y = -1
        phase = jnp.clip(
            (ys - jnp.array(self.SLOT_BASE_Y)) / jnp.array(self.SLOT_HEIGHT), 0.0, 0.999
        )
        n_visible = present.sum()
        mean_phase = jnp.sum(phase * present) / jnp.maximum(n_visible, 1)
        return jnp.where(n_visible > 0, mean_phase, -1.0), present[0]

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, obs):
        now = obs[-self.NUM_FEATURES:]
        prev = obs[-2 * self.NUM_FEATURES:-self.NUM_FEATURES]
        phase_now, nearest_now = self._window_phase(now)
        phase_prev, nearest_prev = self._window_phase(prev)

        known = (phase_now >= 0) & (phase_prev >= 0)
        window_forward = known & (phase_prev - phase_now > 0.5)   # phase wrapped 1 -> 0
        window_backward = known & (phase_now - phase_prev > 0.5)  # phase wrapped 0 -> 1
        # Fallback: the nearest car was about to leave and no car is visible anymore
        vanished = nearest_prev & (phase_now < 0) & (phase_prev > 0.5)

        overtook = (window_forward & nearest_prev) | vanished
        got_overtaken = window_backward & nearest_now
        return jnp.array([overtook, got_overtaken]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, obs):
        return jnp.zeros(())
