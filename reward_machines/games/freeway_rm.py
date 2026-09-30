"""
Reward machine for JAXAtari Freeway.

The Freeway-specific task decomposition is based on vertical progress of the
chicken while crossing the road. The object-centric observation exposes the
chicken position directly.

The RM divides one successful crossing into four sequential subgoals:
lower-road progress, middle-road progress, upper-road progress, and a
completed crossing.

Transitions representing meaningful progress are marked as options for
compatibility with the option-aware / HRM infrastructure.
"""

import functools

import jax
import jax.numpy as jnp

from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class FreewayRm(GameRM):

    # ------------------------------------------------------------------
    # Observation layout
    # ------------------------------------------------------------------
    #
    # 4 stacked frames = 352 features
    # 1 frame          = 88 features
    #
    # chicken.x = index 0
    # chicken.y = index 1
    #
    NUM_FEATURES = 88
    CHICKEN_Y = 1

    Y_SCALE = 210.0

    # Progress thresholds in original screen coordinates.
    #
    # The chicken starts around y=187 and moves upward, so smaller y means
    # more progress.
    #
    # These boundaries roughly divide the ten traffic lanes into stages.
    LOWER_THRESHOLD = 135.0
    MIDDLE_THRESHOLD = 87.0
    UPPER_THRESHOLD = 39.0

    # After reaching the top, Freeway immediately respawns the chicken at
    # the bottom. Detect the characteristic top -> bottom jump.
    CROSS_TOP_MAX = 25.0

    # Shaping potential: normalized vertical progress of the chicken.
    # Observed y range in JAXAtari: start ~188, top ~17.
    Y_START = 188.0
    Y_TOP = 17.0
    PHI_SCALE = 0.2
    CROSS_BOTTOM_MIN = 180.0

    PROP_INDEX = {
        "reached_lower": 0,
        "reached_middle": 1,
        "reached_upper": 2,
        "crossed": 3,
    }

    # Each milestone receives one quarter of the reward of a complete
    # crossing. Progress cannot be repeatedly farmed because the RM moves
    # forward and resets only after a completed crossing.
    #
    # option=True marks meaningful subgoals for option-aware / HRM infrastructure.
    TRANSITIONS = [
        {
            "from": 0,
            "true": ["reached_lower"],
            "to": 1,
            "reward": 0.25,
            "option": True,
        },
        {
            "from": 1,
            "true": ["reached_middle"],
            "to": 2,
            "reward": 0.25,
            "option": True,
        },
        {
            "from": 2,
            "true": ["reached_upper"],
            "to": 3,
            "reward": 0.25,
            "option": True,
        },
        {
            "from": 3,
            "true": ["crossed"],
            "to": 0,
            "reward": 0.25,
            "option": True,
        },
    ]

    def __init__(self):
        (
            self._from,
            self._rt,
            self._rf,
            self._to,
            self._rew,
        ) = build_transitions(
            len(self.PROP_INDEX),
            self.PROP_INDEX,
            self.TRANSITIONS,
        )

    def num_states(self):
        return 4

    def init_state(self):
        return 0

    def terminal_state(self):
        return -99

    def from_states(self):
        return self._from

    def require_true(self):
        return self._rt

    def require_false(self):
        return self._rf

    def to_states(self):
        return self._to

    def rewards(self):
        return self._rew

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, obs):

        frames = obs.reshape(-1, self.NUM_FEATURES)

        now = frames[-1]
        prev = frames[-2]

        y_now = now[self.CHICKEN_Y] * self.Y_SCALE
        y_prev = prev[self.CHICKEN_Y] * self.Y_SCALE

        # Chicken moves upward when y decreases.
        reached_lower = (
            (y_prev > self.LOWER_THRESHOLD)
            & (y_now <= self.LOWER_THRESHOLD)
        )

        reached_middle = (
            (y_prev > self.MIDDLE_THRESHOLD)
            & (y_now <= self.MIDDLE_THRESHOLD)
        )

        reached_upper = (
            (y_prev > self.UPPER_THRESHOLD)
            & (y_now <= self.UPPER_THRESHOLD)
        )

        # A successful crossing causes an immediate jump from near the top
        # back to the starting region at the bottom.
        crossed = (
            (y_prev <= self.CROSS_TOP_MAX)
            & (y_now >= self.CROSS_BOTTOM_MIN)
        )

        return jnp.array(
            [
                reached_lower,
                reached_middle,
                reached_upper,
                crossed,
            ]
        ).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, obs):
        """Phi(s) = PHI_SCALE * progress of the chicken towards the top, in [0, PHI_SCALE]."""
        now = obs.reshape(-1, self.NUM_FEATURES)[-1]
        y = now[self.CHICKEN_Y] * self.Y_SCALE
        progress = (self.Y_START - y) / (self.Y_START - self.Y_TOP)
        return self.PHI_SCALE * jnp.clip(progress, 0.0, 1.0)
