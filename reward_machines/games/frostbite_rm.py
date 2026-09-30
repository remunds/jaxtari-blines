"""
Reward machine for JAXAtari Frostbite.

The Frostbite-specific reward machine, event definitions, and task
decomposition were developed for this project.

The implementation follows the Reward Machine infrastructure provided by
this repository (GameRM, RewardMachine, and build_transitions). Game-specific
event detection is based on the object-centric Frostbite observations exposed
by JAXAtari.
"""


import functools

import jax
import jax.numpy as jnp

from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class FrostbiteRm(GameRM):
    # Events
    PROP_INDEX = {
        "igloo_progressed": 0,
        "igloo_regressed": 1,
        "igloo_completed": 2,
        "lost_life": 3,
        "reached_shore": 4,
        "closer_to_door": 5,
        "at_igloo_door": 6,
        "entered_igloo": 7,
    }

    TRANSITIONS = [
        # State 0: build the igloo
        {"from": 0, "true": ["lost_life"], "to": 0, "reward": -1.0},
        {"from": 0, "true": ["igloo_regressed"], "to": 0, "reward": -0.5},
        {"from": 0, "true": ["igloo_progressed"], "to": 0, "reward": 1.0},
        {"from": 0, "true": ["igloo_completed"], "to": 1, "reward": 5.0},

        # State 1: reach the shore
        {"from": 1, "true": ["lost_life"], "to": 1, "reward": -1.0},
        {"from": 1, "true": ["reached_shore"], "to": 2, "reward": 2.0},

        # State 2: move along the shore toward the real igloo door
        {"from": 2, "true": ["lost_life"], "to": 2, "reward": -1.0},
        {"from": 2, "true": ["at_igloo_door"], "to": 3, "reward": 5.0},
        {"from": 2, "true": ["closer_to_door"], "to": 2, "reward": 0.25},

        # State 3: enter the igloo
        {"from": 3, "true": ["lost_life"], "to": 3, "reward": -1.0},
        {"from": 3, "true": ["entered_igloo"], "to": 0, "reward": 20.0},
    ]

    def __init__(self):
        (self._from, self._rt, self._rf, self._to, self._rew) = build_transitions(
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
        NUM_FEATURES = 181

        BAILEY_X = 0
        BAILEY_Y = 1
        BAILEY_ACTIVE = 4

        IGLOO = -5
        LIVES = -2

        # Real Frostbite constants converted to normalized observation space
        SHORE_Y = 56.0 / 210.0
        SHORE_Y_TOLERANCE = 2.0 / 210.0

        TARGET_DOOR_X = 122.0 / 160.0
        DOOR_X_MIN = 119.0 / 160.0
        DOOR_X_MAX = 126.0 / 160.0

        # Newest frame
        bailey_x_now = obs[-NUM_FEATURES + BAILEY_X]
        bailey_y_now = obs[-NUM_FEATURES + BAILEY_Y]
        bailey_active_now = obs[-NUM_FEATURES + BAILEY_ACTIVE]

        igloo_now = obs[IGLOO]
        lives_now = obs[LIVES]

        # Previous frame
        bailey_x_prev = obs[-2 * NUM_FEATURES + BAILEY_X]
        bailey_y_prev = obs[-2 * NUM_FEATURES + BAILEY_Y]
        bailey_active_prev = obs[-2 * NUM_FEATURES + BAILEY_ACTIVE]

        igloo_prev = obs[IGLOO - NUM_FEATURES]
        lives_prev = obs[LIVES - NUM_FEATURES]

        # -------------------------------------------------
        # Build skill
        # -------------------------------------------------

        igloo_completed = (
            (igloo_now >= 1.0)
            & (igloo_prev < 1.0)
        )

        igloo_progressed = (
            (igloo_now > igloo_prev)
            & ~igloo_completed
        )

        igloo_regressed = (
            (igloo_now < igloo_prev)
            & (bailey_active_now > 0.5)
        )

        lost_life = lives_now < lives_prev

        # -------------------------------------------------
        # Shore / navigation skill
        # -------------------------------------------------

        at_shore_now = (
            jnp.abs(bailey_y_now - SHORE_Y)
            <= SHORE_Y_TOLERANCE
        )

        at_shore_prev = (
            jnp.abs(bailey_y_prev - SHORE_Y)
            <= SHORE_Y_TOLERANCE
        )

        reached_shore = (
            at_shore_now
            & (bailey_active_now > 0.5)
            & (igloo_now >= 1.0)
        )

        at_igloo_door = (
                (igloo_now >= 1.0)
                & (bailey_active_now > 0.5)
                & (bailey_x_now >= DOOR_X_MIN)
                & (bailey_x_now <= DOOR_X_MAX)
                & (bailey_y_now >= (50.0 / 210.0))
                & (bailey_y_now <= (56.0 / 210.0))
        )

        # Reward movement that actually reduces horizontal distance
        # to the real door while Bailey is on the shore.
        distance_now = jnp.abs(
            bailey_x_now - TARGET_DOOR_X
        )

        distance_prev = jnp.abs(
            bailey_x_prev - TARGET_DOOR_X
        )

        closer_to_door = (
            at_shore_now
            & at_shore_prev
            & (igloo_now >= 1.0)
            & (bailey_active_now > 0.5)
            & (distance_now < distance_prev)
            & ~at_igloo_door
        )

        # -------------------------------------------------
        # Final entry skill
        # -------------------------------------------------

        entered_igloo = (
            (bailey_active_prev > 0.5)
            & (bailey_active_now < 0.5)
            & (igloo_prev >= 1.0)
        )

        return jnp.array([
            igloo_progressed,
            igloo_regressed,
            igloo_completed,
            lost_life,
            reached_shore,
            closer_to_door,
            at_igloo_door,
            entered_igloo,
        ]).astype(jnp.int32)