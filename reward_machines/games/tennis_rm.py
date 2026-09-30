"""
Reward machine for JAXAtari Tennis.

The RM tracks four phases:

    u0 = READY
         Waiting for the next rally.

    u1 = SERVE_READY
         Player-side serve geometry is in a hit-ready configuration.

    u2 = RALLY
         The ball is actively in play.

    u3 = POST_POINT
         A point has just been scored. Wait until stale stacked ball motion
         has disappeared before returning to READY.

For player serves, the rally-start shaping reward can be split into:

    READY -> SERVE_READY : +0.10
    SERVE_READY -> RALLY : +0.15

If the intermediate serve-ready phase is not observed,
READY -> RALLY receives +0.25 directly.
"""

import functools

import jax
import jax.numpy as jnp

from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class TennisRm(GameRM):
    # Proposition order must match get_events().
    PROP_INDEX = {
        "ball_in_play": 0,
        "serve_ready": 1,
        "player_point": 2,
        "enemy_point": 3,
        "player_game_progress": 4,
        "enemy_game_progress": 5,
    }

    # HRM options (edges marked "option": True). Options are keyed by guard
    # formula, so edges with the same guard share one option policy:
    #   serve_ready   (u0)         -> get into serve position
    #   ball_in_play  (u0, u1)     -> start the rally
    #   player_point  (u0, u1, u2) -> win the point (also covers game progress,
    #                                 since player_point is set then too)
    #   !ball_in_play (u3)         -> wait for the next rally
    # Penalty edges (enemy_point / enemy_game_progress) are deliberately not
    # options, see RewardMachineWrapper._option_signals.
    PHI_SCALE = 0.1
    PHI_MAX_DIST = 0.8  # ~max normalized player-ball distance observed

    # RM states:
    #
    # u0 = READY
    # u1 = SERVE_READY
    # u2 = RALLY
    # u3 = POST_POINT
    #
    # A higher-level score increment can coincide with a point increment,
    # so these transitions are checked first.
    TRANSITIONS = [
        # ---- u0: READY -------------------------------------------------
        {"from": 0, "true": ["player_game_progress"],
         "to": 3, "reward": 3.0},

        {"from": 0, "true": ["enemy_game_progress"],
         "to": 3, "reward": -3.0},

        {"from": 0, "true": ["player_point"],
         "to": 3, "reward": 1.0, "option": True},

        {"from": 0, "true": ["enemy_point"],
         "to": 3, "reward": -1.0},

        # Direct rally start when the intermediate serve-ready phase
        # is not observed.
        {"from": 0, "true": ["ball_in_play"],
         "to": 2, "reward": 0.25, "option": True},

        # Player-side serve geometry is in a hit-ready configuration.
        {"from": 0, "true": ["serve_ready"],
         "to": 1, "reward": 0.10, "option": True},

        # ---- u1: SERVE_READY -------------------------------------------
        {"from": 1, "true": ["player_game_progress"],
         "to": 3, "reward": 3.0},

        {"from": 1, "true": ["enemy_game_progress"],
         "to": 3, "reward": -3.0},

        {"from": 1, "true": ["player_point"],
         "to": 3, "reward": 1.0, "option": True},

        {"from": 1, "true": ["enemy_point"],
         "to": 3, "reward": -1.0},

        # Complete the rally-start shaping reward.
        {"from": 1, "true": ["ball_in_play"],
         "to": 2, "reward": 0.15, "option": True},

        # Keep SERVE_READY sticky until the rally starts or a score event
        # occurs; returning to READY here would allow repeated +0.10 rewards.

        # ---- u2: RALLY -------------------------------------------------
        {"from": 2, "true": ["player_game_progress"],
         "to": 3, "reward": 3.0},

        {"from": 2, "true": ["enemy_game_progress"],
         "to": 3, "reward": -3.0},

        {"from": 2, "true": ["player_point"],
         "to": 3, "reward": 1.0, "option": True},

        {"from": 2, "true": ["enemy_point"],
         "to": 3, "reward": -1.0},

        # ---- u3: POST_POINT --------------------------------------------
        # Wait until stale movement from the previous rally disappears.
        {"from": 3, "false": ["ball_in_play"],
         "to": 0, "reward": 0.0, "option": True},
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
        NUM_FEATURES = 29

        # ObjectObservation layout:
        # x, y, width, height, active, state, vis_id, orientation
        PLAYER_X = 0
        PLAYER_Y = 1

        BALL_X = 16
        BALL_Y = 17

        SERVE = 24

        PLAYER_POINTS = 25
        ENEMY_POINTS = 26
        PLAYER_GAME_SCORE = 27
        ENEMY_GAME_SCORE = 28

        # --------------------------------------------------------------
        # Ball-in-play detector
        # --------------------------------------------------------------
        EPS = 1e-6

        ball_x_f1 = obs[-3 * NUM_FEATURES + BALL_X]
        ball_y_f1 = obs[-3 * NUM_FEATURES + BALL_Y]

        ball_x_f2 = obs[-2 * NUM_FEATURES + BALL_X]
        ball_y_f2 = obs[-2 * NUM_FEATURES + BALL_Y]

        ball_x_f3 = obs[-NUM_FEATURES + BALL_X]
        ball_y_f3 = obs[-NUM_FEATURES + BALL_Y]

        m2 = (
            (jnp.abs(ball_x_f2 - ball_x_f1) > EPS)
            | (jnp.abs(ball_y_f2 - ball_y_f1) > EPS)
        )

        m3 = (
            (jnp.abs(ball_x_f3 - ball_x_f2) > EPS)
            | (jnp.abs(ball_y_f3 - ball_y_f2) > EPS)
        )

        ball_in_play = m2 & m3

        # --------------------------------------------------------------
        # Serve-preparation detector
        # --------------------------------------------------------------
        #
        # Horizontal distance alone can also match enemy-serve layouts,
        # so both horizontal and vertical alignment are required.
        #
        # Tennis constants:
        # FRAME_HEIGHT  = 210
        # PLAYER_HEIGHT = 23
        # hit tolerance = 3 pixels

        player_x_now = obs[-NUM_FEATURES + PLAYER_X]
        player_y_now = obs[-NUM_FEATURES + PLAYER_Y]

        ball_x_now = obs[-NUM_FEATURES + BALL_X]
        ball_y_now = obs[-NUM_FEATURES + BALL_Y]

        serve_now = obs[-NUM_FEATURES + SERVE] > 0.5

        X_THRESHOLD = 0.08
        PLAYER_HEIGHT_NORMALIZED = 23.0 / 210.0
        Y_THRESHOLD = 3.0 / 210.0

        x_aligned = (
            jnp.abs(player_x_now - ball_x_now)
            <= X_THRESHOLD
        )

        y_aligned = (
            jnp.abs(
                (
                    player_y_now
                    + PLAYER_HEIGHT_NORMALIZED
                )
                - ball_y_now
            )
            <= Y_THRESHOLD
        )

        serve_ready = (
            serve_now
            & x_aligned
            & y_aligned
        )

        # --------------------------------------------------------------
        # Score events
        # --------------------------------------------------------------
        player_points_now = obs[
            -NUM_FEATURES + PLAYER_POINTS
        ]
        enemy_points_now = obs[
            -NUM_FEATURES + ENEMY_POINTS
        ]

        player_game_now = obs[
            -NUM_FEATURES + PLAYER_GAME_SCORE
        ]
        enemy_game_now = obs[
            -NUM_FEATURES + ENEMY_GAME_SCORE
        ]

        player_points_prev = obs[
            -2 * NUM_FEATURES + PLAYER_POINTS
        ]
        enemy_points_prev = obs[
            -2 * NUM_FEATURES + ENEMY_POINTS
        ]

        player_game_prev = obs[
            -2 * NUM_FEATURES + PLAYER_GAME_SCORE
        ]
        enemy_game_prev = obs[
            -2 * NUM_FEATURES + ENEMY_GAME_SCORE
        ]

        player_game_progress = (
            player_game_now > player_game_prev
        )

        enemy_game_progress = (
            enemy_game_now > enemy_game_prev
        )

        player_point = (
            (player_points_now > player_points_prev)
            | player_game_progress
        )

        enemy_point = (
            (enemy_points_now > enemy_points_prev)
            | enemy_game_progress
        )

        return jnp.array([
            ball_in_play,
            serve_ready,
            player_point,
            enemy_point,
            player_game_progress,
            enemy_game_progress,
        ]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, obs):
        """Shaping potential: closeness of the player to the ball (last frame).

        Phi in [0, PHI_SCALE]; kept small relative to the +-1 / +-3 RM rewards.
        """
        NUM_FEATURES = 29
        now = obs[-NUM_FEATURES:]
        dist = jnp.hypot(now[0] - now[16], now[1] - now[17])
        return self.PHI_SCALE * (1.0 - jnp.clip(dist / self.PHI_MAX_DIST, 0.0, 1.0))
