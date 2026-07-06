import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class SeaquestRm(GameRM):
    
    PROP_INDEX = {
        "lost_life": 0,
        "oxygen_low": 1,
        "surfaced": 2,
        "diver_collected": 3,
        "scored": 4,
        "dove" : 5,
        "at_surface_idle": 6,
    }

    #   lost_life > diver_collected > scored > surfaced > at_surface_idle
    TRANSITIONS = [
        # State 0: 0 divers
        {"from": 0, "true": ["lost_life"], "to": 0, "reward": -0.3},                                # t0
        {"from": 0, "true": ["diver_collected"], "to": 1, "reward": 1.0},                           # t1
        {"from": 0, "true": ["scored"], "false": ["diver_collected", "surfaced"], "to": 0, "reward": 0.5},  # t2
        {"from": 0, "true": ["surfaced"], "false": ["diver_collected"], "to": 0, "reward": -0.5},   # t3
        {"from": 0, "true": ["at_surface_idle"], "false": ["lost_life", "diver_collected", "scored", "surfaced"], "to": 0, "reward": -0.02},  # t4

        # State 1: 1 diver
        {"from": 1, "true": ["lost_life"], "to": 0, "reward": -0.5},                                # t5
        {"from": 1, "true": ["diver_collected"], "to": 2, "reward": 1.0},                           # t6
        {"from": 1, "true": ["scored"], "false": ["diver_collected", "surfaced"], "to": 1, "reward": 0.5},  # t7
        {"from": 1, "true": ["surfaced"], "false": ["diver_collected"], "to": 0, "reward": 0.0},    # t8
        {"from": 1, "true": ["at_surface_idle"], "false": ["lost_life", "diver_collected", "scored", "surfaced"], "to": 1, "reward": -0.02},  # t9

        # State 2: 2 divers
        {"from": 2, "true": ["lost_life"], "to": 1, "reward": -0.7},                                # t10
        {"from": 2, "true": ["diver_collected"], "to": 3, "reward": 1.0},                           # t11
        {"from": 2, "true": ["scored"], "false": ["diver_collected", "surfaced"], "to": 2, "reward": 0.5},  # t12
        {"from": 2, "true": ["surfaced"], "false": ["diver_collected"], "to": 1, "reward": 0.0},    # t13
        {"from": 2, "true": ["at_surface_idle"], "false": ["lost_life", "diver_collected", "scored", "surfaced"], "to": 2, "reward": -0.02},  # t14

        # State 3: 3 divers
        {"from": 3, "true": ["lost_life"], "to": 2, "reward": -1.0},                                # t15
        {"from": 3, "true": ["diver_collected"], "to": 4, "reward": 1.0},                           # t16
        {"from": 3, "true": ["scored"], "false": ["diver_collected", "surfaced"], "to": 3, "reward": 0.5},  # t17
        {"from": 3, "true": ["surfaced"], "false": ["diver_collected"], "to": 2, "reward": 0.0},    # t18
        {"from": 3, "true": ["at_surface_idle"], "false": ["lost_life", "diver_collected", "scored", "surfaced"], "to": 3, "reward": -0.02},  # t19

        # State 4: 4 divers
        {"from": 4, "true": ["lost_life"], "to": 3, "reward": -1.3},                                # t20
        {"from": 4, "true": ["diver_collected"], "to": 5, "reward": 1.0},                           # t21
        {"from": 4, "true": ["scored"], "false": ["diver_collected", "surfaced"], "to": 4, "reward": 0.5},  # t22
        {"from": 4, "true": ["surfaced"], "false": ["diver_collected"], "to": 3, "reward": 0.0},    # t23
        {"from": 4, "true": ["at_surface_idle"], "false": ["lost_life", "diver_collected", "scored", "surfaced"], "to": 4, "reward": -0.02},  # t24

        # State 5: 5 divers
        {"from": 5, "true": ["lost_life"], "to": 4, "reward": -1.6},                                # t25
        {"from": 5, "true": ["diver_collected"], "to": 6, "reward": 1.0},                           # t26
        {"from": 5, "true": ["scored"], "false": ["diver_collected", "surfaced"], "to": 5, "reward": 0.5},  # t27
        {"from": 5, "true": ["surfaced"], "false": ["diver_collected"], "to": 4, "reward": 0.0},    # t28
        {"from": 5, "true": ["at_surface_idle"], "false": ["lost_life", "diver_collected", "scored", "surfaced"], "to": 5, "reward": -0.02},  # t29

        # State 6: 6 divers — goal.
        {"from": 6, "true": ["lost_life"], "to": 5, "reward": -2.0},                                # t30
        {"from": 6, "true": ["scored"], "false": ["surfaced"], "to": 6, "reward": 0.5},             # t31
        {"from": 6, "true": ["surfaced"], "to": 0, "reward": 10.0},                                 # t32
        {"from": 6, "true": ["at_surface_idle"], "false": ["lost_life", "scored", "surfaced"], "to": 6, "reward": -0.02},  # t33
    ]

    def __init__(self):
        (self._from, self._rt, self._rf, self._to, self._rew) = build_transitions(
            len(self.PROP_INDEX), self.PROP_INDEX, self.TRANSITIONS
        )

    def num_states(self):     return 7
    def init_state(self):     return 0
    def terminal_state(self): return -99

    def from_states(self):    return self._from
    def require_true(self):   return self._rt
    def require_false(self):  return self._rf
    def to_states(self):      return self._to
    def rewards(self):        return self._rew

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, obs):
        NUM_FEATURES = 284

        # --- field offsets (negative = from end of the whole stack = newest frame) ---
        OXY, SCORE, LIVES, DIVERS = -4, -3, -2, -1
        PLAYER_Y = 1

        # newest frame values:
        oxygen_now = obs[OXY]
        score_now  = obs[SCORE]
        lives_now  = obs[LIVES]
        divers_now = obs[DIVERS]
        player_y_now = obs[PLAYER_Y]

        # previous frame: one full frame (284) earlier
        score_prev      = obs[SCORE  - NUM_FEATURES]
        divers_prev     = obs[DIVERS - NUM_FEATURES]
        lives_prev      = obs[LIVES  - NUM_FEATURES]
        player_y_prev   = obs[PLAYER_Y - NUM_FEATURES]

        scored = score_now > score_prev           # score rose
        lost_life       = lives_now < lives_prev           # a life was lost
        oxygen_low      = oxygen_now < (12.0 / 255.0)       # ~16 oxygen units, tune this
        oxygen_full     = oxygen_now > (63.0 / 255.0)       # ~16 oxygen units, tune this
        surfaced        = (player_y_now < (46.5 / 210.0)) & (player_y_prev > player_y_now)    # player_y == 46 → at surface
        dove            = (player_y_prev <= (52 / 210.0)) & (player_y_prev < player_y_now) & oxygen_full
        diver_collected = divers_now > divers_prev
        # Fires every frame while sitting at the surface with full oxygen —
        # i.e. nothing left to refill, no reason to stay up. Lowest priority
        # event, used as a per-step penalty against the "stay at surface"
        # local optimum.
        at_surface_idle = (player_y_now <= (46.5 / 210.0)) & oxygen_full

        return jnp.array([
            lost_life,
            oxygen_low,
            surfaced,
            diver_collected,
            scored,
            dove,
            at_surface_idle,
        ]).astype(jnp.int32)
