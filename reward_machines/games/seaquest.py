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
        "has_one_diver": 4,
        "has_6_divers": 5,
    }

    # transitions: one state (u0), two events. order = priority.
    TRANSITIONS = [
        {"from": 0, "true": ["lost_life"], "to": 0, "reward": -1.0},
        {"from": 0, "true": ["diver_collected"], "false": ["has_6_divers"], "to": 0, "reward": 1.0},
        {"from": 0, "true": ["surfaced"], "to": 0, "reward": -0.5},
        {"from": 0, "true": ["has_6_divers"], "to": 1, "reward": 1.0},
        {"from": 0, "true": ["oxygen_low"], "to": 1, "reward": -0.5},
        {"from": 1, "true": ["lost_life"], "to": 0, "reward": -1.0},
        {"from": 1, "true": ["surfaced", "has_6_divers"], "to": 0, "reward": 1.0},
        {"from": 1, "true": ["surfaced"], "false": ["has_6_divers"], "to": 0, "reward": 0.5},
    ]

    def __init__(self):
        (self._from, self._rt, self._rf, self._to, self._rew) = build_transitions(
            len(self.PROP_INDEX), self.PROP_INDEX, self.TRANSITIONS
        )

    def num_states(self):     return 2
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

        # killed_enemy = score_now > score_prev           # score rose
        lost_life       = lives_now < lives_prev           # a life was lost
        has_6_divers    = divers_now >= (5.5 / 6.0)         # 6/6 = 1.0; use 5.5 for safety
        oxygen_low      = oxygen_now < (16.0 / 255.0)       # ~16 oxygen units, tune this
        surfaced        = (player_y_now <= (46.5 / 210.0)) & (player_y_prev > player_y_now)    # player_y == 46 → at surface
        has_one_diver   = (divers_now > (0.5 / 6.0)) & (divers_now < (1.5 / 6.0))
        diver_collected = divers_now > divers_prev 

        return jnp.array([
            # killed_enemy,
            lost_life,
            oxygen_low,
            surfaced,
            diver_collected,
            has_one_diver,
            has_6_divers
        ]).astype(jnp.int32)
