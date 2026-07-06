import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class SeaquestRm(GameRM):
    """
    Phase-based Seaquest reward machine.

    DESIGN PRINCIPLE
    ----------------
    RM states are *non-observable mission phases*, NOT the diver count.
    The diver count is observable (`divers_now` in the obs), so it lives in a
    PROPOSITION (`has_6_divers`), never in a state. This avoids any desync
    between the RM state and the real game (e.g. surfacing costs a diver).

    Two phases:
      u0 = COLLECTING  -> gather divers, fight, manage oxygen
      u1 = FULL_LOAD   -> 6 divers collected, surface to deliver

    The phase changes WHAT surfacing means:
      - in COLLECTING, surfacing is just oxygen management (and costs a diver
        in the real game, which the agent feels via lost progress, not an RM
        penalty)
      - in FULL_LOAD, surfacing IS the goal: it delivers all 6 for a big bonus

    The transition COLLECTING -> FULL_LOAD uses both the previous RM state and
    the proposition has_6_divers, i.e. it is a proper delta_u(u, L) -> u'.
    """

    PROP_INDEX = {
        "diver_collected": 0,   # picked up a diver this step
        "has_6_divers":    1,   # carrying the full load (6)
        "lost_life":       2,   # died this step
        "surfaced":        3,   # reached the surface this step
        "oxygen_low":      4,   # oxygen is critically low
        "scored":          5,   # score rose (shooting enemies etc.)
    }

    TRANSITIONS = [
        # ---- u0 = COLLECTING ----------------------------------------------
        # collected the 6th diver -> switch to the deliver phase
        {"from": 0, "true": ["diver_collected", "has_6_divers"], "to": 1, "reward": 1.0},
        # collected a diver (1..5) -> stay collecting, reward the sub-goal
        {"from": 0, "true": ["diver_collected"],                 "to": 0, "reward": 1.0},
        # safety: already full but still in u0 (should not normally happen)
        {"from": 0, "true": ["has_6_divers"],                    "to": 1, "reward": 0.0},
        # died -> penalty
        {"from": 0, "true": ["lost_life"],                       "to": 0, "reward": 0.0},
        # shooting enemies / scoring (not from a diver pickup) -> small reward
        {"from": 0, "true": ["scored"], "false": ["diver_collected"], "to": 0, "reward": 0.1},
        # NOTE: surfacing in u0 has NO transition -> stays in u0, reward 0.
        # It is necessary for oxygen; the real game already "punishes" it by
        # costing a diver (has_6_divers stays false longer). No RM penalty.

        # ---- u1 = FULL_LOAD (carrying 6) ----------------------------------
        # surfaced with 6 divers -> deliver! big reward, back to collecting
        {"from": 1, "true": ["surfaced"],                        "to": 0, "reward": 5.0},
        # died while carrying 6 -> lost them, penalty
        {"from": 1, "true": ["lost_life"],                       "to": 0, "reward": -1.0},
        # keep fighting while heading up
        {"from": 1, "true": ["scored"], "false": ["surfaced"],   "to": 1, "reward": 0.1},
        # safety: somehow no longer full and not surfaced -> drop back to u0
        {"from": 1, "true": [], "false": ["has_6_divers", "surfaced"], "to": 0, "reward": 0.0},
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

        # field offsets (negative = from end of the stack = newest frame)
        OXY, SCORE, LIVES, DIVERS = -4, -3, -2, -1
        PLAYER_Y = 1

        # newest frame
        oxygen_now   = obs[OXY]
        score_now    = obs[SCORE]
        lives_now    = obs[LIVES]
        divers_now   = obs[DIVERS]
        player_y_now = obs[PLAYER_Y]

        # previous frame (one full frame = 284 features earlier)
        score_prev    = obs[SCORE    - NUM_FEATURES]
        divers_prev   = obs[DIVERS   - NUM_FEATURES]
        lives_prev    = obs[LIVES    - NUM_FEATURES]
        player_y_prev = obs[PLAYER_Y - NUM_FEATURES]

        # --- propositions (all read directly from the observation) ---------
        # divers normalized by MAX_COLLECTED_DIVERS = 6
        diver_collected = divers_now > divers_prev
        has_6_divers    = divers_now >= (5.5 / 6.0)

        lost_life       = lives_now < lives_prev

        # oxygen max is 64 but obs-space high is 255 -> full ~= 0.251.
        # Trigger "low" with a buffer so the agent has time to reach y==46.
        oxygen_low      = oxygen_now < (16.0 / 255.0)

        # surface is exactly player_y == 46 (verified from source); /210
        surfaced        = (player_y_now < (46.5 / 210.0)) & (player_y_prev > player_y_now)

        scored          = score_now > score_prev

        # order MUST match PROP_INDEX
        return jnp.array([
            diver_collected,
            has_6_divers,
            lost_life,
            surfaced,
            oxygen_low,
            scored,
        ]).astype(jnp.int32)
