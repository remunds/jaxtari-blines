import functools
import numpy as np
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class PongRm(GameRM):
    PROP_INDEX = {"scored": 0, "conceded": 1}

    # transitions: one state (u0), two events. order = priority.
    TRANSITIONS = [
        {"from": 0, "true": ["scored"],   "to": 0, "reward":  1.0},
        {"from": 0, "true": ["conceded"], "to": 0, "reward": -1.0},
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
        SCORE_PLAYER_IDX    = 24
        SCORE_ENEMY_IDX     = 25
        NUM_OBS             = 26
        # obs: frame stack of shape (frame_stack, 26)
        # detect score changes between the two most recent frames
        scored   = obs[-(NUM_OBS - SCORE_PLAYER_IDX)] > obs[-(2 * NUM_OBS - SCORE_PLAYER_IDX)]
        conceded = obs[-(NUM_OBS - SCORE_ENEMY_IDX)]  > obs[-(2 * NUM_OBS - SCORE_ENEMY_IDX)]

        # boolean proposition vector [scored, conceded] -> bit-encoded by get_prop_index
        return jnp.array([scored, conceded]).astype(jnp.int32)
