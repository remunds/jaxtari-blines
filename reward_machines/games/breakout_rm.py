import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions, field, field_slice

# Layout verified against jaxatari@c805a3d (BreakoutObservation, 126 per frame):
# player (x,y,w,h,active,visual_id,state,orientation)@0..7, ball @8..15,
# blocks@16+108, lives@124, score@125. x/width normalized by 160, y by 210.
NUM_FEATURES = 126
PLAYER_X, PLAYER_W = 0, 2
BALL_X, BALL_Y, BALL_W = 8, 9, 10
BLOCKS = 16
NUM_BLOCKS = 108
LIVES = 124


class BreakoutRm(GameRM):
    """v4: serve / rally machine.

    After a lost life JAXAtari freezes the ball until FIRE is pressed
    (`game_started` is reset to 0), but that flag is NOT in the observation
    (ball.active is always 1). The RM state carries it:

        u0 = SERVE   ball waiting, agent has to launch it
        u1 = RALLY   ball in play

        u0 --ball_launched--> u1   +0.1  (option)
        u1 --brick_hit------> u1   +0.2  (option)
        u1 --wall_cleared---> u1   +5.0  (option)
        u1 --lost_life------> u0   -1.0

    wall_cleared also detects the brick refill (the frame with 0 bricks left
    is not guaranteed to be observed).
    Shaping potential: bricks cleared + paddle/ball horizontal alignment.
    """

    PHI_BRICKS = 1.0
    PHI_ALIGN = 0.1
    MOVE_EPS = 1e-6

    PROP_INDEX = {"wall_cleared": 0, "brick_hit": 1, "lost_life": 2, "ball_launched": 3}

    # First matching row wins, so the most specific condition comes first.
    TRANSITIONS = [
        # ---- u0: SERVE ----
        {"from": 0, "true": ["ball_launched"], "false": ["lost_life"], "to": 1, "reward": 0.1, "option": True},
        {"from": 0, "true": ["lost_life"], "to": 0, "reward": -1.0},
        {"from": 0, "true": ["brick_hit"], "to": 1, "reward": 0.2},   # launch frame missed: re-sync

        # ---- u1: RALLY ----
        {"from": 1, "true": ["wall_cleared"], "to": 1, "reward": 5.0, "option": True},
        {"from": 1, "true": ["brick_hit"], "false": ["wall_cleared"], "to": 1, "reward": 0.2, "option": True},
        {"from": 1, "true": ["lost_life"], "false": ["brick_hit", "wall_cleared"], "to": 0, "reward": -1.0},
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

    def _ball_moved(self, obs, frames_ago):
        """Did the ball move between frame (frames_ago+1) and frame (frames_ago)?"""
        dx = field(obs, NUM_FEATURES, BALL_X, frames_ago=frames_ago) - field(obs, NUM_FEATURES, BALL_X, frames_ago=frames_ago + 1)
        dy = field(obs, NUM_FEATURES, BALL_Y, frames_ago=frames_ago) - field(obs, NUM_FEATURES, BALL_Y, frames_ago=frames_ago + 1)
        return (jnp.abs(dx) + jnp.abs(dy)) > self.MOVE_EPS

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, obs):
        lives_now  = field(obs, NUM_FEATURES, LIVES, frames_ago=0)
        lives_prev = field(obs, NUM_FEATURES, LIVES, frames_ago=1)

        remaining_now  = jnp.sum(field_slice(obs, NUM_FEATURES, BLOCKS, NUM_BLOCKS, frames_ago=0))
        remaining_prev = jnp.sum(field_slice(obs, NUM_FEATURES, BLOCKS, NUM_BLOCKS, frames_ago=1))

        brick_hit     = remaining_now < remaining_prev
        # Bricks never reappear inside a wall, so an increase = new wall.
        # (prev > 0 so a wall whose empty frame WAS observed is not counted twice)
        wall_cleared  = (remaining_prev > 0) & ((remaining_now == 0) | (remaining_now > remaining_prev))
        lost_life     = lives_now < lives_prev
        # Ball starts moving after being still (served).
        ball_launched = self._ball_moved(obs, 0) & ~self._ball_moved(obs, 1)

        return jnp.array([wall_cleared, brick_hit, lost_life, ball_launched]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, obs):
        """Phi = PHI_BRICKS * bricks cleared fraction
               + PHI_ALIGN * (1 - |paddle centre - ball centre|)."""
        remaining = jnp.sum(field_slice(obs, NUM_FEATURES, BLOCKS, NUM_BLOCKS, frames_ago=0))
        cleared = 1.0 - remaining / NUM_BLOCKS

        paddle_c = field(obs, NUM_FEATURES, PLAYER_X) + 0.5 * field(obs, NUM_FEATURES, PLAYER_W)
        ball_c = field(obs, NUM_FEATURES, BALL_X) + 0.5 * field(obs, NUM_FEATURES, BALL_W)
        align = 1.0 - jnp.clip(jnp.abs(paddle_c - ball_c), 0.0, 1.0)

        return self.PHI_BRICKS * cleared + self.PHI_ALIGN * align
