import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions, field, field_slice

# Layout from jaxatari@c805a3d GravitarObservation (194 per frame).
# Every ObjectObservation block is 8 fields x N objects, grouped by field:
# ship@0(N=1) enemies@8(N=4) fuel_tanks@40(N=4) saucer@72 ufo@80
# planets@88(N=7) projectiles@144(N=4) terrain@176 reactor_destination@184
# lives@192 fuel@193
NUM_FEATURES = 194
ENEMIES = 8;                N_ENEMIES = 4
FUEL_TANKS = 40;            N_FUEL_TANKS = 4
TERRAIN = 176
LIVES = 192

ACTIVE_SUBOFFSET = 4     # "active" is the 5th field of an ObjectObservation
VISUAL_SUBOFFSET = 5     # "visual_id" is the 6th field

ENEMY_ACTIVE = ENEMIES + ACTIVE_SUBOFFSET * N_ENEMIES          # 24..27
TANK_ACTIVE = FUEL_TANKS + ACTIVE_SUBOFFSET * N_FUEL_TANKS     # 56..59
TERRAIN_ACTIVE = TERRAIN + ACTIVE_SUBOFFSET                    # 180 (1 inside a planet level)
TERRAIN_VISUAL = TERRAIN + VISUAL_SUBOFFSET                    # 181 (which planet terrain)


class GravitarRm(GameRM):
    """v5: planet-visit machine (v4 + reward fixes, see TRANSITIONS).

    In JAXAtari a planet counts as cleared only when all enemies are gone AND
    the ship then flies out of the top of the level (jax_gravitar.py,
    `reset_level_win`). "Enemies already cleared, now leave" is not visible
    in the observation, so the RM tracks the phase:

        u0 = MAP       on the solar-system map
        u1 = IN_LEVEL  inside a planet level, enemies left
        u2 = CLEARED   all enemies of this level destroyed, must exit

        u0 --in_level---------> u1    0.0  (option: land on a planet)
        u1 --enemies_cleared--> u2   +2.0  (option)
        u1 --fuel / enemy-----> u1   +0.5 / +1.0 (options)
        u1 --!in_level--------> u0    0.0  (left without clearing)
        u2 --!in_level--------> u0   +3.0  (option: exit after clearing)
        u2 --fuel-------------> u2   +0.5  (option)
        any --lost_life-------> u0   -1.0

    `in_level` is a level (not edge) proposition, so the RM always re-syncs
    with where the ship actually is, even if a transition frame is missed.
    Kill / tank events only count inside the SAME level in both frames
    (objects vanish from the obs when leaving a level).
    """

    PHI_SCALE = 0.5

    PROP_INDEX = {
        "lost_life": 0,
        "fuel_tank_collected": 1,
        "enemy_destroyed": 2,
        "enemies_cleared": 3,
        "in_level": 4,
    }

    TRANSITIONS = [
        # No per-frame idle penalty (v4 had -0.01): with episodic_life a lost
        # life ends the episode, and -0.01/(1-gamma) = -1 equals the death
        # penalty, so surviving was worth no more than dying.
        # ---- u0: MAP ----
        {"from": 0, "true": ["lost_life"], "to": 0, "reward": -1.0},
        # 0 reward for landing: +0.2 could be farmed by entering and leaving.
        {"from": 0, "true": ["in_level"], "false": ["lost_life"], "to": 1, "reward": 0.0, "option": True},

        # ---- u1: IN_LEVEL ----
        {"from": 1, "true": ["lost_life"], "to": 0, "reward": -1.0},
        {"from": 1, "false": ["in_level", "lost_life"], "to": 0, "reward": 0.0},
        {"from": 1, "true": ["enemies_cleared"], "false": ["lost_life"], "to": 2, "reward": 2.0, "option": True},
        {"from": 1, "true": ["fuel_tank_collected"], "false": ["lost_life"], "to": 1, "reward": 0.5, "option": True},  # cleared row above wins ties
        {"from": 1, "true": ["enemy_destroyed"], "false": ["lost_life", "enemies_cleared", "fuel_tank_collected"], "to": 1, "reward": 1.0, "option": True},

        # ---- u2: CLEARED ----
        {"from": 2, "true": ["lost_life"], "to": 0, "reward": -1.0},
        {"from": 2, "false": ["in_level", "lost_life"], "to": 0, "reward": 3.0, "option": True},
        {"from": 2, "true": ["fuel_tank_collected"], "false": ["lost_life"], "to": 2, "reward": 0.5, "option": True},
    ]

    def __init__(self):
        (self._from, self._rt, self._rf, self._to, self._rew) = build_transitions(
            len(self.PROP_INDEX), self.PROP_INDEX, self.TRANSITIONS
        )

    def num_states(self):     return 3
    def init_state(self):     return 0
    def terminal_state(self): return -99
    def from_states(self):    return self._from
    def require_true(self):   return self._rt
    def require_false(self):  return self._rf
    def to_states(self):      return self._to
    def rewards(self):        return self._rew

    @staticmethod
    def _in_level(obs, frames_ago=0):
        return field(obs, NUM_FEATURES, TERRAIN_ACTIVE, frames_ago=frames_ago) > 0.5

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, obs):
        lives_now  = field(obs, NUM_FEATURES, LIVES, frames_ago=0)
        lives_prev = field(obs, NUM_FEATURES, LIVES, frames_ago=1)

        tanks_now    = jnp.sum(field_slice(obs, NUM_FEATURES, TANK_ACTIVE, N_FUEL_TANKS, frames_ago=0))
        tanks_prev   = jnp.sum(field_slice(obs, NUM_FEATURES, TANK_ACTIVE, N_FUEL_TANKS, frames_ago=1))
        enemies_now  = jnp.sum(field_slice(obs, NUM_FEATURES, ENEMY_ACTIVE, N_ENEMIES, frames_ago=0))
        enemies_prev = jnp.sum(field_slice(obs, NUM_FEATURES, ENEMY_ACTIVE, N_ENEMIES, frames_ago=1))

        in_level = self._in_level(obs, 0)
        same_level = (
            in_level
            & self._in_level(obs, 1)
            & (field(obs, NUM_FEATURES, TERRAIN_VISUAL, frames_ago=0)
               == field(obs, NUM_FEATURES, TERRAIN_VISUAL, frames_ago=1))
        )

        lost_life           = lives_now < lives_prev
        fuel_tank_collected = same_level & (tanks_now < tanks_prev)
        enemy_destroyed     = same_level & (enemies_now < enemies_prev)
        enemies_cleared     = same_level & (enemies_prev > 0) & (enemies_now == 0)

        return jnp.array([
            lost_life, fuel_tank_collected, enemy_destroyed, enemies_cleared, in_level,
        ]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, obs):
        """Inside a planet level: PHI_SCALE * mean(enemies gone, tanks gone). 0 on the map."""
        enemies_left = jnp.sum(field_slice(obs, NUM_FEATURES, ENEMY_ACTIVE, N_ENEMIES))
        tanks_left = jnp.sum(field_slice(obs, NUM_FEATURES, TANK_ACTIVE, N_FUEL_TANKS))
        progress = 0.5 * ((1.0 - enemies_left / N_ENEMIES) + (1.0 - tanks_left / N_FUEL_TANKS))
        return jnp.where(self._in_level(obs, 0), self.PHI_SCALE * progress, 0.0)
