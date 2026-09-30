import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions, field_slice

# Layout verified against jaxatari@c805a3d (PacmanObservation, 275 per frame):
# player_position@0+2 player_action@2 ghost_positions@3+8 ghost_actions@11+4
# fruit_position@15+2 fruit_action@17 fruit_type@18 pellets@19+252 power_pellets@271+4
NUM_FEATURES = 275
PELLETS = 19
NUM_PELLETS = 252
POWER_PELLETS = 271
NUM_POWER_PELLETS = 4


class MsPacmanRm(GameRM):
    """v4: power-pellet stage machine (same idea as Seaquest's diver count).

    u_k = number of power pellets eaten in the current level (k = 0..4).
    Power pellets are the strategic resource of a level (ghost modes are
    NOT in the observation), so the RM gives the agent an explicit stage:

        u_k --power_pellet_eaten--> u_{k+1}  +0.5  (option)
        u_k --pellet_eaten--------> u_k      +0.05 (option)
        u_k --level_cleared-------> u0       +5.0  (option, pellets refill)
        idle frames                          -0.01 (not an option)

    level_cleared also detects the pellet refill of a new level (the frame
    with 0 pellets left is not guaranteed to be observed).
    Shaping potential: fraction of pellets eaten in the current level.
    """

    PHI_SCALE = 1.0
    NUM_STAGES = NUM_POWER_PELLETS + 1

    PROP_INDEX = {"level_cleared": 0, "power_pellet_eaten": 1, "pellet_eaten": 2}

    TRANSITIONS = []
    for _k in range(NUM_POWER_PELLETS + 1):
        TRANSITIONS.append(
            {"from": _k, "true": ["level_cleared"], "to": 0, "reward": 5.0, "option": True})
        if _k < NUM_POWER_PELLETS:
            TRANSITIONS.append(
                {"from": _k, "true": ["power_pellet_eaten"], "false": ["level_cleared"],
                 "to": _k + 1, "reward": 0.5, "option": True})
        TRANSITIONS.append(
            {"from": _k, "true": ["pellet_eaten"], "false": ["power_pellet_eaten", "level_cleared"],
             "to": _k, "reward": 0.05, "option": True})
        TRANSITIONS.append(
            {"from": _k, "false": ["level_cleared", "power_pellet_eaten", "pellet_eaten"],
             "to": _k, "reward": -0.01})
    del _k

    def __init__(self):
        (self._from, self._rt, self._rf, self._to, self._rew) = build_transitions(
            len(self.PROP_INDEX), self.PROP_INDEX, self.TRANSITIONS
        )

    def num_states(self):     return self.NUM_STAGES
    def init_state(self):     return 0
    def terminal_state(self): return -99
    def from_states(self):    return self._from
    def require_true(self):   return self._rt
    def require_false(self):  return self._rf
    def to_states(self):      return self._to
    def rewards(self):        return self._rew

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, obs):
        pellets_left_now  = jnp.sum(field_slice(obs, NUM_FEATURES, PELLETS, NUM_PELLETS, frames_ago=0))
        pellets_left_prev = jnp.sum(field_slice(obs, NUM_FEATURES, PELLETS, NUM_PELLETS, frames_ago=1))
        power_now  = jnp.sum(field_slice(obs, NUM_FEATURES, POWER_PELLETS, NUM_POWER_PELLETS, frames_ago=0))
        power_prev = jnp.sum(field_slice(obs, NUM_FEATURES, POWER_PELLETS, NUM_POWER_PELLETS, frames_ago=1))

        pellet_eaten       = pellets_left_now < pellets_left_prev
        power_pellet_eaten = power_now < power_prev
        # Pellets never increase inside a level, so an increase = next level.
        # (prev > 0 so a level whose empty frame WAS observed is not counted twice)
        level_cleared = (pellets_left_prev > 0) & (
            (pellets_left_now == 0) | (pellets_left_now > pellets_left_prev)
        )

        return jnp.array([level_cleared, power_pellet_eaten, pellet_eaten]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, obs):
        """Phi = PHI_SCALE * fraction of pellet slots already eaten in this level."""
        pellets_now = field_slice(obs, NUM_FEATURES, PELLETS, NUM_PELLETS, frames_ago=0)
        return self.PHI_SCALE * (1.0 - jnp.sum(pellets_now) / NUM_PELLETS)
