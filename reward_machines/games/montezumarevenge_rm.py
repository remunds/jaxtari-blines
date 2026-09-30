import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions, field, field_slice

# Layout from jaxatari@c805a3d MontezumaRevengeObservation (192 per frame).
# Every ObjectObservation block is 8 fields x N objects, grouped by field
# (x, y, width, height, active, visual_id, state, orientation):
# player@0(N=1) enemies@8(N=3) items@32(N=3) conveyors@56(N=1)
# doors@64(N=2) ropes@80(N=2) platforms@96(N=12)
# x is normalized by 160, y by 210.
NUM_FEATURES = 192
PLAYER_X, PLAYER_Y = 0, 1
ACTIVE_SUBOFFSET = 4

ITEMS = 32
N_ITEMS = 3
ITEMS_X = ITEMS                                    # 32..34
ITEMS_Y = ITEMS + N_ITEMS                          # 35..37
ITEMS_ACTIVE = ITEMS + ACTIVE_SUBOFFSET * N_ITEMS  # 44..46

DOORS = 64
N_DOORS = 2
DOORS_ACTIVE = DOORS + ACTIVE_SUBOFFSET * N_DOORS  # 72..73

# Static room geometry used to detect a room change:
# platforms x, y, width (3 fields x 12 platforms) = indices 96..131.
PLATFORM_GEOM = 96
PLATFORM_GEOM_LEN = 36


class MontezumaRm(GameRM):
    """v4: key -> door machine (Icarte et al. style).

    The inventory (keys, sword, ...) is NOT part of the observation, but a
    door only opens when the player carries a key. The RM state is exactly
    that missing memory:

        u0 = EMPTY_HANDED   looking for an item (key)
        u1 = CARRYING       has picked something up, should use it on a door

        u0 --item_collected--> u1   +1.0  (option)
        u1 --door_opened-----> u0   +2.0  (option)
        u1 --item_collected--> u1   +0.5  (option, e.g. second key / sword)
        every other frame           +0.01 survival bonus (not an option)

    Events are only counted inside the same room: items/doors are reloaded
    per room, so a room change would otherwise look like a pick-up / opening.
    Shaping potential: closeness of the player to the nearest active item.
    """

    PHI_SCALE = 0.2

    PROP_INDEX = {"item_collected": 0, "door_opened": 1}

    TRANSITIONS = [
        # ---- u0: EMPTY_HANDED ----
        {"from": 0, "true": ["item_collected"], "to": 1, "reward": 1.0, "option": True},
        # door without a tracked item (e.g. a key picked up earlier): pay, stay
        {"from": 0, "true": ["door_opened"], "false": ["item_collected"], "to": 0, "reward": 2.0},
        {"from": 0, "false": ["item_collected", "door_opened"], "to": 0, "reward": 0.01},

        # ---- u1: CARRYING ----
        {"from": 1, "true": ["door_opened"], "to": 0, "reward": 2.0, "option": True},
        {"from": 1, "true": ["item_collected"], "to": 1, "reward": 0.5, "option": True},  # door row above wins ties
        {"from": 1, "false": ["item_collected", "door_opened"], "to": 1, "reward": 0.01},
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
        geom_now  = field_slice(obs, NUM_FEATURES, PLATFORM_GEOM, PLATFORM_GEOM_LEN, frames_ago=0)
        geom_prev = field_slice(obs, NUM_FEATURES, PLATFORM_GEOM, PLATFORM_GEOM_LEN, frames_ago=1)
        same_room = jnp.all(jnp.abs(geom_now - geom_prev) < 1e-6)

        items_now  = jnp.sum(field_slice(obs, NUM_FEATURES, ITEMS_ACTIVE, N_ITEMS, frames_ago=0))
        items_prev = jnp.sum(field_slice(obs, NUM_FEATURES, ITEMS_ACTIVE, N_ITEMS, frames_ago=1))
        doors_now  = jnp.sum(field_slice(obs, NUM_FEATURES, DOORS_ACTIVE, N_DOORS, frames_ago=0))
        doors_prev = jnp.sum(field_slice(obs, NUM_FEATURES, DOORS_ACTIVE, N_DOORS, frames_ago=1))

        item_collected = same_room & (items_now < items_prev)
        door_opened    = same_room & (doors_now < doors_prev)

        return jnp.array([item_collected, door_opened]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, obs):
        """Phi = PHI_SCALE * (1 - distance to nearest active item); 0 if the room has none."""
        px = field(obs, NUM_FEATURES, PLAYER_X)
        py = field(obs, NUM_FEATURES, PLAYER_Y)
        ix = field_slice(obs, NUM_FEATURES, ITEMS_X, N_ITEMS)
        iy = field_slice(obs, NUM_FEATURES, ITEMS_Y, N_ITEMS)
        active = field_slice(obs, NUM_FEATURES, ITEMS_ACTIVE, N_ITEMS) > 0.5

        dist = jnp.where(active, jnp.hypot(ix - px, iy - py), jnp.inf)
        nearest = jnp.min(dist)
        return jnp.where(
            jnp.any(active),
            self.PHI_SCALE * (1.0 - jnp.clip(nearest, 0.0, 1.0)),
            0.0,
        )
