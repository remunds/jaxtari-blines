"""
Reward machine for JAXAtari Kangaroo.

This implementation uses only the normalized, flattened, frame-stacked
object-centric observation available to the generic RewardMachineWrapper.

The event definitions were derived from the Kangaroo environment mechanics
and independently validated with observation inspection scripts.
"""

import functools

import jax
import jax.numpy as jnp

from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class KangarooRm(GameRM):

    # ---------------------------------------------------------
    # Propositions / events
    # ---------------------------------------------------------

    PROP_INDEX = {
        "died": 0,
        "level_changed": 1,
        "on_floor_0": 2,
        "on_floor_1": 3,
        "on_floor_2": 4,
        "on_floor_3": 5,
        "fruit_collected": 6,
        "monkey_punched": 7,
    }

    # ---------------------------------------------------------
    # Reward constants
    # ---------------------------------------------------------

    PHI = [0.0, 1.0, 2.0, 3.0]

    DEATH_PENALTY = 1.0
    LEVEL_REWARD = 10.0
    FRUIT_REWARD = 0.5
    MONKEY_REWARD = 0.3

    # ---------------------------------------------------------
    # Reward Machine
    #
    # u0 = bottom floor
    # u1 = first upper floor
    # u2 = second upper floor
    # u3 = top floor
    #
    # Transition ordering gives priority because RewardMachine
    # selects the first matching transition.
    # ---------------------------------------------------------

    TRANSITIONS = [

        # =====================================================
        # State 0
        # =====================================================

        {
            "from": 0,
            "true": ["died"],
            "to": 0,
            "reward": -(PHI[0] + DEATH_PENALTY),
        },
        {
            "from": 0,
            "true": ["level_changed"],
            "to": 0,
            "reward": LEVEL_REWARD - PHI[0],
        },

        {
            "from": 0,
            "true": ["on_floor_1"],
            "to": 1,
            "reward": PHI[1] - PHI[0],
            "option": True,
        },
        {
            "from": 0,
            "true": ["on_floor_2"],
            "to": 2,
            "reward": PHI[2] - PHI[0],
        },
        {
            "from": 0,
            "true": ["on_floor_3"],
            "to": 3,
            "reward": PHI[3] - PHI[0],
        },

        {
            "from": 0,
            "true": ["fruit_collected"],
            "to": 0,
            "reward": FRUIT_REWARD,
        },
        {
            "from": 0,
            "true": ["monkey_punched"],
            "to": 0,
            "reward": MONKEY_REWARD,
            "option": True,
        },

        # =====================================================
        # State 1
        # =====================================================

        {
            "from": 1,
            "true": ["died"],
            "to": 0,
            "reward": -(PHI[1] + DEATH_PENALTY),
        },
        {
            "from": 1,
            "true": ["level_changed"],
            "to": 0,
            "reward": LEVEL_REWARD - PHI[1],
        },

        {
            "from": 1,
            "true": ["on_floor_0"],
            "to": 0,
            "reward": PHI[0] - PHI[1],
        },
        {
            "from": 1,
            "true": ["on_floor_2"],
            "to": 2,
            "reward": PHI[2] - PHI[1],
            "option": True,
        },
        {
            "from": 1,
            "true": ["on_floor_3"],
            "to": 3,
            "reward": PHI[3] - PHI[1],
        },

        {
            "from": 1,
            "true": ["fruit_collected"],
            "to": 1,
            "reward": FRUIT_REWARD,
            "option": True,
        },
        {
            "from": 1,
            "true": ["monkey_punched"],
            "to": 1,
            "reward": MONKEY_REWARD,
            "option": True,
        },

        # =====================================================
        # State 2
        # =====================================================

        {
            "from": 2,
            "true": ["died"],
            "to": 0,
            "reward": -(PHI[2] + DEATH_PENALTY),
        },
        {
            "from": 2,
            "true": ["level_changed"],
            "to": 0,
            "reward": LEVEL_REWARD - PHI[2],
        },

        {
            "from": 2,
            "true": ["on_floor_0"],
            "to": 0,
            "reward": PHI[0] - PHI[2],
        },
        {
            "from": 2,
            "true": ["on_floor_1"],
            "to": 1,
            "reward": PHI[1] - PHI[2],
        },
        {
            "from": 2,
            "true": ["on_floor_3"],
            "to": 3,
            "reward": PHI[3] - PHI[2],
            "option": True,
        },

        {
            "from": 2,
            "true": ["fruit_collected"],
            "to": 2,
            "reward": FRUIT_REWARD,
            "option": True,
        },
        {
            "from": 2,
            "true": ["monkey_punched"],
            "to": 2,
            "reward": MONKEY_REWARD,
            "option": True,
        },

        # =====================================================
        # State 3
        # =====================================================

        {
            "from": 3,
            "true": ["died"],
            "to": 0,
            "reward": -(PHI[3] + DEATH_PENALTY),
        },
        {
            "from": 3,
            "true": ["level_changed"],
            "to": 0,
            "reward": LEVEL_REWARD - PHI[3],
            "option": True,
        },

        {
            "from": 3,
            "true": ["on_floor_0"],
            "to": 0,
            "reward": PHI[0] - PHI[3],
        },
        {
            "from": 3,
            "true": ["on_floor_1"],
            "to": 1,
            "reward": PHI[1] - PHI[3],
        },
        {
            "from": 3,
            "true": ["on_floor_2"],
            "to": 2,
            "reward": PHI[2] - PHI[3],
        },

        {
            "from": 3,
            "true": ["fruit_collected"],
            "to": 3,
            "reward": FRUIT_REWARD,
        },
        {
            "from": 3,
            "true": ["monkey_punched"],
            "to": 3,
            "reward": MONKEY_REWARD,
        },
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

    # ---------------------------------------------------------
    # Event detector
    # ---------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, obs):

        NUM_FEATURES = 440

        PLAYER_Y = 1
        PLAYER_H = 3

        PLATFORM_Y = slice(28, 48)
        PLATFORM_ACTIVE = slice(88, 108)

        FRUIT_ACTIVE = slice(340, 343)

        MONKEY_STATE = slice(400, 404)

        Y_SCALE = 210.0
        STATE_SCALE = 255.0

        CRASH_Y = 186.0
        FLOOR_EDGES = jnp.array([28.5, 76.5, 124.5])

        # -----------------------------------------------------
        # Newest and previous object-centric frames
        # -----------------------------------------------------

        frames = obs.reshape(-1, NUM_FEATURES)

        prev = frames[-2]
        now = frames[-1]

        # -----------------------------------------------------
        # Player / grounded / floor
        # -----------------------------------------------------

        player_y_now = now[PLAYER_Y] * Y_SCALE
        player_h_now = now[PLAYER_H] * Y_SCALE
        feet_now = player_y_now + player_h_now

        player_y_prev = prev[PLAYER_Y] * Y_SCALE
        player_h_prev = prev[PLAYER_H] * Y_SCALE
        feet_prev = player_y_prev + player_h_prev

        platform_y = now[PLATFORM_Y] * Y_SCALE
        platform_active = now[PLATFORM_ACTIVE] > 0.5

        platform_match = (
            platform_active
            & (jnp.abs(platform_y - feet_now) <= 0.5)
        )

        stable = jnp.abs(feet_now - feet_prev) <= 0.5

        grounded = jnp.any(platform_match) & stable

        floor = (
            3
            - jnp.searchsorted(
                FLOOR_EDGES,
                feet_now,
                side="left",
            )
        )

        on_floor_0 = grounded & (floor == 0)
        on_floor_1 = grounded & (floor == 1)
        on_floor_2 = grounded & (floor == 2)
        on_floor_3 = grounded & (floor == 3)

        # -----------------------------------------------------
        # Death / respawn
        # -----------------------------------------------------

        died = (
            (player_y_prev >= CRASH_Y)
            & (player_y_now < CRASH_Y)
        )

        # -----------------------------------------------------
        # Level change
        # -----------------------------------------------------

        prev_platform_active = prev[PLATFORM_ACTIVE] > 0.5
        now_platform_active = now[PLATFORM_ACTIVE] > 0.5

        prev_platform_count = jnp.sum(prev_platform_active)
        now_platform_count = jnp.sum(now_platform_active)

        level_changed = prev_platform_count != now_platform_count

        # -----------------------------------------------------
        # Fruit collection
        # -----------------------------------------------------

        prev_fruit_active = prev[FRUIT_ACTIVE] > 0.5
        now_fruit_active = now[FRUIT_ACTIVE] > 0.5

        fruit_collected = jnp.any(
            prev_fruit_active & ~now_fruit_active
        )

        # -----------------------------------------------------
        # Monkey punch
        # -----------------------------------------------------

        prev_monkey_state = jnp.rint(
            prev[MONKEY_STATE] * STATE_SCALE
        ).astype(jnp.int32)

        now_monkey_state = jnp.rint(
            now[MONKEY_STATE] * STATE_SCALE
        ).astype(jnp.int32)

        monkey_candidate = jnp.any(
            (prev_monkey_state > 0)
            & (prev_monkey_state != 5)
            & (now_monkey_state == 0)
        )

        # Reset/death transitions can also remove monkeys.
        monkey_punched = (
            monkey_candidate
            & ~died
            & ~level_changed
        )

        return jnp.array(
            [
                died,
                level_changed,
                on_floor_0,
                on_floor_1,
                on_floor_2,
                on_floor_3,
                fruit_collected,
                monkey_punched,
            ],
            dtype=jnp.int32,
        )