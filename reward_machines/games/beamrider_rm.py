import functools

import jax
import jax.numpy as jnp

from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class BeamriderRm(GameRM):
    """
    Reward machine for JAXAtari Beamrider.

    RM states:
        u0 = CLEARING_UFOS
             Destroy the White UFOs in the current sector.

        u1 = MOTHERSHIP_PHASE
             All White UFOs are cleared; handle the mothership phase
             before advancing to the next sector.
    
    """

    # ---------------------------------------------------------
    # Observation layout
    # ---------------------------------------------------------

    NUM_FEATURES = 267

    WHITE_UFO_LEFT = 250
    LIVES = 251
    SECTOR = 252
    MOTHERSHIP_STAGE = 265

    # ---------------------------------------------------------
    # Propositions
    # ---------------------------------------------------------

    PROP_INDEX = {
        "lost_life": 0,
        "sector_advanced": 1,
        "all_ufos_cleared": 2,
        "ufo_destroyed": 3,
        "mothership_hit": 4,
    }

    # ---------------------------------------------------------
    # Reward Machine
    # ---------------------------------------------------------

    TRANSITIONS = [
        # =====================================================
        # u0 = CLEARING_UFOS
        # =====================================================

        # Death has highest priority.
        {
            "from": 0,
            "true": ["lost_life"],
            "to": 0,
            "reward": -1.0,
        },

        # The last UFO was destroyed.
        # This must appear before the generic ufo_destroyed
        # transition because both propositions fire together.
        {
            "from": 0,
            "true": ["all_ufos_cleared"],
            "false": ["lost_life"],
            "to": 1,
            "reward": 1.0,
        },

        # Normal UFO kill.
        {
            "from": 0,
            "true": ["ufo_destroyed"],
            "false": ["lost_life", "all_ufos_cleared"],
            "to": 0,
            "reward": 0.2,
        },

        # =====================================================
        # u1 = MOTHERSHIP_PHASE
        # =====================================================
        # Successful mothership hit.
        # Stay in u1 until the explosion finishes and the sector advances.
        {
            "from": 1,
            "true": ["mothership_hit"],
            "false": ["lost_life", "sector_advanced"],
            "to": 1,
            "reward": 2.0,
        },


        # Edge case: death and sector advancement can happen together
        {
            "from": 1,
            "true": ["lost_life", "sector_advanced"],
            "to": 0,
            "reward": 4.0,
        },

        {
            "from": 1,
            "true": ["sector_advanced"],
            "false": ["lost_life"],
            "to": 0,
            "reward": 5.0,
        },

        {
            "from": 1,
            "true": ["lost_life"],
            "false": ["sector_advanced"],
            "to": 1,
            "reward": -1.0,
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
        return 2

    def init_state(self):
        return 0

    def terminal_state(self):
        # Beamrider consists of repeating sectors.
        # Environment termination is handled by the environment.
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
    # Event detection
    # ---------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, obs):
        frames = obs.reshape(-1, self.NUM_FEATURES)

        prev = frames[-2]
        now = frames[-1]

        white_ufo_left_prev = prev[self.WHITE_UFO_LEFT]
        white_ufo_left_now = now[self.WHITE_UFO_LEFT]

        lives_prev = prev[self.LIVES]
        lives_now = now[self.LIVES]

        sector_prev = prev[self.SECTOR]
        sector_now = now[self.SECTOR]

        mothership_stage_prev = prev[self.MOTHERSHIP_STAGE]
        mothership_stage_now = now[self.MOTHERSHIP_STAGE]

        # Persistent counters make these events robust against frame skip.
        lost_life = lives_now < lives_prev

        sector_advanced = sector_now > sector_prev

        ufo_destroyed = white_ufo_left_now < white_ufo_left_prev

        # Fires only on the transition from >0 UFOs to zero UFOs.
        all_ufos_cleared = (
            (white_ufo_left_prev > 0.0)
            & (white_ufo_left_now <= 0.0)
        )
        # Stage 5 is the mothership explosion after a successful torpedo hit.
        # In the normalized observation, raw stage 5 corresponds to 1.0.
        # Detect only the entry into stage 5 so the reward fires once.

        mothership_hit = (
                (mothership_stage_prev < 0.9)
                & (mothership_stage_now > 0.9)
        )

        return jnp.array([
            lost_life,
            sector_advanced,
            all_ufos_cleared,
            ufo_destroyed,
            mothership_hit,
        ]).astype(jnp.int32)
