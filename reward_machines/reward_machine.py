import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM

class RewardMachine:
    def __init__(self, game_rm: GameRM):
        self.num_states: int    = game_rm.num_states()
        self.init_state: int    = game_rm.init_state()
        self.from_states        = game_rm.from_states()
        self.require_true       = game_rm.require_true()
        self.require_false      = game_rm.require_false()
        self.to_states          = game_rm.to_states()
        self.rewards            = game_rm.rewards()
        self.terminal_state     = game_rm.terminal_state()  # only one final teminal state for all states
        self.get_events         = game_rm.get_events

    @functools.partial(jax.jit, static_argnums=(0,))
    def _match_transitions(self, current_state, true_props):
        true_ok = jnp.all(
            self.require_true * true_props == self.require_true, axis=1
        )
        false_ok = jnp.all(
            self.require_false * true_props == 0, axis=1
        )
        clause_ok = true_ok & false_ok

        # only transitions leaving the current state count
        from_ok = self.from_states == current_state
        return clause_ok & from_ok

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, current_state, game_obs):
        true_props = self.get_events(game_obs)
        valid = self._match_transitions(current_state, true_props)

        any_valid = jnp.any(valid)
        first_idx = jnp.argmax(valid)
        fired_idx = jnp.where(any_valid, first_idx, -1)

        next_state = jnp.where(any_valid, self.to_states[first_idx], current_state)
        rew        = jnp.where(any_valid, self.rewards[first_idx], 0.0)
        done       = (next_state == self.terminal_state)
        return next_state, rew, fired_idx, done
