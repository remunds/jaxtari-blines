import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM

class RewardMachine:
    # ToDo implement Baseclass GameRM
    def __init__(self, game_rm: GameRM):
        self.num_states: int    = game_rm.num_states()
        self.init_state: int    = game_rm.init_state()
        self.delta_u            = game_rm.delta_u()         # state-transition function
        self.delta_r            = game_rm.delta_r()         # reward-transition function
        self.terminal_state     = game_rm.terminal_state()  # only one final teminal state for all states
        self.get_events         = game_rm.get_events

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_next_state(self, current_state, game_obs):
        true_props = self.get_events(game_obs)
        prop_index = self.get_prop_index(true_props)
        next_state = self.delta_u[current_state, prop_index]
        return next_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_prop_index(self, true_props):
        powers = 2 ** jnp.arange(true_props.shape[-1])
        return jnp.sum(true_props * powers, axis=-1).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_reward( self, current_state, next_state):
        rew = self.delta_r[current_state,next_state]
        return rew

    @functools.partial(jax.jit, static_argnums=(0,))
    def step( self, current_state, game_obs):
        next_state = self.get_next_state(current_state, game_obs)
        done = (next_state == self.terminal_state)
        rew = self._get_reward(current_state,next_state)
        return next_state, rew, done
