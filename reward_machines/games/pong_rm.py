import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM


class PongRm(GameRM):
    def num_states(self) -> int:
        return 3

    def init_state(self) -> int:
        return 0

    def terminal_state(self) -> int:
        return -99

    def delta_u(self):
        # rows = current state (0,1,2)
        # cols = prop_index: 0=nothing, 1=scored, 2=conceded, 3=both(impossible)
        # column order MUST match bit-encoding: scored=bit0(+1), conceded=bit1(+2)
        return jnp.array([
            # nothing  scored  conceded  both
            [   0,       1,       2,      0 ],   # from u0
            [   0,       1,       2,      0 ],   # from u1
            [   0,       1,       2,      0 ],   # from u2
        ], dtype=jnp.int32)

    def delta_r(self):
        # reward on transition [from_state, to_state]; shape (3,3)
        # reaching u1 (scored) -> +1, reaching u2 (conceded) -> -1, back to u0 -> 0
        return jnp.array([
            # to:  u0     u1     u2
            [     0.0,   1.0,  -1.0 ],   # from u0
            [     0.0,   1.0,  -1.0 ],   # from u1
            [     0.0,   1.0,  -1.0 ],   # from u2
        ], dtype=jnp.float32)

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
