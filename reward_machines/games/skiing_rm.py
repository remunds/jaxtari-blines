import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions, field, field_slice

# Layout from jaxatari@c805a3d SkiingObservation (73 per frame):
# skier@0(N=1) flags@8(N=2) trees@24(N=4) moguls@56(N=2) successful_gates@72
# Each flag entry is ONE gate: left pole at x, right pole at x + flag_distance.
# x is normalized by 160, y by 210.
NUM_FEATURES = 73
SKIER_X = 0
N_FLAGS = 2
FLAGS_X = 8                      # 8..9
FLAGS_Y = 8 + N_FLAGS            # 10..11
FLAGS_ACTIVE = 8 + 4 * N_FLAGS   # 16..17
SUCCESSFUL_GATES = 72

GATE_HALF_WIDTH = 16.0 / 160.0   # consts.flag_distance = 32 px
SKIER_Y = 46.0 / 210.0           # consts.skier_y

# NOTE: despite its name, `successful_gates` is a COUNTDOWN of gates remaining
# (starts at 20, -1 per passed gate; env reward = prev - now). See
# jaxatari/games/jax_skiing.py. So a gate is passed when it DECREASES.


class SkiingRm(GameRM):
    """v5: gate-streak machine + time pressure.

    ALE Skiing punishes every missed gate at the end of the run, so passing
    gates CONSECUTIVELY is what matters. How many gates in a row were passed
    is history, not visible in the observation; the RM tracks it:

        u0 = NO_STREAK, u1 = 1 in a row, u2 = 2+ in a row

        u0 --gate_passed--> u1   +1.0  (option)
        u1 --gate_passed--> u2   +1.2  (option)
        u2 --gate_passed--> u2   +1.5  (option)
        any --gate_missed-> u0   -0.5
        every other step         -0.01 (time cost, not an option)

    gate_missed: a gate scrolled past the skier line (same test as the env's
    own `crossed`) without the gate counter going down in that frame.
    Shaping potential: alignment of the skier with the next gate's centre.
    """

    PHI_SCALE = 0.2
    PASS_REWARD = (1.0, 1.2, 1.5)
    MISS_REWARD = -0.5
    # Time pressure. The env (ALE reward) charges time on every frame and
    # -500 per missed gate at the end; without a time cost the RM variants had
    # no reason to go fast. Skiing has no lives, so a step cost cannot make
    # "dying early" attractive. 0.5 miss ~ 50 steps, close to the env ratio.
    STEP_REWARD = -0.01

    PROP_INDEX = {"gate_passed": 0, "gate_missed": 1}

    TRANSITIONS = [
        {"from": 0, "true": ["gate_passed"], "to": 1, "reward": PASS_REWARD[0], "option": True},
        {"from": 0, "true": ["gate_missed"], "false": ["gate_passed"], "to": 0, "reward": MISS_REWARD},
        {"from": 0, "false": ["gate_passed", "gate_missed"], "to": 0, "reward": STEP_REWARD},
        {"from": 1, "true": ["gate_passed"], "to": 2, "reward": PASS_REWARD[1], "option": True},
        {"from": 1, "true": ["gate_missed"], "false": ["gate_passed"], "to": 0, "reward": MISS_REWARD},
        {"from": 1, "false": ["gate_passed", "gate_missed"], "to": 1, "reward": STEP_REWARD},
        {"from": 2, "true": ["gate_passed"], "to": 2, "reward": PASS_REWARD[2], "option": True},
        {"from": 2, "true": ["gate_missed"], "false": ["gate_passed"], "to": 0, "reward": MISS_REWARD},
        {"from": 2, "false": ["gate_passed", "gate_missed"], "to": 2, "reward": STEP_REWARD},
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

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, obs):
        gates_now  = field(obs, NUM_FEATURES, SUCCESSFUL_GATES, frames_ago=0)
        gates_prev = field(obs, NUM_FEATURES, SUCCESSFUL_GATES, frames_ago=1)
        gate_passed = gates_now < gates_prev  # counter counts DOWN, see note above

        fy_now  = field_slice(obs, NUM_FEATURES, FLAGS_Y, N_FLAGS, frames_ago=0)
        fy_prev = field_slice(obs, NUM_FEATURES, FLAGS_Y, N_FLAGS, frames_ago=1)
        crossed = jnp.any((fy_prev > SKIER_Y) & (fy_now <= SKIER_Y))
        gate_missed = crossed & ~gate_passed

        return jnp.array([gate_passed, gate_missed]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, obs):
        """Phi = PHI_SCALE * (1 - |skier_x - centre of next gate| / 0.5); 0 if no gate ahead."""
        skier_x = field(obs, NUM_FEATURES, SKIER_X)
        fx = field_slice(obs, NUM_FEATURES, FLAGS_X, N_FLAGS)
        fy = field_slice(obs, NUM_FEATURES, FLAGS_Y, N_FLAGS)
        active = field_slice(obs, NUM_FEATURES, FLAGS_ACTIVE, N_FLAGS) > 0.5

        ahead = active & (fy > SKIER_Y)                 # flags scroll up towards the skier
        next_idx = jnp.argmin(jnp.where(ahead, fy, jnp.inf))
        centre = fx[next_idx] + GATE_HALF_WIDTH
        align = 1.0 - jnp.clip(jnp.abs(skier_x - centre) / 0.5, 0.0, 1.0)
        return jnp.where(jnp.any(ahead), self.PHI_SCALE * align, 0.0)
