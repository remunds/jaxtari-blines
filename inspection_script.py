"""
Diagnostic for SeaquestRm events.

Two checks:
  1) FRAME SIZE: prints obs.shape and obs.shape[-1] // frame_stack. This MUST
     equal NUM_FEATURES (284). If it doesn't, every diff-based event
     (diver_collected, lost_life, scored, surfaced) reads the wrong previous
     frame and fires spuriously.
  2) EVENT TRACE: steps the env and prints the raw fields + which events fire,
     so you can compare against the video ("did a diver actually get collected
     when diver_collected fired?").

Run from the repo root so the wrappers import:
    uv run python inspect_seaquest_divers.py

IMPORTANT: the build_env() chain below must match make_env's object-centric
branch EXACTLY (same wrappers, same args), otherwise indices won't line up.
"""

import jax
import jax.numpy as jnp
import numpy as np
import jaxatari
from jaxatari.wrappers import (
    AtariWrapper,
    ObjectCentricWrapper,
    NormalizeObservationWrapper,
    FlattenObservationWrapper,
)

ENV_ID = "seaquest"
FRAME_STACK = 4
NUM_FEATURES = 284   # the value used in get_events — we will VERIFY this

# field offsets (negative = from end of the whole stack = newest frame)
OXY, SCORE, LIVES, DIVERS = -4, -3, -2, -1
PLAYER_Y = 1

# Seaquest actions (verified from environment.py)
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5


def build_env():
    """Must mirror the object-centric branch of make_env (NO RM wrapper)."""
    env = jaxatari.make(ENV_ID)
    env = AtariWrapper(
        env,
        sticky_actions=0.0,
        episodic_life=False,
        first_fire=True,
        noop_max=30,
        full_action_space=False,
    )
    env = ObjectCentricWrapper(env, frame_stack_size=FRAME_STACK, frame_skip=4, clip_reward=False)
    env = FlattenObservationWrapper(NormalizeObservationWrapper(env))
    return env


def check_frame_size(obs):
    print("=" * 70)
    print("CHECK 1: FRAME SIZE")
    print(f"  obs.shape            = {obs.shape}")
    total = obs.shape[-1]
    per_frame = total // FRAME_STACK
    print(f"  total features       = {total}")
    print(f"  frame_stack          = {FRAME_STACK}")
    print(f"  per-frame features   = {total} // {FRAME_STACK} = {per_frame}")
    print(f"  NUM_FEATURES (used)  = {NUM_FEATURES}")
    if per_frame == NUM_FEATURES:
        print("  -> OK: NUM_FEATURES matches the real frame size.")
    else:
        print(f"  -> !!! MISMATCH !!! Set NUM_FEATURES = {per_frame}")
        print("     (this alone breaks ALL diff-based events)")
    if total % FRAME_STACK != 0:
        print(f"  -> WARNING: total {total} not divisible by {FRAME_STACK} "
              f"— frame size may be non-integer / stacking differs.")
    print("=" * 70)


def extract(obs):
    return dict(
        divers_now    = float(obs[DIVERS]),
        divers_prev   = float(obs[DIVERS - NUM_FEATURES]),
        oxygen_now    = float(obs[OXY]),
        score_now     = float(obs[SCORE]),
        score_prev    = float(obs[SCORE - NUM_FEATURES]),
        lives_now     = float(obs[LIVES]),
        lives_prev    = float(obs[LIVES - NUM_FEATURES]),
        player_y_now  = float(obs[PLAYER_Y]),
        player_y_prev = float(obs[PLAYER_Y - NUM_FEATURES]),
    )


def main():
    env = build_env()
    obs, state = env.reset(jax.random.PRNGKey(0))
    obs = obs.squeeze()

    check_frame_size(obs)

    # Phase plan: dive down (toward divers), then move around, then surface.
    phases = (
        [("noop", NOOP)] * 3
        + [("down", DOWN)] * 60     # dive toward divers
        + [("left", LEFT)] * 20     # move to find a diver
        + [("right", RIGHT)] * 20
        + [("down", DOWN)] * 30
        + [("up", UP)] * 80         # surface
    )

    print("\nCHECK 2: EVENT TRACE")
    print("Watch divers_now: it should jump in clean 1/6 steps (0, .167, .333 ...)")
    print("and divers_prev should always equal the previous step's divers_now.\n")
    print(f"{'step':>4} {'act':>5} {'divers_now':>10} {'divers_prev':>11} "
          f"{'dCol':>4} {'py_now':>7} {'oxy':>6} {'score':>7} {'lives':>6}")
    print("-" * 78)

    key = jax.random.PRNGKey(0)
    for i in range(5000):
        key, k = jax.random.split(key)
        action = jax.random.randint(k, (), 0, env.action_space().n)
        obs, state, *_ = env.step(state, action)
        obs = obs.squeeze()
        v = extract(obs)
        if v["divers_now"] != v["divers_prev"]:
            print(f"step {i}: divers {v['divers_prev']:.4f} -> {v['divers_now']:.4f}  "
                  f"dCol={int(v['divers_now'] > v['divers_prev'])}")

    print("\n--- How to read ---")
    print("* If CHECK 1 said MISMATCH: fix NUM_FEATURES first, rerun.")
    print("* divers_now should be a multiple of 1/6 (~0.1667). If it shows other")
    print("  values, the DIVERS index points at the wrong field.")
    print("* diver_collected (dCol=1) should coincide with divers_now going UP by")
    print("  one 1/6 step. If dCol=1 appears while divers_now did NOT just step up,")
    print("  the previous-frame read (NUM_FEATURES) is wrong.")
    print("* Cross-check oxygen_now: full should be ~64/255 ≈ 0.251, draining over time.")


if __name__ == "__main__":
    main()
