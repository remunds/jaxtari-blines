"""
SCoBots — Successive Concept Bottleneck agents on JAXtari.

The policy sees neither pixels nor the raw object list, but *relational
concepts* computed from the objects ("distance from my paddle to the ball",
"where is the ball heading"). Every input dimension has a name, which is what
lets `agents/scobots/viper.py` distill a trained policy into a readable decision tree.

    JAXtari observation
      -> objects    GAME_OBJECTS: SCoBots object name -> JAXtari observation field
      -> concepts   FOCUS[game]:  which properties / relations feed the policy
      -> policy     separate actor and critic MLPs, trained with PPO

Credits
-------
* Delfosse, Blüml, Gregori, Kersting — "Interpretable Concept Bottlenecks to
  Align Reinforcement Learning Agents", NeurIPS 2024,
  https://github.com/k4ntz/SCoBots. The concept functions, the focus-file
  selections (the paper's pruned focus files for pong, tennis, seaquest,
  freeway, kangaroo and skiing), the action pruning, the concept rewards for
  pong, kangaroo and skiing (scobi/focus.py), and the policy architecture and
  hyperparameters (`config/alg/scobots_original.yaml`) follow that reference;
  this file is a from-scratch JAX re-implementation.
* The PPO trainer is `agents/ppo/ppo.py` of this repository, itself adapted from
  CleanRL's `ppo_atari_envpool_xla_jax_scan.py` (https://github.com/vwxyzjn/cleanrl).
* The focus selections for the other nine games and the concept rewards for
  tennis, freeway and mspacman are our own.

Usage
-----
    uv run python main.py +alg=scobots_tuned ENV_ID=pong
    uv run python main.py +alg=scobots_original ENV_ID=pong
    uv run python main.py +alg=scobots_tuned ENV_ID=pong alg.REWARD_MODE=2
    uv run python -m agents.scobots.scobots --describe pong   # the concept vector
    uv run python -m agents.scobots.scobots --selftest        # step all 15 games
"""

import os
import random
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, NamedTuple, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from rtpt import RTPT

import jaxatari
from jaxatari import spaces
from jaxatari.environment import JAXAtariAction
from jaxatari.wrappers import AtariState, AtariWrapper, JaxatariWrapper, LogWrapper

from agents.scobots.scobots_eval import evaluate


# ===========================================================================
# Objects: where each SCoBots object lives in the JAXtari observation
# ===========================================================================


class ObsField(NamedTuple):
    field: str          # attribute of the observation (or of Frame.extra)
    index: int | None   # row of a batched field, None for a single object


def _rows(name: str, field: str, n: int, start: int = 0) -> dict[str, ObsField]:
    """`Name1..Name{n}` -> rows `start..start+n-1` of a batched field."""
    return {f"{name}{i + 1}": ObsField(field, start + i) for i in range(n)}


GAME_OBJECTS: dict[str, dict[str, ObsField]] = {
    "asteroids": {"Player1": ObsField("player", None), **_rows("Missile", "missiles", 2),
                  **_rows("Asteroid", "asteroids", 5)},
    # The white saucers are the enemies a sector is cleared of.
    "beamrider": {"Player1": ObsField("player", None), **_rows("Enemy", "white_ufo", 2),
                  "Mothership1": ObsField("mothership", None)},
    "breakout": {"Player1": ObsField("player", None), "Ball1": ObsField("ball", None)},
    # Slot 0 is the car nearest to the player; empty slots have x = -1. The
    # player's car is not part of the observation (see EXTRA_OBJECTS).
    "enduro": {"Player1": ObsField("player", None), **_rows("Car", "enemy_positions", 4)},
    "freeway": {"Chicken1": ObsField("chicken", None), **_rows("Car", "car", 10)},
    "frostbite": {"Bailey1": ObsField("bailey", None), "Bear1": ObsField("bear", None),
                  **_rows("Obstacle", "obstacles", 4)},
    "gravitar": {"Player1": ObsField("ship", None), **_rows("Enemy", "enemies", 4),
                 **_rows("FuelTank", "fuel_tanks", 2)},
    "kangaroo": {"Player1": ObsField("player", None), **_rows("Monkey", "monkeys", 2),
                 **_rows("Fruit", "fruits", 3), "Bell1": ObsField("bell", None),
                 "Child1": ObsField("child", None), **_rows("Ladder", "ladders", 2),
                 **_rows("Platform", "platforms", 4),
                 "FallingCoconut1": ObsField("falling_coconut", None),
                 "ThrownCoconut1": ObsField("thrown_coconuts", 0)},
    "montezumarevenge": {"Player1": ObsField("player", None), **_rows("Enemy", "enemies", 3),
                         **_rows("Item", "items", 2), **_rows("Platform", "platforms", 2),
                         "Rope1": ObsField("ropes", 0), "Door1": ObsField("doors", 0)},
    "mspacman": {"Player1": ObsField("player_position", None),
                 **_rows("Ghost", "ghost_positions", 4)},
    "phoenix": {"Player1": ObsField("player", None), **_rows("Enemy", "enemies", 8),
                "Boss1": ObsField("boss", None)},
    "pong": {"Player1": ObsField("player", None), "Enemy1": ObsField("enemy", None),
             "Ball1": ObsField("ball", None)},
    # enemies = [12 sharks, 12 submarines, surface sub], projectiles = [player
    # missile, 4 enemy missiles]. OxygenBar1 comes from EXTRA_OBJECTS; Lives1
    # and CollectedDiver* are HUD counters JAXtari reports as numbers, unmapped.
    "seaquest": {"Player1": ObsField("player", None), "OxygenBar1": ObsField("oxygen_bar", None),
                 **_rows("Diver", "divers", 4),
                 **_rows("Shark", "enemies", 12), **_rows("Submarine", "enemies", 12, 12),
                 "SurfaceSubmarine1": ObsField("enemies", 24),
                 "PlayerMissile1": ObsField("projectiles", 0),
                 **_rows("EnemyMissile", "projectiles", 4, 1)},
    "skiing": {"Player1": ObsField("skier", None), **_rows("Flag", "flags", 2),
               **_rows("Mogul", "moguls", 2), **_rows("Tree", "trees", 2)},
    # JAXtari's tennis has no ball-shadow object, so BallShadow1 stays unmapped
    # (its concepts are constant) rather than aliased to the ball.
    "tennis": {"Player1": ObsField("player", None), "Enemy1": ObsField("enemy", None),
               "Ball1": ObsField("ball", None)},
}

# Objects SCoBots sees (OCAtari reads them from RAM) that a JAXtari observation
# does not give as positions, built from the observation or the game state:
# * enduro: the player's car is not in the observation; read it from the state.
# * seaquest: oxygen is a number (0-64) in the observation. The original agent
#   reads it through OCAtari's OxygenBarDepleted object, whose x is the right
#   edge of the filled bar; OxygenBar1 is given that position here. The bar is
#   drawn at (49, 170), 63 px wide when full, in both JAXtari and OCAtari.
EXTRA_OBJECTS: dict[str, Callable[[Any, Any], dict[str, jax.Array]]] = {
    "enduro": lambda obs, s: {"player": jnp.stack([s.player_x, s.player_y]).astype(jnp.float32)},
    "seaquest": lambda obs, s: {"oxygen_bar": jnp.stack(
        [49.0 + jnp.asarray(obs.oxygen_level, jnp.float32) * 63.0 / 64.0, jnp.float32(170.0)])},
}


@struct.dataclass
class Frame:
    obs: Any    # the JAXtari observation
    extra: Any  # {field: position} from EXTRA_OBJECTS


def _extract(frame: Frame, name: str, game: str) -> tuple[jax.Array, jax.Array]:
    """``(position[2], visible)`` of a SCoBots object; unmapped names are never visible."""
    field = GAME_OBJECTS[game].get(name)
    if field is None:
        return jnp.zeros(2, jnp.float32), jnp.bool_(False)
    obj = frame.extra[field.field] if field.field in frame.extra else getattr(frame.obs, field.field)

    if hasattr(obj, "x"):  # ObjectObservation, single or batched
        pos = jnp.stack([jnp.asarray(obj.x), jnp.asarray(obj.y)], -1).astype(jnp.float32)
        visible = jnp.asarray(obj.active, bool)
    else:                  # plain (x, y) array(s); JAXtari marks empty rows with x = -1
        pos = jnp.asarray(obj, jnp.float32)
        visible = pos[..., 0] >= 0
    if field.index is not None:
        pos, visible = pos[field.index], visible[field.index]
    return pos.reshape(2), visible.reshape(())


# ===========================================================================
# Concepts
#
# A property or function maps object positions to a few named outputs. A
# POSITION_HISTORY is [x, y, prev_x, prev_y] over one agent step, the layout
# SCoBots gets from OCAtari; the formulas below are SCoBots' scobi/concepts.py.
# `bounds` is each output's range in screen units, used to map it into [-1, 1].
# ===========================================================================

# An object moves a few pixels per step, so velocities are bounded by a quarter
# of the screen instead of all of it; larger jumps (respawns) saturate.
MAX_SPEED_FRAC = 0.25


@dataclass(frozen=True)
class Concept:
    inputs: tuple[str, ...]  # "POSITION" or "POSITION_HISTORY" per object argument
    labels: tuple[str, ...]  # one per output, "" for a scalar
    bounds: Callable[[float, float], tuple[list[float], list[float]]]
    fn: Callable[..., jax.Array]


def _linear_trajectory(pos, hist):
    """x/y distance from `pos` to the line through the other object's last two positions."""
    m = (hist[3] - hist[1]) / (hist[2] - hist[0] + 0.1)
    b = hist[1] - m * hist[0]
    dist_y = (m * pos[0] + b) - pos[1]
    dist_x = (pos[1] - b) / (m + jnp.finfo(jnp.float32).eps) - pos[0]
    return jnp.nan_to_num(jnp.stack([dist_x, dist_y]))  # +-inf saturates in the [-1, 1] map


def _speed(w, h):
    return MAX_SPEED_FRAC * float(np.hypot(w, h))


CONCEPTS: dict[str, Concept] = {
    # properties
    "POSITION": Concept(("POSITION",), ("x", "y"), lambda w, h: ([0, 0], [w, h]), lambda p: p),
    "POSITION_HISTORY": Concept(("POSITION_HISTORY",), ("x", "y", "prev_x", "prev_y"),
                                lambda w, h: ([0] * 4, [w, h, w, h]), lambda hist: hist),
    # functions
    "DISTANCE": Concept(("POSITION", "POSITION"), ("x", "y"),
                        lambda w, h: ([-w, -h], [w, h]), lambda a, b: b - a),
    "EUCLIDEAN_DISTANCE": Concept(("POSITION", "POSITION"), ("",),
                                  lambda w, h: ([0], [float(np.hypot(w, h))]),
                                  lambda a, b: jnp.linalg.norm(b - a)[None]),
    "CENTER": Concept(("POSITION", "POSITION"), ("x", "y"),
                      lambda w, h: ([0, 0], [w, h]), lambda a, b: (a + b) / 2),
    "VELOCITY": Concept(("POSITION_HISTORY",), ("",), lambda w, h: ([0], [_speed(w, h)]),
                        lambda hist: jnp.linalg.norm(hist[2:] - hist[:2])[None]),
    # prev - curr, as in SCoBots' get_dir_velocity
    "DIR_VELOCITY": Concept(("POSITION_HISTORY",), ("x", "y"),
                            lambda w, h: ([-MAX_SPEED_FRAC * w, -MAX_SPEED_FRAC * h],
                                          [MAX_SPEED_FRAC * w, MAX_SPEED_FRAC * h]),
                            lambda hist: hist[2:] - hist[:2]),
    "LINEAR_TRAJECTORY": Concept(("POSITION", "POSITION_HISTORY"), ("x", "y"),
                                 lambda w, h: ([-w, -h], [w, h]), _linear_trajectory),
}


# ===========================================================================
# Focus: the SELECTION block of each game's SCoBots focus file
#
# Properties come first, then functions, in the listed order. ORIENTATION and
# RGB properties are left out: JAXtari objects carry no colour, and their
# orientation encoding differs per game.
# ===========================================================================


class Focus(NamedTuple):
    features: tuple[tuple[str, tuple[str, ...]], ...]  # (concept, objects), in vector order
    actions: tuple[str, ...]                           # JAXAtariAction names the agent may use


def _focus(positions: str, histories: str, functions: str, actions: str) -> Focus:
    features = [("POSITION", (o,)) for o in positions.split()]
    features += [("POSITION_HISTORY", (o,)) for o in histories.split()]
    features += [(fn, tuple(a.strip() for a in args.split(",")))
                 for fn, args in re.findall(r"(\w+)\(([^)]*)\)", functions)]
    unknown = {fn for fn, _ in features} - set(CONCEPTS)
    assert not unknown, f"unknown concepts {unknown}"
    return Focus(tuple(features), tuple(actions.split()))


def _n(name: str, n: int) -> str:
    return " ".join(f"{name}{i + 1}" for i in range(n))


_KANGAROO = ("Player1 Monkey1 Monkey2 Fruit1 Fruit2 Bell1 Child1 Ladder1 Ladder2 "
             "Platform1 Platform2 FallingCoconut1 ThrownCoconut1").split()
_SEAQUEST = (f"Player1 Lives1 {_n('Diver', 4)} {_n('Submarine', 12)} SurfaceSubmarine1 "
             f"{_n('CollectedDiver', 6)} {_n('Shark', 12)} {_n('EnemyMissile', 4)} "
             "PlayerMissile1 OxygenBar1")
_ALL_DIRECTIONS = "UP RIGHT LEFT DOWN UPRIGHT UPLEFT DOWNRIGHT DOWNLEFT"

FOCUS: dict[str, Focus] = {
    "asteroids": _focus(
        "Player1 Asteroid1 Asteroid2 Asteroid3 Missile1", "Player1 Asteroid1 Asteroid2",
        "DISTANCE(Player1, Asteroid1) DISTANCE(Player1, Asteroid2) DISTANCE(Player1, Asteroid3) "
        "DISTANCE(Player1, Missile1) "
        "DIR_VELOCITY(Player1) DIR_VELOCITY(Asteroid1) DIR_VELOCITY(Asteroid2)",
        "NOOP FIRE UP RIGHT LEFT UPRIGHT UPLEFT UPFIRE RIGHTFIRE LEFTFIRE"),
    "beamrider": _focus(
        "Player1 Enemy1 Enemy2", "Player1 Enemy1 Enemy2",
        "DISTANCE(Player1, Enemy1) DISTANCE(Player1, Enemy2) EUCLIDEAN_DISTANCE(Player1, Enemy1) "
        "DIR_VELOCITY(Player1) DIR_VELOCITY(Enemy1)",
        "NOOP FIRE RIGHT LEFT RIGHTFIRE LEFTFIRE"),
    "breakout": _focus(
        "Player1 Ball1", "Player1 Ball1",
        "DISTANCE(Player1, Ball1) EUCLIDEAN_DISTANCE(Player1, Ball1) DIR_VELOCITY(Ball1) "
        "VELOCITY(Ball1) DIR_VELOCITY(Player1) LINEAR_TRAJECTORY(Player1, Ball1)",
        "NOOP FIRE RIGHT LEFT"),
    "enduro": _focus(
        "Player1 Car1 Car2 Car3 Car4", "Player1 Car1",
        "DISTANCE(Player1, Car1) DISTANCE(Player1, Car2) DISTANCE(Player1, Car3) "
        "DISTANCE(Player1, Car4) DIR_VELOCITY(Player1)",
        "NOOP FIRE RIGHT LEFT DOWN DOWNRIGHT DOWNLEFT RIGHTFIRE LEFTFIRE"),
    "freeway": _focus(
        "Chicken1 Car1 Car2 Car3 Car4", "Chicken1 Car1 Car2 Car3 Car4",
        "DISTANCE(Chicken1, Car1) DISTANCE(Chicken1, Car2) DISTANCE(Chicken1, Car3) "
        "DISTANCE(Chicken1, Car4) VELOCITY(Chicken1) VELOCITY(Car1) VELOCITY(Car2) "
        "VELOCITY(Car3) VELOCITY(Car4)",
        "NOOP UP DOWN"),
    "frostbite": _focus(
        "Bailey1 Bear1 Obstacle1 Obstacle2", "Bailey1 Obstacle1 Obstacle2",
        "DISTANCE(Bailey1, Bear1) DISTANCE(Bailey1, Obstacle1) DISTANCE(Bailey1, Obstacle2) "
        "DIR_VELOCITY(Bailey1) DIR_VELOCITY(Obstacle1) DIR_VELOCITY(Obstacle2)",
        f"NOOP {_ALL_DIRECTIONS}"),
    "gravitar": _focus(
        "Player1 Enemy1 Enemy2 FuelTank1", "Player1 Enemy1",
        "DISTANCE(Player1, Enemy1) DISTANCE(Player1, Enemy2) DISTANCE(Player1, FuelTank1) "
        "EUCLIDEAN_DISTANCE(Player1, FuelTank1) DIR_VELOCITY(Player1) DIR_VELOCITY(Enemy1)",
        "NOOP FIRE UP RIGHT LEFT UPRIGHT UPLEFT UPFIRE"),
    "kangaroo": _focus(
        " ".join(_KANGAROO), " ".join(_KANGAROO),
        # the player and both monkeys against every object listed after them
        " ".join(f"DISTANCE({a}, {b})" for i, a in enumerate(_KANGAROO[:3])
                 for b in _KANGAROO[i + 1:])
        + " " + " ".join(f"DIR_VELOCITY({o})" for o in _KANGAROO
                         if not o.startswith(("Ladder", "Platform"))),
        "NOOP FIRE UP RIGHT LEFT DOWN UPRIGHT UPLEFT"),
    "montezumarevenge": _focus(
        "Player1 Enemy1 Item1 Rope1 Platform1", "Player1 Enemy1",
        "DISTANCE(Player1, Enemy1) DISTANCE(Player1, Item1) DISTANCE(Player1, Rope1) "
        "EUCLIDEAN_DISTANCE(Player1, Enemy1) DIR_VELOCITY(Player1) DIR_VELOCITY(Enemy1)",
        f"NOOP FIRE {_ALL_DIRECTIONS}"),
    "mspacman": _focus(
        "Player1 Ghost1 Ghost2 Ghost3 Ghost4", "Player1 Ghost1",
        "DISTANCE(Player1, Ghost1) DISTANCE(Player1, Ghost2) DISTANCE(Player1, Ghost3) "
        "DISTANCE(Player1, Ghost4) EUCLIDEAN_DISTANCE(Player1, Ghost1) "
        "DIR_VELOCITY(Player1) DIR_VELOCITY(Ghost1)",
        f"NOOP {_ALL_DIRECTIONS}"),
    "phoenix": _focus(
        "Player1 Enemy1 Enemy2 Enemy3 Boss1", "Player1 Enemy1",
        "DISTANCE(Player1, Enemy1) DISTANCE(Player1, Enemy2) DISTANCE(Player1, Enemy3) "
        "DISTANCE(Player1, Boss1) EUCLIDEAN_DISTANCE(Player1, Enemy1) "
        "DIR_VELOCITY(Player1) DIR_VELOCITY(Enemy1)",
        "NOOP FIRE RIGHT LEFT DOWN RIGHTFIRE LEFTFIRE DOWNFIRE"),
    "pong": _focus(
        "Ball1 Player1", "Ball1 Player1",
        "LINEAR_TRAJECTORY(Player1, Ball1) DISTANCE(Player1, Ball1) "
        "EUCLIDEAN_DISTANCE(Player1, Ball1) CENTER(Player1, Ball1) VELOCITY(Player1) "
        "VELOCITY(Ball1) DIR_VELOCITY(Player1) DIR_VELOCITY(Ball1)",
        "NOOP FIRE RIGHT LEFT"),
    "seaquest": _focus(
        _SEAQUEST, _SEAQUEST, "",
        f"NOOP FIRE {_ALL_DIRECTIONS} UPFIRE RIGHTFIRE LEFTFIRE DOWNFIRE "
        "UPRIGHTFIRE UPLEFTFIRE DOWNRIGHTFIRE DOWNLEFTFIRE"),
    "skiing": _focus(
        "Player1 Mogul1 Flag1 Flag2 Tree1 Tree2", "Player1 Flag1",
        "DISTANCE(Player1, Flag1) CENTER(Flag1, Flag2) DIR_VELOCITY(Player1) DIR_VELOCITY(Flag1)",
        "NOOP RIGHT LEFT"),
    "tennis": _focus(
        "Player1 Enemy1 Ball1 BallShadow1", "Player1 Enemy1 Ball1 BallShadow1",
        "LINEAR_TRAJECTORY(Player1, Ball1) LINEAR_TRAJECTORY(Player1, BallShadow1) "
        "LINEAR_TRAJECTORY(Enemy1, Ball1) LINEAR_TRAJECTORY(Enemy1, BallShadow1) "
        "LINEAR_TRAJECTORY(Ball1, BallShadow1) "
        "DISTANCE(Player1, Enemy1) DISTANCE(Player1, Ball1) DISTANCE(Player1, BallShadow1) "
        "DISTANCE(Enemy1, Ball1) DISTANCE(Enemy1, BallShadow1) DISTANCE(Ball1, BallShadow1) "
        "VELOCITY(Player1) VELOCITY(Enemy1) VELOCITY(Ball1) VELOCITY(BallShadow1)",
        f"NOOP FIRE {_ALL_DIRECTIONS} UPFIRE RIGHTFIRE LEFTFIRE DOWNFIRE"),
}


def feature_layout(game: str, width: float, height: float):
    """Name and ``(low, high)`` screen-unit range of every concept-vector entry."""
    labels, low, high = [], [], []
    for kind, objs in FOCUS[game].features:
        concept = CONCEPTS[kind]
        name = objs[0] if kind == "POSITION" else f"{kind}({', '.join(objs)})"
        labels += [f"{name}.{label}" if label else name for label in concept.labels]
        lo, hi = concept.bounds(width, height)
        low += lo
        high += hi
    return labels, np.asarray(low, np.float32), np.asarray(high, np.float32)


def concept_vector(game: str, curr: Frame, prev: Frame) -> tuple[jax.Array, jax.Array]:
    """Raw concept values at step t and whether every object they read is visible."""
    values, visible = [], []
    for kind, objs in FOCUS[game].features:
        concept = CONCEPTS[kind]
        args, vis = [], jnp.bool_(True)
        for input_type, obj in zip(concept.inputs, objs):
            pos, v = _extract(curr, obj, game)
            if input_type == "POSITION_HISTORY":
                prev_pos, prev_v = _extract(prev, obj, game)
                pos, v = jnp.concatenate([pos, prev_pos]), v & prev_v
            args.append(pos)
            vis = vis & v
        out = concept.fn(*args)
        values.append(out)
        visible.append(jnp.broadcast_to(vis, out.shape))
    return jnp.concatenate(values).astype(jnp.float32), jnp.concatenate(visible)


# ===========================================================================
# Concept rewards
#
# Dense shaping in the same object vocabulary, for games whose env reward is
# too sparse to bootstrap from (freeway only pays for a full crossing):
#   REWARD_MODE 0: env reward   1: concept reward ("human")   2: both ("mixed")
# pong, kangaroo and skiing port SCoBots' reward functions (scobi/focus.py);
# tennis, freeway and mspacman are ours, and clip the jumps that point resets
# and respawns cause.
# ===========================================================================


@struct.dataclass
class RewardAux:
    """Fixed-shape carry for the stateful shaping terms."""

    kangaroo_best_y: jax.Array
    skiing_last_dist: jax.Array
    skiing_in_gate: jax.Array
    pacman_frightened_timer: jax.Array
    freeway_best_y: jax.Array
    initialized: jax.Array


def make_reward_aux() -> RewardAux:
    zero = jnp.float32(0.0)
    return RewardAux(zero, zero, jnp.bool_(False), zero, zero, jnp.bool_(False))


def _reward_pong(curr, prev, aux):
    """Close the vertical gap between the paddle and the ball (as SCoBots)."""
    player, vis_p = _extract(curr, "Player1", "pong")
    ball, vis_b = _extract(curr, "Ball1", "pong")
    player_prev, _ = _extract(prev, "Player1", "pong")
    ball_prev, _ = _extract(prev, "Ball1", "pong")
    delta = jnp.abs(player_prev[1] - ball_prev[1]) - jnp.abs(player[1] - ball[1])
    return jnp.where(vis_p & vis_b, 0.1 * delta, 0.0), aux


_TENNIS_NET_Y = 87.0  # court centre in player.y units; the defended side swaps between games


def _reward_tennis(curr, prev, aux):
    """Intercept the ball, win points, and mildly prefer standing near the net.

    Without shaping, never serving is a local optimum; the score term dominates
    the geometry terms so an endless rally never beats going for a winner.
    """
    player, vis_p = _extract(curr, "Player1", "tennis")
    ball, vis_b = _extract(curr, "Ball1", "tennis")
    player_prev, _ = _extract(prev, "Player1", "tennis")
    ball_prev, _ = _extract(prev, "Ball1", "tennis")

    # Clipped symmetrically: a point reset must not refund distance never closed.
    approach = jnp.clip(jnp.linalg.norm(player_prev - ball_prev) - jnp.linalg.norm(player - ball),
                        -15.0, 15.0)
    approach = jnp.where(vis_b, approach, 0.0)
    to_net = jnp.clip(jnp.abs(player_prev[1] - _TENNIS_NET_Y) - jnp.abs(player[1] - _TENNIS_NET_Y),
                      -15.0, 15.0)

    def gained(field):
        diff = jnp.asarray(getattr(curr.obs, field), jnp.float32) - getattr(prev.obs, field)
        return jnp.clip(diff, 0.0, 1.0)

    # The point that wins a game resets both point counters and bumps the game
    # counter instead ("*_sets" in the observation).
    won = jnp.maximum(gained("player_points"), gained("player_sets"))
    lost = jnp.maximum(gained("enemy_points"), gained("enemy_sets"))
    geometry = jnp.where(vis_p, 0.1 * approach + 0.02 * to_net, 0.0)
    return geometry + 2.0 * (won - lost), aux


def _reward_kangaroo(curr, prev, aux):
    """Pay for climbing to a new height and for closing in on the first ladder (as SCoBots).

    SCoBots drops only approach spikes above +100; dropping both signs keeps a
    ladder jumping on a level change from costing up to 500 in one step.
    """
    player, _ = _extract(curr, "Player1", "kangaroo")
    ladder, vis_l = _extract(curr, "Ladder1", "kangaroo")
    ladder_prev, vis_lp = _extract(prev, "Ladder1", "kangaroo")

    best_y = jnp.where(aux.initialized, aux.kangaroo_best_y, player[1])
    climb = jnp.maximum(best_y - player[1], 0.0)
    dist = jnp.where(vis_l, jnp.abs(player[0] - ladder[0]), 0.0)
    dist_prev = jnp.where(vis_lp, jnp.abs(player[0] - ladder_prev[0]), 0.0)
    approach = jnp.where(jnp.abs(dist_prev - dist) < 100.0, dist_prev - dist, 0.0)

    aux = aux.replace(kangaroo_best_y=jnp.minimum(best_y, player[1]), initialized=jnp.bool_(True))
    return climb + 5.0 * approach, aux


def _reward_skiing(curr, prev, aux):
    """Pay for speed, for approaching the next gate, and for passing through it (as SCoBots)."""
    player, _ = _extract(curr, "Player1", "skiing")
    flag1, _ = _extract(curr, "Flag1", "skiing")
    flag2, _ = _extract(curr, "Flag2", "skiing")
    flag1_prev, _ = _extract(prev, "Flag1", "skiing")

    gate = (flag1 + flag2) / 2.0
    dist = jnp.linalg.norm(player - gate)
    dist_prev = jnp.where(aux.initialized, aux.skiing_last_dist, dist)
    # Only credit approaching from above; below the gate it is already missed.
    approach = jnp.where((jnp.abs(dist_prev - dist) < 20.0) & (player[1] < gate[1]),
                         dist_prev - dist, 0.0)
    # The flags scroll past at the skier's speed, so their motion measures it.
    speed = jnp.clip(jnp.linalg.norm(flag1 - flag1_prev), 0.0, 10.0)
    in_gate = ((jnp.abs(player[0] - gate[0]) < 10) & (jnp.abs(player[1] - gate[1]) < 5))
    passed = jnp.where(in_gate & ~(aux.initialized & aux.skiing_in_gate), 100.0, 0.0)

    aux = aux.replace(skiing_last_dist=dist, skiing_in_gate=in_gate, initialized=jnp.bool_(True))
    return passed + speed + approach, aux


def _reward_freeway(curr, prev, aux):
    """Pay for new best progress upwards, and dock a little for being hit.

    Progress is a high-water mark, so being knocked back neither refunds reward
    nor is punished twice; the small collision penalty then favours dodging
    without making every crossing attempt net-negative. The mark resets when the
    chicken scores and reappears at the bottom (a knock-back moves it at most
    4 px per agent step, a new crossing ~170 px).
    """
    chicken, cars = curr.obs.chicken, curr.obs.car
    y = jnp.asarray(chicken.y, jnp.float32)
    crossed = y - jnp.asarray(prev.obs.chicken.y, jnp.float32) > 100.0
    best_y = jnp.where(aux.initialized & ~crossed, aux.freeway_best_y, y)
    progress = 0.1 * jnp.maximum(best_y - y, 0.0)

    x, w, h = (jnp.asarray(getattr(chicken, k), jnp.float32) for k in ("x", "width", "height"))
    cx, cy, cw, ch = (jnp.asarray(getattr(cars, k), jnp.float32) for k in ("x", "y", "width", "height"))
    hit = jnp.any(jnp.asarray(cars.active, bool) & (x < cx + cw) & (x + w > cx)
                  & (y < cy + ch) & (y + h > cy))

    aux = aux.replace(freeway_best_y=jnp.minimum(best_y, y), initialized=jnp.bool_(True))
    return progress + jnp.where(hit, -0.3, 0.0), aux


# JAXtari's MsPacmanConstants.FRIGHTENED_DURATION is 13 * 20 frames; the reward
# is computed once per agent step of 4 frames.
_PACMAN_FRIGHTENED_STEPS = 13 * 20 / 4


def _pacman_ghost_dist(obs):
    player = jnp.asarray(obs.player_position, jnp.float32)
    return jnp.min(jnp.linalg.norm(jnp.asarray(obs.ghost_positions, jnp.float32) - player, axis=-1))


def _pacman_pellet_dist(obs):
    """Distance to the nearest remaining pellet (+inf once the board is clear).

    Inverts the grid-cell -> pixel mapping of JAXtari's MsPacmanRenderer
    (18x14 grid, split at x = 76 for the tunnel gap).
    """
    gx = jnp.arange(obs.pellets.shape[0], dtype=jnp.float32)[:, None]
    gy = jnp.arange(obs.pellets.shape[1], dtype=jnp.float32)[None, :]
    px = jnp.where(gx < 9, gx * 8 + 10, gx * 8 + 14)
    py = gy * 12 + 9
    player = jnp.asarray(obs.player_position, jnp.float32)
    dists = jnp.sqrt((px - player[0]) ** 2 + (py - player[1]) ** 2)
    return jnp.min(jnp.where(jnp.asarray(obs.pellets, bool), dists, jnp.inf))


def _reward_pacman(curr, prev, aux):
    """Flee the nearest ghost, chase it while powered up, and keep eating.

    The observation has no frightened flag, so it is approximated: a power
    pellet disappearing starts a _PACMAN_FRIGHTENED_STEPS window.
    """
    curr, prev = curr.obs, prev.obs
    ghost_delta = _pacman_ghost_dist(curr) - _pacman_ghost_dist(prev)
    ghost_delta = jnp.where(jnp.abs(ghost_delta) < 20.0, ghost_delta, 0.0)  # tunnel / jail jumps

    ate_power = jnp.any(jnp.asarray(prev.power_pellets, bool) & ~jnp.asarray(curr.power_pellets, bool))
    timer = jnp.where(ate_power, _PACMAN_FRIGHTENED_STEPS,
                      jnp.maximum(jnp.where(aux.initialized, aux.pacman_frightened_timer, 0.0) - 1.0, 0.0))
    ghost_reward = 0.02 * jnp.where(timer > 0.0, -ghost_delta, ghost_delta)

    pellet_delta = _pacman_pellet_dist(prev) - _pacman_pellet_dist(curr)
    valid = jnp.isfinite(pellet_delta) & (jnp.abs(pellet_delta) < 30.0)  # level refill / empty board
    pellet_reward = 0.02 * jnp.where(valid, pellet_delta, 0.0)

    aux = aux.replace(pacman_frightened_timer=timer, initialized=jnp.bool_(True))
    return ghost_reward + pellet_reward, aux


CONCEPT_REWARDS: dict[str, Callable] = {
    "pong": _reward_pong,
    "tennis": _reward_tennis,
    "kangaroo": _reward_kangaroo,
    "skiing": _reward_skiing,
    "freeway": _reward_freeway,
    "mspacman": _reward_pacman,
}


# ===========================================================================
# The wrapper: a sibling of jaxatari's ObjectCentricWrapper
# ===========================================================================


@struct.dataclass
class ScobotState:
    atari_state: AtariState
    prev: Frame       # the previous agent step, for histories and shaping
    aux: RewardAux


class ScobotWrapper(JaxatariWrapper):
    """Emits the SCoBots concept vector instead of the flattened object state.

    Apply directly after `AtariWrapper`. Frame skipping, reward clipping and
    autoreset follow `ObjectCentricWrapper`. Features are mapped to [-1, 1] with
    the analytic bounds of `feature_layout`, so training and evaluation see the
    same mapping (like `NormalizeObservationWrapper`, unlike SCoBots' running
    VecNormalize). Entries whose objects are absent are 0 before that mapping,
    as in SCoBots. The agent's actions are the focus file's selection.
    """

    def __init__(self, env: AtariWrapper, game: str, frame_skip: int = 4,
                 clip_reward: bool = True, reward_mode: int = 0):
        super().__init__(env)
        if game not in FOCUS:
            raise ValueError(f"no SCoBots focus for '{game}'; known: {sorted(FOCUS)}")
        if reward_mode not in (0, 1, 2) or (reward_mode and game not in CONCEPT_REWARDS):
            raise ValueError(f"REWARD_MODE={reward_mode} is not available for '{game}' "
                             f"(concept rewards exist for {sorted(CONCEPT_REWARDS)})")
        self.game, self.frame_skip = game, frame_skip
        self.clip_reward, self.reward_mode = clip_reward, reward_mode

        # The env's action i stands for ACTION_SET[i] (action_set[i] in some
        # ports, action constant i in gravitar).
        table = getattr(env, "ACTION_SET", getattr(env, "action_set", None))
        table = [int(a) for a in (range(env.action_space().n) if table is None else np.asarray(table))]

        # A modification may remove actions (pong's no_fire): the policy keeps
        # its action space, and a missing action falls back to its non-firing
        # version, i.e. the button is pressed but does nothing.
        def index(name):
            for candidate in (name, name.replace("FIRE", "") or "NOOP", "NOOP"):
                if int(getattr(JAXAtariAction, candidate)) in table:
                    return table.index(int(getattr(JAXAtariAction, candidate)))
            raise ValueError(f"{game}: no action in {table} can stand in for {name}")

        self.action_names = FOCUS[game].actions
        self._actions = jnp.asarray([index(a) for a in self.action_names], jnp.int32)

        height, width = env.image_space().shape[:2]
        self.labels, low, high = feature_layout(game, float(width), float(height))
        self.n_features = len(self.labels)
        self._low = jnp.asarray(low)
        self._scale = jnp.asarray(2.0 / np.maximum(high - low, 1e-6))

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(self.n_features,), dtype=jnp.float32)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_names))

    def _frame(self, obs, atari_state: AtariState) -> Frame:
        extra = EXTRA_OBJECTS[self.game](obs, atari_state.env_state) if self.game in EXTRA_OBJECTS else {}
        return Frame(obs, extra)

    def _features(self, curr: Frame, prev: Frame) -> jax.Array:
        values, visible = concept_vector(self.game, curr, prev)
        values = jnp.where(visible, values, 0.0)
        return jnp.clip((values - self._low) * self._scale - 1.0, -1.0, 1.0)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array) -> tuple[jax.Array, ScobotState]:
        obs, atari_state = self._env.reset(key)
        frame = self._frame(obs, atari_state)
        # No history at t=0: prev == curr, so velocities start at zero.
        return self._features(frame, frame), ScobotState(atari_state, frame, make_reward_aux())

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: ScobotState, action: int):
        def body_fn(atari_state, _):
            obs, atari_state, reward, terminated, truncated, info = self._env.step(
                atari_state, self._actions[action])
            return atari_state, (obs, reward, terminated, truncated, info)

        atari_state, (obs, rewards, terminations, truncations, infos) = jax.lax.scan(
            body_fn, state.atari_state, None, length=self.frame_skip)

        frame = self._frame(jax.tree.map(lambda x: x[-1], obs), atari_state)
        features = self._features(frame, state.prev)

        reward = jnp.sum(rewards)
        if self.clip_reward:
            reward = jnp.sign(reward)
        aux = state.aux
        if self.reward_mode != 0:
            # Added after clipping: signing it would flatten the dense signal to +-1.
            concept_reward, aux = CONCEPT_REWARDS[self.game](frame, state.prev, aux)
            reward = concept_reward if self.reward_mode == 1 else reward + concept_reward

        terminated, truncated = terminations.any(), truncations.any()
        # Autoreset (gym's SAME_STEP mode), with fresh history and shaping state.
        features, state = jax.lax.cond(
            jnp.logical_or(infos["env_done"].any(), truncated),
            lambda: self.reset(atari_state.key),
            lambda: (features, ScobotState(atari_state, frame, aux)),
        )

        def reduce_info(k, v):
            if k in ("env_reward", "all_rewards"):
                return jnp.sum(v, axis=0)
            return jnp.any(v) if k == "env_done" else v[-1]

        info = {k: reduce_info(k, v) for k, v in infos.items()}
        return features, state, reward, terminated, truncated, info


def make_env(env_id, mods=(), reward_mode=0, eval=False):
    mods = list(mods) or None
    if not eval and mods:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        env = jaxatari.make(env_id, mods=mods)
        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=not eval,  # only active during training
            first_fire=True,
            noop_max=30,
            full_action_space=False,
        )
        # Evaluation scores the game itself: unclipped, unshaped.
        env = ScobotWrapper(env, game=env_id, clip_reward=not eval,
                            reward_mode=0 if eval else reward_mode)
        return LogWrapper(env)

    return thunk


# ===========================================================================
# Policy: SCoBots' MlpPolicy, i.e. separate actor and critic MLPs
# (SB3 net_arch pi=[64, 64], vf=[64, 64], ReLU, orthogonal init)
# ===========================================================================


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    out_scale: float  # orthogonal gain of the output layer

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        return nn.Dense(self.out_dim, kernel_init=orthogonal(self.out_scale),
                        bias_init=constant(0.0))(x)


def make_actor_critic(hidden_dims: Sequence[int], n_actions: int) -> tuple[MLP, MLP]:
    return MLP(tuple(hidden_dims), n_actions, 0.01), MLP(tuple(hidden_dims), 1, 1.0)


class AgentParams(NamedTuple):
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class Storage:
    obs: jnp.ndarray
    actions: jnp.ndarray
    logprobs: jnp.ndarray
    dones: jnp.ndarray
    values: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    rewards: jnp.ndarray


# ===========================================================================
# PPO: agents/ppo/ppo.py, with separate actor/critic and optional clip annealing
# ===========================================================================


def single_run(config):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    config["TRAIN_MODS"] = tuple(config["TRAIN_MODS"])
    config["EVAL_MODS"] = tuple(config["EVAL_MODS"])
    config["BATCH_SIZE"] = int(config["NUM_ENVS"] * config["NUM_STEPS"])
    config["MINIBATCH_SIZE"] = int(config["BATCH_SIZE"] // config["NUM_MINIBATCHES"])
    config["NUM_ITERATIONS"] = int(config["TOTAL_TIMESTEPS"] // config["BATCH_SIZE"])
    reward_mode = int(config["REWARD_MODE"])

    run_name = f'{config["ENV_ID"]}_{config["EXP_NAME"]}_mode{reward_mode}_{config["SEED"]}'
    wandb.init(project=config["PROJECT"], entity=config["ENTITY"], config=config,
               name=run_name, save_code=True)

    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])
    key, actor_key, critic_key = jax.random.split(key, 3)

    env_fn = partial(make_env, reward_mode=reward_mode)
    env = env_fn(config["ENV_ID"], mods=config["TRAIN_MODS"])()
    n_actions = env.action_space().n
    print(f'SCoBots on {config["ENV_ID"]}: {env.n_features} concepts, '
          f'{n_actions} actions {env.action_names}, reward mode {reward_mode}')

    @jax.jit
    def vmap_reset(keys):
        return jax.vmap(env.reset)(keys)

    @jax.jit
    def vmap_step(state, action):
        obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        return obs, state, reward, jnp.logical_or(terminated, truncated), info

    def anneal(count):  # 1 -> 1/NUM_ITERATIONS, one step per iteration
        return 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_ITERATIONS"]

    actor, critic = make_actor_critic(config["HIDDEN_DIMS"], n_actions)
    dummy_obs = jnp.zeros((1, env.n_features))
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(actor.init(actor_key, dummy_obs), critic.init(critic_key, dummy_obs)),
        tx=optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=(lambda c: config["LEARNING_RATE"] * anneal(c))
                if config["ANNEAL_LR"] else config["LEARNING_RATE"],
                eps=1e-5,
            ),
        ),
    )

    @jax.jit
    def get_action_and_value(agent_state: TrainState, next_obs, key):
        logits = actor.apply(agent_state.params.actor_params, next_obs)
        # sample action: Gumbel-max trick
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = critic.apply(agent_state.params.critic_params, next_obs)
        return action, logprob, value.squeeze(1), key

    def get_action_and_value2(params: AgentParams, x, action):
        logits = actor.apply(params.actor_params, x)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        entropy = -(logits * jax.nn.softmax(logits)).sum(-1)
        return logprob, entropy, critic.apply(params.critic_params, x).squeeze(-1)

    def compute_gae_once(advantages, inp):
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone
        delta = reward + config["GAMMA"] * nextvalues * nextnonterminal - curvalues
        advantages = delta + config["GAMMA"] * config["GAE_LAMBDA"] * nextnonterminal * advantages
        return advantages, advantages

    @jax.jit
    def compute_gae(agent_state: TrainState, next_obs, next_done, storage: Storage):
        next_value = critic.apply(agent_state.params.critic_params, next_obs).squeeze(-1)
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once, jnp.zeros((config["NUM_ENVS"],)),
            (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True)
        return storage.replace(advantages=advantages, returns=advantages + storage.values)

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, clip_coef):
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if config["NORM_ADV"]:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        pg_loss = jnp.maximum(-mb_advantages * ratio,
                              -mb_advantages * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)).mean()
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
        entropy_loss = entropy.mean()
        loss = pg_loss - config["ENT_COEF"] * entropy_loss + v_loss * config["VF_COEF"]
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(agent_state: TrainState, storage: Storage, key, clip_coef):
        def update_epoch(carry, _):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x.reshape((-1,) + x.shape[2:]))
                return jnp.reshape(x, (config["NUM_MINIBATCHES"], -1) + x.shape[1:])

            def update_minibatch(agent_state, mb):
                (loss, aux), grads = ppo_loss_grad_fn(
                    agent_state.params, mb.obs, mb.actions, mb.logprobs, mb.advantages,
                    mb.returns, clip_coef)
                return agent_state.apply_gradients(grads=grads), (loss, *aux)

            agent_state, losses = jax.lax.scan(
                update_minibatch, agent_state, jax.tree.map(convert_data, storage))
            return (agent_state, key), losses

        (agent_state, key), losses = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=config["UPDATE_EPOCHS"])
        return agent_state, losses, key

    def save_and_eval(iteration):
        # evaluate() loads from disk, so without SAVE_PATH the model goes to a temp dir.
        save_dir = (os.path.join(config["SAVE_PATH"], run_name) if config["SAVE_PATH"]
                    else tempfile.mkdtemp())
        model_path = os.path.join(save_dir, f'{config["EXP_NAME"]}_{iteration}_{time.time()}.model')
        os.makedirs(save_dir, exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(
                [config, [agent_state.params.actor_params, agent_state.params.critic_params]]))
        if config["SAVE_PATH"]:
            print(f"model saved to {model_path}")

        # evaluate across all mods (and default train env)
        eval_configs = [((), "default")]
        for mod in config["EVAL_MODS"] or config["TRAIN_MODS"]:
            mods = (mod,) if isinstance(mod, str) else tuple(mod)
            eval_configs.append((mods, "_".join(mods)))

        metrics = {}
        for mods, label in eval_configs:
            print(f"Evaluating on {label} ...")
            episodic_returns, env_states = evaluate(
                model_path, partial(env_fn, mods=mods, eval=True), config["ENV_ID"],
                eval_episodes=10, Model=MLP, seed=config["SEED"] + 42)
            metrics[label] = float(np.mean(jax.device_get(episodic_returns)))
            wandb.log({f"eval/episodic_return_{label}": metrics[label]}, step=iteration)

            if config["CAPTURE_VIDEO"]:
                renderer = jaxatari.make(config["ENV_ID"], mods=list(mods) or None).renderer
                frames = jnp.transpose(jax.vmap(renderer.render)(env_states), (0, 3, 1, 2))
                wandb.log({f"eval/video_{label}": wandb.Video(np.array(frames), fps=30, format="mp4")},
                          step=iteration)
                print(f"Video (eval) logged to wandb with {frames.shape[0]} frames.")
        if not config["SAVE_PATH"]:
            shutil.rmtree(save_dir, ignore_errors=True)
        return metrics

    key, reset_key = jax.random.split(key)
    global_step = 0
    next_obs, env_state = vmap_reset(jax.random.split(reset_key, config["NUM_ENVS"]))
    next_done = jnp.zeros(config["NUM_ENVS"], dtype=jnp.bool_)

    # based on https://github.com/google/evojax/blob/main/evojax/sim_mgr.py
    def step_once(carry, _):
        agent_state, obs, done, key, env_state = carry
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)
        next_obs, env_state, reward, next_done, info = vmap_step(env_state, action)
        storage = Storage(obs=obs, actions=action, logprobs=logprob, dones=done, values=value,
                          rewards=reward, returns=jnp.zeros_like(reward),
                          advantages=jnp.zeros_like(reward))
        return (agent_state, next_obs, next_done, key, env_state), (storage, info)

    @jax.jit
    def rollout(agent_state, next_obs, next_done, key, env_state):
        (agent_state, next_obs, next_done, key, env_state), (storage, info) = jax.lax.scan(
            step_once, (agent_state, next_obs, next_done, key, env_state), (), config["NUM_STEPS"])
        return agent_state, next_obs, next_done, storage, key, env_state, info

    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name,
                max_iterations=config["NUM_ITERATIONS"])
    rtpt.start()
    start_time = time.time()
    compile_time = None
    for iteration in range(1, config["NUM_ITERATIONS"] + 1):
        rtpt.step()
        if config["EVAL_DURING_TRAIN"] and iteration % config["EVAL_EVERY"] == 0:
            save_and_eval(iteration)

        iteration_time_start = time.time()
        agent_state, next_obs, next_done, storage, key, env_state, info = rollout(
            agent_state, next_obs, next_done, key, env_state)
        global_step += config["NUM_STEPS"] * config["NUM_ENVS"]
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        # SCoBots (SB3) anneals the clip range with the learning rate.
        clip_coef = config["CLIP_COEF"] * (anneal((iteration - 1) * config["NUM_MINIBATCHES"]
                                                  * config["UPDATE_EPOCHS"])
                                           if config["ANNEAL_CLIP"] else 1.0)
        agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl), key = update_ppo(
            agent_state, storage, key, clip_coef)
        if compile_time is None:
            compile_time = time.time()
            print(f"Compile + first iteration time: {compile_time - start_time:.2f} seconds.")

        wandb.log({
            "charts/avg_episodic_return": info["returned_episode_returns"].mean(),
            "charts/avg_episodic_length": info["returned_episode_lengths"].mean(),
            "charts/learning_rate": agent_state.opt_state[1].hyperparams["learning_rate"].item(),
            "charts/clip_coef": clip_coef,
            "losses/value_loss": v_loss[-1, -1].item(),
            "losses/policy_loss": pg_loss[-1, -1].item(),
            "losses/entropy": entropy_loss[-1, -1].item(),
            "losses/approx_kl": approx_kl[-1, -1].item(),
            "losses/loss": loss[-1, -1].item(),
            "charts/SPS": int(global_step / (time.time() - start_time)),
            "charts/SPS_update": int(config["BATCH_SIZE"] / (time.time() - iteration_time_start)),
            "charts/time": time.time() - start_time,
            "charts/global_step": global_step,
        }, step=iteration)

    end_time = time.time()
    print("Training done.")
    if compile_time is not None:
        print(f"Run time after first iteration: {end_time - compile_time:.2f} seconds.")
    print(f"Total train time: {end_time - start_time:.2f} seconds / {(end_time - start_time) / 60:.2f} minutes.")

    print("Evaluating final model ...")
    eval_metrics = save_and_eval(config["NUM_ITERATIONS"] + 1)
    print("Done.")
    wandb.finish()
    return eval_metrics


# ===========================================================================
# CLI: inspect the bottleneck / smoke-test all games without training
# ===========================================================================


def _describe(game: str) -> None:
    env = make_env(game)()
    height, width = env.image_space().shape[:2]
    labels, low, high = feature_layout(game, float(width), float(height))
    print(f"{game}: {len(labels)} concepts, actions {list(env.action_names)}")
    for i, (label, lo, hi) in enumerate(zip(labels, low, high)):
        print(f"  {i:4d}  {label:<44s} [{lo:7.1f}, {hi:7.1f}]")
    print(f"  concept reward (REWARD_MODE 1/2): {'yes' if game in CONCEPT_REWARDS else 'no'}")


def _selftest(games: Sequence[str]) -> int:
    failures = 0
    for game in games:
        try:
            mode = 2 if game in CONCEPT_REWARDS else 0
            env = make_env(game, reward_mode=mode)()
            obs, state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(0), 4))
            step = jax.jit(jax.vmap(env.step))
            for i in range(20):
                actions = jax.random.randint(jax.random.PRNGKey(i), (4,), 0, env.action_space().n)
                obs, state, reward, *_ = step(state, actions)
            obs, reward = np.asarray(obs), np.asarray(reward)
            assert obs.shape == (4, env.n_features), obs.shape
            assert np.isfinite(obs).all() and np.abs(obs).max() <= 1.0, "feature outside [-1, 1]"
            assert np.isfinite(reward).all(), "non-finite reward"
            print(f"  ok    {game:<17s} {env.n_features:4d} concepts, "
                  f"{env.action_space().n:2d} actions, reward mode {mode}")
        except Exception as exc:  # noqa: BLE001 - report every game
            failures += 1
            print(f"  FAIL  {game:<17s} {type(exc).__name__}: {exc}")
    print(f"{len(games) - failures}/{len(games)} games ok")
    return failures


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SCoBots concept bottleneck on JAXtari")
    parser.add_argument("--describe", metavar="GAME", choices=sorted(FOCUS),
                        help="print a game's concept vector layout")
    parser.add_argument("--selftest", nargs="*", metavar="GAME", choices=sorted(FOCUS),
                        help="reset and step every game (or the given ones)")
    args = parser.parse_args()
    if args.describe:
        _describe(args.describe)
    elif args.selftest is not None:
        raise SystemExit(_selftest(args.selftest or sorted(FOCUS)))
    else:
        parser.print_help()
