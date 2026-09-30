"""Static option structure derived from a GameRM.

HRM keys options on the formula of an RM edge, not on the edge itself, so
edges labelled with the same formula share a single option policy. Everything
here is computed once in Python and baked into the jitted graph as constants.
"""

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from reward_machines.games.game_rm import GameRM


def _formula_key(t: dict) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Canonical, order-independent identity of an edge's guard formula."""
    return (tuple(sorted(t.get("true", []))), tuple(sorted(t.get("false", []))))


def _formula_name(key: Tuple[Tuple[str, ...], Tuple[str, ...]]) -> str:
    pos, neg = key
    return "&".join(list(pos) + [f"!{p}" for p in neg]) or "true"


@dataclass(frozen=True)
class OptionSpec:
    """Immutable option structure. Safe to close over inside jit."""

    num_options: int
    names: Tuple[str, ...]
    option_of_transition: jnp.ndarray  # (T,) int32, -1 for non-option edges
    is_option_edge: jnp.ndarray  # (T,) bool
    availability: jnp.ndarray  # (num_states, num_options) bool
    scatter: jnp.ndarray  # (T, num_options) float32, one-hot of option_of_transition

    def describe(self) -> str:
        lines = [f"{self.num_options} options from {self.is_option_edge.shape[0]} edges:"]
        for i, name in enumerate(self.names):
            states = np.where(np.asarray(self.availability)[:, i])[0].tolist()
            lines.append(f"  o{i}: {name:<40} available in u={states}")
        n_choice = int((np.asarray(self.availability).sum(axis=1) > 1).sum())
        lines.append(f"  states with a real meta-decision (>1 option): {n_choice}")
        return "\n".join(lines)


def build_options(game_rm: GameRM) -> OptionSpec:
    transitions = game_rm.TRANSITIONS
    num_states = game_rm.num_states()
    num_transitions = len(transitions)

    key_to_id: dict = {}
    names: list = []
    option_of_transition = np.full(num_transitions, -1, dtype=np.int32)

    for i, t in enumerate(transitions):
        if not t.get("option", False):
            continue
        key = _formula_key(t)
        if key not in key_to_id:
            key_to_id[key] = len(names)
            names.append(_formula_name(key))
        option_of_transition[i] = key_to_id[key]

    num_options = len(names)
    if num_options == 0:
        raise ValueError(
            "No option edges found. Mark subgoal transitions with "
            '\'"option": True\' in GameRM.TRANSITIONS.'
        )

    availability = np.zeros((num_states, num_options), dtype=bool)
    for i, t in enumerate(transitions):
        if option_of_transition[i] >= 0:
            availability[t["from"], option_of_transition[i]] = True

    # The meta-controller must always have something to pick. A state with no
    # outgoing option edge would make argmax over an all-masked row undefined.
    orphans = np.where(~availability.any(axis=1))[0].tolist()
    if orphans:
        raise ValueError(
            f"RM states {orphans} have no outgoing option edge. Add an option "
            "edge (a self-loop subgoal is fine) or the meta-controller has "
            "nothing to choose in those states."
        )

    opt_of_t = jnp.asarray(option_of_transition)
    return OptionSpec(
        num_options=num_options,
        names=tuple(names),
        option_of_transition=opt_of_t,
        is_option_edge=jnp.asarray(option_of_transition >= 0),
        availability=jnp.asarray(availability),
        # one_hot(-1, N) is all zeros, so non-option edges contribute nothing.
        scatter=jax.nn.one_hot(opt_of_t, num_options, dtype=jnp.float32),
    )
