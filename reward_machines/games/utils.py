import numpy as np
import jax
import jax.numpy as jnp


def build_transitions(num_props, prop_index, transitions):
    """Convert a readable transition list into the five fixed arrays."""
    T = len(transitions)
    from_s = np.zeros(T, dtype=np.int32)
    to_s   = np.zeros(T, dtype=np.int32)
    rew    = np.zeros(T, dtype=np.float32)
    req_t  = np.zeros((T, num_props), dtype=np.int32)
    req_f  = np.zeros((T, num_props), dtype=np.int32)
    for i, tr in enumerate(transitions):
        from_s[i] = tr["from"]
        to_s[i]   = tr["to"]
        rew[i]    = tr.get("reward", 0.0)
        for name in tr.get("true", []):
            req_t[i, prop_index[name]] = 1
        for name in tr.get("false", []):
            req_f[i, prop_index[name]] = 1
    return (jnp.array(from_s), jnp.array(req_t), jnp.array(req_f),
            jnp.array(to_s), jnp.array(rew))


# ---------------------------------------------------------------------------
# Frame-indexing helpers for object-centric observations.
#
# The obs seen by GameRM.get_events(obs) / GameRM.potential(obs) is the
# flattened stack of `frame_stack_size` frames with the NEWEST frame LAST.
# Within one frame, fields follow the declaration order of the game's
# Observation; every ObjectObservation contributes
# x, y, width, height, active, visual_id, state, orientation, each as a
# length-N block for N objects (grouped by field, not interleaved).
# NormalizeObservationWrapper only rescales values (by the space bounds).
# ---------------------------------------------------------------------------

def frame_base(num_features, frame_stack_size=4, frames_ago=0):
    """Start index of a frame in the stacked obs (frames_ago=0 -> newest)."""
    return (frame_stack_size - 1 - frames_ago) * num_features


def field(obs, num_features, offset, frame_stack_size=4, frames_ago=0):
    """Scalar field at `offset` within a frame."""
    return obs[frame_base(num_features, frame_stack_size, frames_ago) + offset]


def field_slice(obs, num_features, offset, length, frame_stack_size=4, frames_ago=0):
    """Length-`length` block starting at `offset` within a frame."""
    start = frame_base(num_features, frame_stack_size, frames_ago) + offset
    return jax.lax.dynamic_slice(obs, (start,), (length,))
