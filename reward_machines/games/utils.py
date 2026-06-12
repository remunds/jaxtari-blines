import numpy as np
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
