import re
import jax.numpy as jnp
import numpy as _np


def _parse_patterns(patterns):
    """Helper: normalize patterns input to a list of strings."""
    if patterns is None:
        return None
    if isinstance(patterns, str):
        # split by comma / semicolon
        return [p.strip() for p in re.split(',|;', patterns) if p.strip()]
    # assume iterable
    return list(patterns)


def compute_state_mask(states,
                        env_config=None,
                        exclude_patterns=None,
                        variance_threshold: float = 1e-8,
                        log: bool = False):
    """Return boolean mask of state vars to KEEP.

    Parameters
    ----------
    states : jnp.ndarray (N, D)
        State vectors.
    env_config : Optional object with attribute `state_names` matching size D.
    exclude_patterns : None, str, or Iterable[str]
        Regex patterns for variable names that should be *excluded*.
    variance_threshold : float
        Minimum variance required to keep a variable.
    log : bool
        Whether to print informative messages.

    Returns
    -------
    mask : jnp.ndarray bool of length D
        True for variables to keep.
    info : dict
        Auxiliary information (currently filtered_state_names, excluded_names, exclude_mask).
    """
    # Variance filter
    var_states = jnp.var(states, axis=0) > variance_threshold

    patterns = _parse_patterns(exclude_patterns)
    exclude_mask = jnp.zeros_like(var_states, dtype=bool)
    excluded_names = []
    if patterns:
        if env_config is not None and hasattr(env_config, 'state_names'):
            compiled = [re.compile(p) for p in patterns]
            names = env_config.state_names
            exclude_mask = jnp.array([any(p.search(name) for p in compiled) for name in names])
            excluded_names = [name for name, m in zip(names, exclude_mask) if m]
            if log and excluded_names:
                print(f"[state_utils] Excluding {len(excluded_names)} variables via patterns: {patterns}")
        else:
            if log:
                print("[state_utils] Warning: env_config.state_names unavailable; cannot apply exclude patterns.")
    # Combine masks: keep vars with variance AND not excluded
    keep_mask = jnp.logical_and(var_states, ~exclude_mask)

    filtered_state_names = None
    if env_config is not None and hasattr(env_config, 'state_names'):
        filtered_state_names = [name for name, keep in zip(env_config.state_names, keep_mask) if keep]

    info = {
        'filtered_state_names': filtered_state_names,
        'excluded_names': excluded_names,
        'exclude_mask': exclude_mask,
        'variance_mask': var_states,
    }
    return keep_mask, info 


def vector_to_state_dict(vec, state_names, state_mappings, state_dims):
    """
    Reconstruct a state dict from a flat vector.

    Parameters:
    - vec: 1D array of concatenated state values.
    - state_names: list of keys in the order they were flattened.
    - state_mappings: dict mapping each key to None or a dict of value->code.
    - state_dims: list of ints giving number of vector entries per key.

    Returns:
    - state: dict mapping each key to its original value (scalar, list, or category).
    """
    vec = _np.asarray(vec)
    state = {}
    idx = 0
    for name, dim in zip(state_names, state_dims):
        sub = vec[idx:idx+dim]
        mapping = state_mappings.get(name)
        if mapping is None:
            # numeric or boolean scalar
            if dim == 1:
                val = sub[0]
            else:
                val = sub.copy()
            state[name] = val.item() if hasattr(val, 'item') else val
        else:
            # categorical or multi-hot list
            inv_map = {v: k for k, v in mapping.items()}
            if dim == 1:
                code = int(sub[0])
                state[name] = inv_map[code]
            else:
                # multi-hot: collect all indices with non-zero
                idxs = _np.nonzero(sub)[0]
                state[name] = [inv_map[i] for i in idxs.tolist()]
        idx += dim
    return state 