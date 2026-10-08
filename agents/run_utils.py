def plan_chunks(total_steps: int, steps_per_update: int, scan_steps: int):
    """Split an env-step budget into whole jitted chunks.

    Returns (scan_steps, num_chunks, steps_per_chunk). Only whole chunks are run,
    so every agent sees exactly num_chunks * steps_per_chunk <= total_steps env steps.
    """
    num_updates = total_steps // steps_per_update
    if num_updates == 0:
        raise ValueError(f"TOTAL_TIMESTEPS={total_steps} is smaller than one update ({steps_per_update} steps)")
    scan_steps = min(scan_steps, num_updates)
    num_chunks = num_updates // scan_steps
    return scan_steps, num_chunks, scan_steps * steps_per_update


def eval_mod_configs(config: dict):
    """(mods, label) pairs to evaluate on: the default env plus each eval mod."""
    eval_mods = list(config["EVAL_MODS"] or config["TRAIN_MODS"] or [])
    mod_configs = [([], "default")]
    for mod in eval_mods:
        mods = list(mod) if isinstance(mod, (list, tuple)) else [mod]
        label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods)
        mod_configs.append((mods, label))
    return mod_configs
