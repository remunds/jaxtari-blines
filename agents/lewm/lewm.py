"""LeWorldModel (LeWM) baseline: entry point for this repository's runner.

LeWM is a joint-embedding predictive world model (Maes, Le Lidec, Scieur, LeCun,
Balestriero, 2026, arXiv:2603.19312; reference code github.com/lucas-maes/le-wm).
It predicts the *embedding* of the next frame rather than the next frame's pixels,
and relies on a regulariser called SIGReg rather than a stop-gradient to keep the
representation from collapsing.

Being reward-free, it produces no game score on its own. `single_run` therefore
does two things in sequence, and the score it returns is the second one:

  1. train the world model on reward-free data from a uniform-random policy
  2. train a PPO agent on the representation that produced

NOTE ON FRAMEWORK. The rest of this repository is JAX-native. This baseline keeps
the environment in JAX and writes the networks in PyTorch, because it was
developed on Apple silicon, where JAX has no GPU backend and runs on CPU while
PyTorch reaches the GPU through Metal. One forward and backward pass of the trunk
at batch 512 measured 772 ms under JAX on CPU against 37 ms under PyTorch on
Metal, which took the agent from 8 to roughly 1300 environment steps per second.
On the CUDA machines this repository targets that argument does not apply, and a
JAX port is the obvious next step. See agents/lewm/README.md.
"""

from argparse import Namespace
from pathlib import Path

from agents.lewm import ppo_lewm, world_model

# PPO knobs this repository's configs express, mapped onto the agent's own names.
_PPO_KEYS = {
    "TOTAL_TIMESTEPS": "total_timesteps",
    "LEARNING_RATE": "lr",
    "NUM_ENVS": "num_envs",
    "NUM_STEPS": "num_steps",
    "ANNEAL_LR": "anneal_lr",
    "GAMMA": "gamma",
    "GAE_LAMBDA": "gae_lambda",
    "NUM_MINIBATCHES": "num_minibatches",
    "UPDATE_EPOCHS": "update_epochs",
    "NORM_ADV": "norm_adv",
    "CLIP_COEF": "clip_coef",
    "ENT_COEF": "ent_coef",
    "VF_COEF": "vf_coef",
    "MAX_GRAD_NORM": "max_grad_norm",
}

# World-model knobs, all optional, prefixed WM_ so they cannot collide with PPO's.
_WM_KEYS = {
    "WM_TOTAL_STEPS": "total_steps",
    "WM_SEQ_LEN": "seq_len",
    "WM_BATCH_SIZE": "batch_size",
    "WM_LR": "lr",
    "WM_SIGREG_WEIGHT": "sigreg_weight",
    "WM_INIT_SEQUENCES": "init_sequences",
    "WM_COLLECT_EVERY": "collect_every",
    "WM_COLLECT_N": "collect_n",
    "WM_EVAL_SEQ": "eval_seq",
    "WM_ACTION_COND": "action_cond",
    "WM_MAX_EPISODE_STEPS": "max_episode_steps",
}


def _namespace(defaults: dict, mapping: dict, config: dict, **forced) -> Namespace:
    """Build an args Namespace: documented defaults, overridden by this config."""
    args = dict(defaults)
    for cfg_key, arg_key in mapping.items():
        if config.get(cfg_key) is not None:
            value = config[cfg_key]
            if arg_key in ("total_timesteps", "total_steps", "num_envs", "num_steps",
                           "num_minibatches", "update_epochs", "seq_len", "batch_size",
                           "init_sequences", "collect_every", "collect_n", "eval_seq",
                           "max_episode_steps"):
                value = int(float(value))
            elif isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    pass
            args[arg_key] = value
    args.update(forced)
    return Namespace(**args)


def single_run(config: dict):
    """Train LeWM, then PPO on its representation. Returns {label: mean return}.

    `FEATURES` selects the trunk: "lewm_frozen" (the default, and the configuration
    the accompanying report evaluates), "lewm_finetune", or "scratch" for the
    control that skips world-model training entirely.
    """
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    game = config.get("ENV_ID", "pong")
    seed = int(config.get("SEED", 1))
    features = config.get("FEATURES", "lewm_frozen")
    outdir = Path(config.get("SAVE_PATH") or "./models") / "lewm"
    outdir.mkdir(parents=True, exist_ok=True)

    encoder_path = None
    if features != "scratch":
        wm_args = _namespace(
            world_model.DEFAULTS, _WM_KEYS, config,
            game=game, seed=seed, outdir=str(outdir / "world_model"), plot=True,
        )
        print(f"[lewm] training the world model on {game} "
              f"for {wm_args.total_steps} steps")
        world_model.train(wm_args)
        encoder_path = outdir / "world_model" / game / "model.pt"

    ppo_defaults = vars(ppo_lewm.build_parser().parse_args([]))
    ppo_args = _namespace(
        ppo_defaults, _PPO_KEYS, config,
        game=game, seed=seed, features=features,
        outdir=str(outdir / "agent"),
        encoder=str(encoder_path) if encoder_path else None,
    )
    print(f"[lewm] training PPO on {game} with the {features} trunk "
          f"for {ppo_args.total_timesteps} steps")
    history = ppo_lewm.train(ppo_args)

    entries = history["history"] if isinstance(history, dict) else history
    returns = [h["episodic_return_mean"] for h in entries
               if h.get("episodic_return_mean") is not None]
    tail = returns[-max(1, len(returns) // 10):] if returns else []
    final = sum(tail) / len(tail) if tail else None

    return {"default": final}
