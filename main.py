import traceback

import hydra
import wandb
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    merged_config = {**config, **config.get("alg", {})}
    print("Config:\n", OmegaConf.to_yaml(OmegaConf.create(config)))
    n_seeds = merged_config.get("NUM_SEEDS", 1)

    if merged_config["ALG"] == "PPO":
        from agents.ppo.ppo import single_run
    elif merged_config["ALG"] == "DQN":
        from agents.dqn.dqn import single_run
    elif merged_config["ALG"] == "RAINBOW":
        from agents.rainbow.rainbow import single_run
    elif merged_config["ALG"] == "C51":
        from agents.c51.c51 import single_run
    else:
        raise ValueError(f"unknown ALG {merged_config['ALG']!r}")
    run_fn = single_run

    all_metrics = []
    starting_seed = merged_config.get("SEED", 0)
    for seed in range(n_seeds):
        used_seed = starting_seed + seed
        print(f"Running seed {used_seed} ...")
        merged_config["SEED"] = used_seed
        try:
            metrics = run_fn(merged_config)
        except Exception as e:
            # One seed failing must not take the remaining seeds down with it.
            print(f"[ERROR] seed {used_seed} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            if wandb.run is not None:
                wandb.finish(exit_code=1)
            continue
        metrics["ALG"] = merged_config["ALG"]
        metrics["ENV_ID"] = merged_config["ENV_ID"]
        metrics["PIXEL_BASED"] = merged_config.get("PIXEL_BASED", False)
        metrics["SEED"] = used_seed
        all_metrics.append(metrics)

    print("Metrics: ", all_metrics)


if __name__ == "__main__":
    main()