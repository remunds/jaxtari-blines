import hydra
from omegaconf import OmegaConf
from agents.dqn.dqn import dqn_run
from agents.ppo.ppo import single_run

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    merged_config = {**config, **config.get("alg", {})}
    print("Config:\n", OmegaConf.to_yaml(OmegaConf.create(config)))
    n_seeds = merged_config.get("NUM_SEEDS", 1)

    if isinstance(merged_config.get("TRAIN_MODS"), list):
        merged_config["TRAIN_MODS"] = tuple(merged_config["TRAIN_MODS"])
    if isinstance(merged_config.get("EVAL_MODS"), list):
        merged_config["EVAL_MODS"] = tuple(merged_config["EVAL_MODS"])

    all_metrics = []
    for seed in range(n_seeds):
        if merged_config["ALG"] == "PPO":
            run_fn = single_run
        elif merged_config["ALG"] == "DQN":
            run_fn = dqn_run
        print(f"Running seed {seed} ...")
        merged_config["SEED"] = seed
        metrics = run_fn(merged_config)
        metrics["ALG"] = merged_config["ALG"]
        metrics["ENV_ID"] = merged_config["ENV_ID"]
        metrics["PIXEL_BASED"] = merged_config.get("PIXEL_BASED", False)
        metrics["SEED"] = seed
        all_metrics.append(metrics)

    print("Metrics: ", all_metrics)
if __name__ == "__main__":
    main()
