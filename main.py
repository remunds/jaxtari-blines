import hydra
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    merged_config = {**config, **config.get("alg", {})}
    print("Config:\n", OmegaConf.to_yaml(OmegaConf.create(config)))
    n_seeds = merged_config.get("NUM_SEEDS", 1)
    seed_base = int(merged_config.get("SEED", 0))  # SEED acts as base: seeds run as SEED, SEED+1, ...

    all_metrics = []
    for seed in range(n_seeds):
        if merged_config["ALG"] == "PPO":
            from agents.ppo.ppo import single_run
            run_fn = single_run
        elif merged_config["ALG"] == "DQN":
            from agents.dqn.dqn import single_run
            run_fn = single_run
        elif merged_config["ALG"] == "C51":
            from agents.c51.c51 import single_run
            run_fn = single_run
        elif merged_config["ALG"] == "PQN":
            from agents.pqn.pqn import single_run
            run_fn = single_run
        elif merged_config["ALG"] == "DDPG":
            from agents.ddpg.ddpg import single_run
            run_fn = single_run
        elif merged_config["ALG"] == "CROSSQ":
            from agents.crossq.crossq import single_run
            run_fn = single_run

        print(f"Running seed {seed_base + seed} ...")
        merged_config["SEED"] = seed_base + seed
        metrics = run_fn(merged_config)
        metrics["ALG"] = merged_config["ALG"]
        metrics["ENV_ID"] = merged_config["ENV_ID"]
        metrics["PIXEL_BASED"] = merged_config.get("PIXEL_BASED", False)
        metrics["SEED"] = seed
        all_metrics.append(metrics)

    print("Metrics: ", all_metrics)
if __name__ == "__main__":
    main()
