import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    merged_config = {**config, **config.get("alg", {})}
    print("Config:\n", OmegaConf.to_yaml(OmegaConf.create(config)))
    n_seeds = merged_config.get("NUM_SEEDS", 1)
    starting_seed = merged_config.get("SEED", 0)

    alg = merged_config["ALG"]

    all_metrics = []
    for seed in range(n_seeds):
        if alg == "PPO":
            from agents.ppo.ppo import single_run
        elif alg == "DQN":
            from agents.dqn.dqn import single_run
        elif alg == "PQN":
            from agents.pqn.pqn import single_run
        elif alg == "RAINBOW":
            from agents.rainbow.rainbow import single_run
        elif alg == "C51":
            from agents.c51.c51 import single_run
        elif alg == "APQN":
            from agents.apqn.apqn import single_run
        elif alg == "DROQ":
            from agents.droq.droq import single_run
        elif alg == "IQN":
            from agents.iqn.iqn import single_run
        elif alg == "SIMBA_PQN":
            from agents.simba_pqn.simba_pqn import single_run
        elif alg == "SIMBA_SAC":
            from agents.simba_sac.simba_sac import single_run
        else:
            raise ValueError(f"Unknown ALG: {alg}")

        used_seed = starting_seed + seed
        print(f"Running seed {used_seed} ...")
        merged_config["SEED"] = used_seed
        metrics = single_run(merged_config)
        metrics["ALG"] = alg
        metrics["ENV_ID"] = merged_config["ENV_ID"]
        metrics["PIXEL_BASED"] = merged_config.get("PIXEL_BASED", False)
        metrics["SEED"] = used_seed
        all_metrics.append(metrics)

    print("Metrics: ", all_metrics)


if __name__ == "__main__":
    main()
