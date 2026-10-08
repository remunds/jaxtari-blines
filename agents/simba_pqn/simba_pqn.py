import os
import random
import time
from functools import partial

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, he_normal, orthogonal
from flax.training.train_state import TrainState
from rtpt import RTPT

import jaxtari
from jaxtari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
)
from agents.run_utils import eval_mod_configs, plan_chunks
from agents.simba_pqn.simba_pqn_eval import evaluate


def make_env(env_id, mods=None, pixel_based=True, native_downscaling=True, eval=False):
    mods = list(mods) if mods else None
    if not eval and mods:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        env = jaxtari.make(env_id, mods=mods)
        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=not eval,
            first_fire=True,
            noop_max=30,
            full_action_space=False,
        )
        if pixel_based:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(84, 84),
                grayscale=True,
                use_native_downscaling=native_downscaling,
                smooth_image=False,
                frame_stack_size=4,
                frame_skip=4,
                max_pooling=True,
                clip_reward=not eval,
            )
        else:
            env = FlattenObservationWrapper(
                NormalizeObservationWrapper(
                    ObjectCentricWrapper(
                        env,
                        frame_stack_size=4,
                        frame_skip=4,
                        clip_reward=not eval,
                    )
                )
            )
        return LogWrapper(env)

    return thunk


class SimBaResidualBlock(nn.Module):
    hidden_dim: int
    expansion_factor: int = 4

    @nn.compact
    def __call__(self, x):
        res = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim * self.expansion_factor, kernel_init=he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=he_normal(), bias_init=constant(0.0))(x)
        return res + x


class SimBaTail(nn.Module):
    action_dim: int
    hidden_dim: int
    num_blocks: int
    expansion_factor: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)

        for _ in range(self.num_blocks):
            x = SimBaResidualBlock(self.hidden_dim, self.expansion_factor)(x)
        x = nn.LayerNorm()(x)
        return nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)


class SimBaQNetwork(nn.Module):
    action_dim: int
    hidden_dim: int = 512
    num_blocks: int = 2
    expansion_factor: int = 4

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.astype(jnp.float32)
        x = x / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID", kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID", kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID", kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))

        return SimBaTail(self.action_dim, self.hidden_dim, self.num_blocks, self.expansion_factor)(x)


class SimBaMLP_QNetwork(nn.Module):
    action_dim: int
    hidden_dim: int = 512
    num_blocks: int = 2
    expansion_factor: int = 4

    @nn.compact
    def __call__(self, x):
        return SimBaTail(self.action_dim, self.hidden_dim, self.num_blocks, self.expansion_factor)(x)


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    rewards: jnp.array
    dones: jnp.array
    values: jnp.array
    returns: jnp.array


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    env_id = config["ENV_ID"]
    pixel_based = config["PIXEL_BASED"]
    run_name = f"{env_id}_{config['EXP_NAME']}_{'oc' if not pixel_based else 'pixel'}_{config['SEED']}"

    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
        mode=config.get("WANDB_MODE", "online"),
    )
    wandb.define_metric("*", step_metric="charts/global_step")

    # same seeding order as pqn/dqn: python/numpy seeds, then key -> q_key -> env resets
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    gamma = config.get("GAMMA", 0.99)
    q_lambda = config.get("Q_LAMBDA", 0.65)
    num_steps = config.get("NUM_STEPS", 32)
    num_envs = config["NUM_ENVS"]
    num_minibatches = config.get("NUM_MINIBATCHES", 4)
    update_epochs = config.get("UPDATE_EPOCHS", 4)
    max_grad_norm = config.get("MAX_GRAD_NORM", 10.0)
    total_timesteps = config["TOTAL_TIMESTEPS"]
    start_e = config.get("START_E", 1.0)
    end_e = config.get("END_E", 0.01)
    exploration_steps = config.get("EXPLORATION_FRACTION", 0.10) * total_timesteps
    learning_rate = config.get("LEARNING_RATE", 2.5e-4)
    anneal_lr = config.get("ANNEAL_LR", True)

    simba_hidden_dim = config.get("SIMBA_HIDDEN_DIM", 512)
    simba_num_blocks = config.get("SIMBA_NUM_BLOCKS", 2)
    simba_expansion_factor = config.get("SIMBA_EXPANSION_FACTOR", 4)
    Network = partial(
        SimBaQNetwork if pixel_based else SimBaMLP_QNetwork,
        hidden_dim=simba_hidden_dim,
        num_blocks=simba_num_blocks,
        expansion_factor=simba_expansion_factor,
    )

    env = make_env(env_id, list(config.get("TRAIN_MODS") or []), pixel_based, config.get("NATIVE_DOWNSCALING", True))()
    action_dim = env.action_space().n
    obs_shape = env.observation_space().shape
    if pixel_based:
        obs_shape = obs_shape[:-1]  # drop the trailing grayscale channel

    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    scan_steps, num_chunks, steps_per_chunk = plan_chunks(total_timesteps, batch_size, config.get("SCAN_STEPS", 1000))

    network = Network(action_dim=action_dim)
    total_grad_steps = scan_steps * num_chunks * update_epochs * num_minibatches
    lr = optax.linear_schedule(learning_rate, 0.0, total_grad_steps) if anneal_lr else learning_rate
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.radam(lr))

    def reset_envs(rng):
        obs, state = jax.vmap(env.reset)(rng)
        return obs.reshape(num_envs, *obs_shape), state

    def step_envs(state, action):
        obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        done = jnp.logical_or(terminated, truncated)
        return obs.reshape(num_envs, *obs_shape), state, reward, done, info

    key, q_key = jax.random.split(key, 2)
    q_state = TrainState.create(
        apply_fn=network.apply,
        params=network.init(q_key, jnp.zeros((1, *obs_shape))),
        tx=tx,
    )
    obs, env_state = reset_envs(jax.random.split(key, num_envs))
    carry = (
        q_state,
        env_state,
        obs,
        jnp.zeros(num_envs, dtype=jnp.float32),  # done flags of the last step
        key,
        jnp.array(0, dtype=jnp.int32),  # env steps taken
    )

    def pqn_update(carry):
        """One rollout of num_steps per env, Q(lambda) targets, and update_epochs of minibatch SGD."""
        q_state, env_state, obs, last_done, key, global_step = carry

        def rollout_step(c, _):
            env_state, last_obs, last_done, key, global_step = c
            epsilon = jnp.maximum(
                end_e,
                start_e + (end_e - start_e) * global_step.astype(jnp.float32) / exploration_steps,
            )
            q_vals = network.apply(q_state.params, last_obs)
            greedy = jnp.argmax(q_vals, axis=-1)

            key, act_key, exp_key = jax.random.split(key, 3)
            random_actions = jax.random.randint(act_key, (num_envs,), 0, action_dim)
            explore = jax.random.uniform(exp_key, (num_envs,)) < epsilon
            actions = jnp.where(explore, random_actions, greedy)

            next_obs, next_state, reward, next_done, info = step_envs(env_state, actions)
            storage = Storage(
                obs=last_obs,
                actions=actions,
                rewards=reward,
                dones=last_done,
                values=jnp.max(q_vals, axis=-1),
                returns=jnp.zeros_like(reward),
            )
            new_c = (next_state, next_obs, next_done.astype(jnp.float32), key, global_step + num_envs)
            return new_c, (storage, info)

        (env_state, next_obs, next_done, key, global_step), (storage, infos) = jax.lax.scan(
            rollout_step,
            (env_state, obs, last_done, key, global_step),
            None,
            length=num_steps,
        )

        # Q(lambda) returns, computed backwards through the rollout
        next_val = jnp.max(network.apply(q_state.params, next_obs), axis=-1)
        next_values = jnp.concatenate([storage.values[1:], next_val[None]], axis=0)
        next_dones = jnp.concatenate([storage.dones[1:], next_done[None]], axis=0)

        def q_lambda_step(next_return, inp):
            reward, next_value, nd = inp
            ret = reward + gamma * (q_lambda * next_return + (1.0 - q_lambda) * next_value) * (1.0 - nd)
            return ret, ret

        _, returns = jax.lax.scan(
            q_lambda_step,
            next_val,
            (storage.rewards, next_values, next_dones),
            reverse=True,
        )
        storage = storage.replace(returns=returns)

        def update_epoch(carry, _):
            q_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x):
                x = jax.random.permutation(subkey, x)
                return jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])

            flat = jax.tree_util.tree_map(flatten, storage)
            shuffled = jax.tree_util.tree_map(convert_data, flat)

            def update_minibatch(q_state, mb):
                def loss_fn(params):
                    q_vals = network.apply(params, mb.obs)
                    q_sel = q_vals[jnp.arange(minibatch_size), mb.actions]
                    return 0.5 * jnp.mean((mb.returns - q_sel) ** 2), q_sel.mean()

                (loss, q_mean), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_state.params)
                return q_state.apply_gradients(grads=grads), (loss, q_mean)

            q_state, (loss, q_mean) = jax.lax.scan(update_minibatch, q_state, shuffled)
            return (q_state, key), (loss, q_mean)

        (q_state, key), (loss, q_mean) = jax.lax.scan(update_epoch, (q_state, key), None, length=update_epochs)

        new_carry = (q_state, env_state, next_obs, next_done, key, global_step)
        return new_carry, (infos, loss[-1, -1], q_mean[-1, -1])

    def save_and_eval(q_params, step_count):
        if config.get("SAVE_PATH") is not None:
            save_dir = os.path.join(config["SAVE_PATH"], run_name)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model')
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes([config, q_params]))
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")
        metrics = {}
        for mods_cfg, mod_label in eval_mod_configs(config):
            episodic_returns, env_states = evaluate(
                q_params,
                partial(
                    make_env,
                    mods=mods_cfg,
                    pixel_based=pixel_based,
                    native_downscaling=config.get("NATIVE_DOWNSCALING", True),
                    eval=True,
                ),
                env_id,
                eval_episodes=10,
                Model=Network,
                seed=config["SEED"] + 42,  # use a different seed for evaluation
            )
            metrics[mod_label] = float(np.mean(jax.device_get(episodic_returns)))
            wandb.log({f"eval/episodic_return_{mod_label}": metrics[mod_label]}, step=step_count)

            if config.get("CAPTURE_VIDEO", False):
                # Instantiate a clean renderer immune to the training env's downscaling
                clean_renderer = jaxtari.make(env_id, mods=mods_cfg).renderer
                frames = jnp.transpose(jax.vmap(clean_renderer.render)(env_states), (0, 3, 1, 2))
                video = wandb.Video(np.array(frames), fps=30, format="mp4")
                wandb.log({f"eval/video_{mod_label}": video}, step=step_count)
                print(f"Video (eval) logged to wandb with {frames.shape[0]} frames ({mod_label}).")
        return metrics

    def train_chunk(carry):
        return jax.lax.scan(lambda c, _: pqn_update(c), carry, None, length=scan_steps)

    print("[simba_pqn] start compile...")
    compile_start = time.perf_counter()
    compiled = jax.jit(train_chunk, donate_argnums=(0,)).lower(carry).compile()
    print(f"[simba_pqn] compilation time: {time.perf_counter() - compile_start:.2f}s")

    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=num_chunks)
    rtpt.start()
    run_time = time.perf_counter()
    print(f"[simba_pqn] starting training: {num_chunks} chunks x {steps_per_chunk} env steps")
    for chunk in range(num_chunks):
        rtpt.step()
        if config["EVAL_DURING_TRAIN"] and chunk > 0 and chunk % config["EVAL_EVERY"] == 0:
            save_and_eval(carry[0].params, int(carry[-1]))

        chunk_start = time.perf_counter()
        carry, (infos, loss, q_mean) = compiled(carry)
        global_step = int(carry[-1])
        metrics = {
            "charts/avg_episodic_return": float(infos["returned_episode_returns"][-1].mean()),
            "charts/avg_episodic_length": float(infos["returned_episode_lengths"][-1].mean()),
            "losses/td_loss": loss[-1].item(),
            "losses/q_values": q_mean[-1].item(),
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(steps_per_chunk / (time.perf_counter() - chunk_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }
        wandb.log(metrics, step=global_step)
        print(
            f"[simba_pqn] chunk {chunk + 1}/{num_chunks} | global_step {global_step} | "
            f"avg_return {metrics['charts/avg_episodic_return']:.2f} | "
            f"avg_length {metrics['charts/avg_episodic_length']:.2f} | "
            f"td_loss {metrics['losses/td_loss']:.4f} | q_val {metrics['losses/q_values']:.4f} | "
            f"SPS {metrics['charts/SPS']}"
        )

    eval_metrics = save_and_eval(carry[0].params, global_step + 1)
    wandb.finish()
    return eval_metrics
