"""
PureJaxRL version of CleanRL's DQN: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py
"""
from functools import partial
import time
import os
import random
import jax
import jax.numpy as jnp
from jax.random import orthogonal
import numpy as np
import chex
import flax
import jaxatari
from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper, LogWrapper, NormalizeObservationWrapper, ObjectCentricWrapper, PixelObsWrapper
import wandb
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from agents.dqn.dqn_eval import evaluate_dqn
import flashbax as fbx
from agents.dqn.types import TimeStep
from reward_machines.games.game_rm import GameRM
from reward_machines.reward_machine import RewardMachine
from reward_machines.reward_machine_wrapper import RewardMachineWrapper
from reward_machines.rm_registry import GAME_RM_REGISTRY

def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False, game_rm: GameRM | None=None):
    def thunk():
        # For training (eval=False), avoid applying multiple potentially conflicting
        # mods at once. In that case, fall back to the base environment.
        # For evaluation (eval=True), we trust the caller to pass either a single
        # mod or an explicit list; this is used in the per-mod video generation.
        active_mods = mods
        if not eval and isinstance(active_mods, (list, tuple)) and len(active_mods) > 1:
            active_mods = []

        # Normalize to None or list for jaxatari.make
        if isinstance(active_mods, (list, tuple)) and len(active_mods) == 0:
            mods_arg = None
        else:
            mods_arg = active_mods

        env = jaxatari.make(env_id, mods=mods_arg)

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
                clip_reward=False, # only active during training
            )
        else:
            env = ObjectCentricWrapper(
                        env,
                        frame_stack_size=4,
                        frame_skip=4,
                        clip_reward=True,
                    )
            env = FlattenObservationWrapper(
                NormalizeObservationWrapper(
                    env,
                    dtype=jnp.float32,
                )
            )
            if game_rm is not None:
                rm = RewardMachine(game_rm)
                env = RewardMachineWrapper(env, rm)
        env = LogWrapper(env)
        return env
    return thunk


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

# class MLP_QNetwork(nn.Module):
#     action_dim: int
#
#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(461, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
#         x = nn.relu(x)
#         x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
#         x = nn.relu(x)
#         x = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
#         return x


class DQNTrainState(TrainState):
    target_params: flax.core.FrozenDict


@flax.struct.dataclass
class EpisodeStatistics: 
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


def build_eval_fn(env, apply_fn, eval_episodes, max_steps, action_dim):
   

    def wrapped_reset(key):
        next_obs, state = env.reset(key)
        return next_obs.squeeze()[None, ...], state

    def wrapped_step(state, action):
        next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze()[None, ...], next_state, reward, done, info

    def get_action(params, obs, key, epsilon):
        q_values = apply_fn(params, obs)
        greedy_action = jnp.argmax(q_values, axis=1)

        key, subkey = jax.random.split(key)
        random_action = jax.random.randint(subkey, greedy_action.shape, 0, action_dim)
        explore = jax.random.uniform(key, greedy_action.shape) < epsilon
        action = jnp.where(explore, random_action, greedy_action)
        return action, key

    def step_fn(carry, _):
        obs, env_state, keys, params, epsilon = carry

        actions, keys = jax.vmap(get_action, in_axes=(None, 0, 0, None))(params, obs, keys, epsilon)
        next_obs, next_env_state, reward, done, info = jax.vmap(wrapped_step)(env_state, actions)
        first_state = jax.tree.map(lambda x: x[0], next_env_state)

        return (next_obs, next_env_state, keys, params, epsilon), (first_state, done, reward)

    @jax.jit
    def eval_fn(params, reset_keys, epsilon):
        obs, env_state = jax.vmap(wrapped_reset)(reset_keys)

        _, (first_states_history, dones, rewards) = jax.lax.scan(
            step_fn, (obs, env_state, reset_keys, params, epsilon), None, length=max_steps)
        has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
        mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
        masked_rewards = rewards * (1 - mask_after_first_done)
        episodic_returns = jnp.sum(masked_rewards, axis=0)

        first_done = jnp.argmax(dones, axis=0)
        return episodic_returns, first_states_history, first_done

    return eval_fn


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    if isinstance(config.get("TRAIN_MODS"), list):
        config["TRAIN_MODS"] = tuple(config["TRAIN_MODS"])
    if isinstance(config.get("EVAL_MODS"), list):
        config["EVAL_MODS"] = tuple(config["EVAL_MODS"])

    if config.get("PIXEL_BASED", True) and config.get("NUM_ENVS", 1) > 16:
        config["NUM_ENVS"] = 8

    run_name = f"{config["ENV_ID"]}_{config["EXP_NAME"]}_{"oc" if not config["PIXEL_BASED"] else "pixel"}_{config["SEED"]}"

    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
    )

@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array

class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int


def dqn_run(config: dict):

    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"])

    run_name = f'{config["ENV_ID"]}_{config["EXP_NAME"]}_{"oc" if not config["PIXEL_BASED"] else "pixel"}_{config["SEED"]}'
    wandb.init(
        project=config["PROJECT"],
        entity=config["ENTITY"],
        config=config,
        name=run_name,
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])
    key, _network_key, _actor_key, _critic_key = jax.random.split(key, 4)
    # key, obs_sample_key1, obs_sample_key2, obs_sample_key3 = jax.random.split(key, 4)

    # Add Reward Machine
    rm_name = config.get("GAME_RM", None)
    game_rm: GameRM | None = GAME_RM_REGISTRY[rm_name]() if rm_name is not None else None
    use_rm = game_rm is not None

    # env setup
    env = make_env(
        config["ENV_ID"],
        list(config["TRAIN_MODS"]),
        config["PIXEL_BASED"],
        config["NATIVE_DOWNSCALING"],
        False,
        game_rm)()

    @jax.jit
    def vmap_reset(key):
        obs, state = jax.vmap(env.reset)(key)
        return obs.squeeze(), state
    
    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        next_done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze(), state, reward, next_done, info

    key, _rng = jax.random.split(key)
    init_obs, env_state = vmap_reset(jax.random.split(_rng, config["NUM_ENVS"]))


    # INIT BUFFER
    # if use_rm:
    #     buffer = fbx.make_flat_buffer(
    #         max_length=config["BUFFER_SIZE"],
    #         min_length=config["BUFFER_BATCH_SIZE"],
    #         sample_batch_size=config["BUFFER_BATCH_SIZE"],
    #         add_sequences=False,
    #         add_batch_size=config["NUM_ENVS"]
    #     )
    # else: 
    buffer = fbx.make_flat_buffer(
        max_length=config["BUFFER_SIZE"],
        min_length=config["BUFFER_BATCH_SIZE"],
        sample_batch_size=config["BUFFER_BATCH_SIZE"],
        add_sequences=False,
        add_batch_size=(config["NUM_ENVS"] * game_rm.num_states())
    )


    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )
    dummy_rng = jax.random.PRNGKey(0)
    _action = env.action_space().sample(dummy_rng)
    _obs, _env_state = env.reset(dummy_rng)
    _obs, _env_state, _reward, _term, _trunc, _info = env.step(_env_state, _action)
    _done = jnp.logical_or(_term, _trunc)
    _timestep = TimeStep(obs=_obs.squeeze(), action=_action, reward=_reward, next_obs=_obs.squeeze(), done=_done)
    buffer_state = buffer.init(_timestep)

    # INIT NETWORK AND OPTIMIZER
    network = QNetwork(action_dim=env.action_space().n)
    init_x = jnp.zeros(env.observation_space().shape)
    key, _rng = jax.random.split(key)
    network_params = network.init(_rng, init_x)

    def linear_schedule(count):
        frac = 1.0 - (count / config["NUM_UPDATES"])
        return config["LR"] * frac

    lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]
    tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(learning_rate=lr))

    train_state = CustomTrainState.create(
        apply_fn=network.apply,
        params=network_params,
        target_network_params=jax.tree.map(lambda x: jnp.copy(x), network_params),
        tx=tx,
        timesteps=0,
        n_updates=0,
    )

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, t):
        rng_a, rng_e = jax.random.split(
            rng, 2
        )  # a key for sampling random actions and one for picking
        eps = jnp.clip(  # get epsilon
            (
                (config["EPSILON_FINISH"] - config["EPSILON_START"])
                / config["EPSILON_ANNEAL_TIME"]
            )
            * t
            + config["EPSILON_START"],
            config["EPSILON_FINISH"],
        )
        greedy_actions = jnp.argmax(q_vals, axis=-1)  # get the greedy actions
        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),  # sample random actions,
            greedy_actions,
        )
        return chosed_actions

    # TRAINING LOOP
    def _update_step(runner_state, unused):

        train_state, buffer_state, env_state, last_obs, rng = runner_state

        # STEP THE ENV
        rng, rng_a, rng_s = jax.random.split(rng, 3)
        q_vals = network.apply(train_state.params, last_obs)
        action = eps_greedy_exploration(
            rng_a, q_vals, train_state.timesteps
        )  # explore with epsilon greedy_exploration

        obs, env_state, reward, done, info = vmap_step(env_state, action)

        train_state = train_state.replace(
            timesteps=train_state.timesteps + config["NUM_ENVS"]
        )  # update timesteps count

        # BUFFER UPDATE
        # timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done)
        crm = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), info["crm_experiences"])
        buffer_state = buffer.add(buffer_state, crm) 
        # buffer_state = buffer.add(buffer_state, info["crm_experiences"])

        # NETWORKS UPDATE
        def _learn_phase(train_state, rng):

            learn_batch = buffer.sample(buffer_state, rng).experience.first

            # q_next_target = network.apply(
            #     train_state.target_network_params, learn_batch.second.obs
            # )  # (batch_size, num_actions)
            # q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
            q_next_online = network.apply(train_state.params, learn_batch.next_obs)
            next_actions = jnp.argmax(q_next_online, axis=-1)
            q_next_target = network.apply(train_state.target_network_params, learn_batch.next_obs)
            q_next_target = jnp.take_along_axis(q_next_target, next_actions[:, None], axis=-1).squeeze(-1)

            target = (
                learn_batch.reward
                + (1 - learn_batch.done) * config["GAMMA"] * q_next_target
            )

            def _loss_fn(params):
                q_vals = network.apply(params, learn_batch.obs)
                chosen = jnp.take_along_axis(q_vals, learn_batch.action[:, None], axis=-1).squeeze(-1)
                return jnp.mean((chosen - target) ** 2)

            loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            return train_state, loss

        rng, _rng = jax.random.split(rng)
        env_iters = train_state.timesteps // config["NUM_ENVS"]
        is_learn_time = (
            (buffer.can_sample(buffer_state))
            & (  # enough experience in buffer
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            & (  # pure exploration phase ended
                env_iters % config["TRAINING_INTERVAL"] == 0
            )  # training interval
        )
        train_state, loss = jax.lax.cond(
            is_learn_time,
            lambda train_state, rng: _learn_phase(train_state, rng),
            lambda train_state, rng: (train_state, jnp.array(0.0)),  # do nothing
            train_state,
            _rng,
        )

        # update target network
        train_state = jax.lax.cond(
            env_iters % config["TARGET_UPDATE_INTERVAL"] == 0,
            lambda train_state: train_state.replace(
                target_network_params=optax.incremental_update(
                    train_state.params,
                    train_state.target_network_params,
                    config["TAU"],
                )
            ),
            lambda train_state: train_state,
            operand=train_state,
        )
        fired = info["rm_fired_idx"]
        hist = jnp.sum(jax.nn.one_hot(fired, num_transitions), axis=0)
        metrics = {
            "timesteps": train_state.timesteps,
            "updates": train_state.n_updates,
            "loss": loss.mean(),
            "returns": info["returned_episode_returns"].mean(),
            "env_reward": info["env_reward"].mean(),
            "rm_reward": info["rm_reward"].mean(),
            "fired_hist": hist
        }

        runner_state = (train_state, buffer_state, env_state, obs, rng)
        return runner_state, metrics

    def save_and_eval(params, step):
        if config.get("SAVE_PATH") is not None:
            model_path = f'{config["SAVE_PATH"]}/{run_name}/{config["EXP_NAME"]}_{step}_{time.time()}.cleanrl_model'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes([config, params]))
            print(f"model saved to {model_path}")

        eval_mods = config["EVAL_MODS"] if len(config["EVAL_MODS"]) > 0 else config["TRAIN_MODS"]
        eval_configs = [([], "default")]
        for mod in list(eval_mods):
            eval_configs.append(([mod], mod))

        metrics = {}
        for mods_config, mod_label in eval_configs:
            print(f"Evaluating on {mod_label} ...")
            episodic_returns, env_states = evaluate_dqn(
                params,
                partial(make_env, mods=mods_config, pixel_based=config["PIXEL_BASED"],
                        native_downscaling=config["NATIVE_DOWNSCALING"], eval=True, game_rm=game_rm),
                config["ENV_ID"], eval_episodes=10, QNetwork=QNetwork,
                seed=config["SEED"], use_rm=use_rm,
            )
            mean_ret = float(np.mean(jax.device_get(episodic_returns)))
            metrics[mod_label] = mean_ret
            wandb.log({f"eval/episodic_return_{mod_label}": mean_ret}, step=step)

            if config.get("CAPTURE_VIDEO", False):
                renderer = jaxatari.make(config["ENV_ID"], mods=mods_config).renderer
                frames = jax.vmap(renderer.render)(env_states)
                frames = jnp.transpose(frames, (0, 3, 1, 2))
                wandb.log({f"eval/video_{mod_label}": wandb.Video(np.array(frames), fps=30, format="mp4")}, step=step)
                print(f"Video (eval) logged with {frames.shape[0]} frames.")
        return metrics

    # --- CHUNKED TRAINING LOOP (replaces the single big scan) ---
    updates_per_eval = config["EVAL_EVERY"]
    num_chunks = config["NUM_UPDATES"] // updates_per_eval
    num_transitions = len(game_rm.TRANSITIONS)

    @jax.jit
    def train_chunk(runner_state):
        return jax.lax.scan(_update_step, runner_state, None, updates_per_eval)

    key, _rng = jax.random.split(key)
    runner_state = (train_state, buffer_state, env_state, init_obs, _rng)

    start_time = time.time()
    eval_metrics = {}
    for chunk in range(1, num_chunks + 1):
        runner_state, chunk_metrics = train_chunk(runner_state)
        train_state = runner_state[0]
        global_step = int(train_state.timesteps)

        for idx in range(num_transitions):
            wandb.log({f"transitions/t{idx}": float(chunk_metrics["fired_hist"][..., idx].sum())}, step=global_step)
        wandb.log({
            "charts/rm_reward_per_step": float(chunk_metrics["rm_reward"].mean()),
            "charts/avg_episodic_return": float(chunk_metrics["returns"].mean()),
            "losses/td_loss": float(chunk_metrics["loss"].mean()),
            "charts/global_step": global_step,
            "charts/SPS": int(global_step / (time.time() - start_time)),
            "charts/time": time.time() - start_time,
        }, step=global_step)

        if config["EVAL_DURING_TRAIN"]:
            eval_metrics = save_and_eval(train_state.params, global_step)

    print(f"Total train time: {(time.time() - start_time)/60:.2f} minutes.")
    eval_metrics = save_and_eval(train_state.params, train_state.timesteps)
    wandb.finish()

    return eval_metrics
