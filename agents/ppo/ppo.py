# Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax_scan.py
import os
import random
import time
from functools import partial
from typing import Sequence, NamedTuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import jaxtari
from jaxtari.wrappers import NormalizeObservationWrapper, ObjectCentricWrapper, PixelObsWrapper, AtariWrapper, LogWrapper, FlattenObservationWrapper
from jaxtari import spaces
from agents.ppo.ppo_eval import evaluate
from agents.run_utils import eval_mod_configs, plan_chunks
from rtpt import RTPT

def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False):
    assert mods is None or isinstance(mods, list), "mods must be None or a list of strings"
    if mods is not None and len(mods) == 0:
        mods = None
    if not eval and mods is not None and len(mods) > 0:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        env = jaxtari.make(env_id, mods=mods)
        env = AtariWrapper(
                env,
                sticky_actions=0.0,
                episodic_life=not eval, # only active during training 
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
        env = LogWrapper(env)
        return env
    return thunk


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x

class MLP_Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        # 1. Hidden Layer
        x = nn.Dense(
            461,  # Hidden size H=461
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)

        # 2. Output Layer (matches the last layer of the CNN)
        x = nn.Dense(
            512,  # Output size
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)  # The CNN's last layer also has a ReLU
        return x

class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


class AgentParams(NamedTuple):
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array

def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    pixel_based = config["PIXEL_BASED"]
    env_id = config["ENV_ID"]

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

    # do not modify the seeding
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    num_envs = config["NUM_ENVS"]
    num_steps = config["NUM_STEPS"]
    num_minibatches = config["NUM_MINIBATCHES"]
    update_epochs = config["UPDATE_EPOCHS"]

    env = make_env(env_id, list(config.get("TRAIN_MODS") or []), pixel_based, config.get("NATIVE_DOWNSCALING", True), False)()
    assert isinstance(env.action_space(), spaces.Discrete), "only discrete action space is supported"
    action_dim = env.action_space().n
    obs_shape = env.observation_space().shape
    if pixel_based:
        obs_shape = obs_shape[:-1]  # drop the trailing grayscale channel

    batch_size = num_envs * num_steps
    scan_steps, num_chunks, steps_per_chunk = plan_chunks(config["TOTAL_TIMESTEPS"], batch_size, config.get("SCAN_STEPS", 1000))
    num_updates = scan_steps * num_chunks

    def reset_envs(rng):
        obs, state = jax.vmap(env.reset)(rng)
        return obs.reshape(num_envs, *obs_shape), state

    def step_envs(state, action):
        obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        done = jnp.logical_or(terminated, truncated)
        return obs.reshape(num_envs, *obs_shape), state, reward, done, info

    def linear_schedule(count):
        # anneal learning rate linearly after one update which contains
        # (NUM_MINIBATCHES * UPDATE_EPOCHS) gradient steps
        frac = 1.0 - (count // (num_minibatches * update_epochs)) / num_updates
        return config["LEARNING_RATE"] * frac

    network = Network() if pixel_based else MLP_Network()
    actor = Actor(action_dim=action_dim)
    critic = Critic()

    # same root split as dqn/c51/pqn, so env resets for a given seed match across agents
    key, init_key = jax.random.split(key, 2)
    network_key, actor_key, critic_key = jax.random.split(init_key, 3)
    network_params = network.init(network_key, jnp.zeros((1, *obs_shape)))
    hidden_example = network.apply(network_params, jnp.zeros((1, *obs_shape)))
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            network_params=network_params,
            actor_params=actor.init(actor_key, hidden_example),
            critic_params=critic.init(critic_key, hidden_example),
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LEARNING_RATE"], eps=1e-5
            ),
        ),
    )

    def get_action_and_value(params, obs, key):
        """sample action, calculate value and logprob"""
        hidden = network.apply(params.network_params, obs)
        logits = actor.apply(params.actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = critic.apply(params.critic_params, hidden)
        return action, logprob, value.squeeze(1), key

    def get_action_and_value2(params, x, action):
        """calculate value, logprob of supplied `action`, and entropy"""
        hidden = network.apply(params.network_params, x)
        logits = actor.apply(params.actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = critic.apply(params.critic_params, hidden).squeeze()
        return logprob, entropy, value

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns):
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if config["NORM_ADV"]:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - config["CLIP_COEF"], 1 + config["CLIP_COEF"])
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - config["ENT_COEF"] * entropy_loss + v_loss * config["VF_COEF"]
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    obs, env_state = reset_envs(jax.random.split(key, num_envs))
    carry = (
        agent_state,
        env_state,
        obs,
        jnp.zeros(num_envs, dtype=jnp.bool_),  # done flags of the last step
        key,
        jnp.array(0, dtype=jnp.int32),  # env steps taken
    )

    def ppo_update(carry):
        """One rollout of num_steps per env, GAE, and update_epochs of minibatch SGD."""
        agent_state, env_state, obs, last_done, key, global_step = carry

        def rollout_step(c, _):
            env_state, last_obs, last_done, key = c
            action, logprob, value, key = get_action_and_value(agent_state.params, last_obs, key)
            next_obs, next_state, reward, next_done, info = step_envs(env_state, action)
            storage = Storage(
                obs=last_obs,
                actions=action,
                logprobs=logprob,
                dones=last_done,
                values=value,
                rewards=reward,
                returns=jnp.zeros_like(reward),
                advantages=jnp.zeros_like(reward),
            )
            return (next_state, next_obs, next_done, key), (storage, info)

        (env_state, next_obs, next_done, key), (storage, infos) = jax.lax.scan(
            rollout_step, (env_state, obs, last_done, key), None, length=num_steps
        )

        # GAE, computed backwards through the rollout
        next_value = critic.apply(
            agent_state.params.critic_params, network.apply(agent_state.params.network_params, next_obs)
        ).squeeze(1)
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0).astype(jnp.float32)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)

        def gae_step(advantages, inp):
            nextdone, nextvalues, curvalues, reward = inp
            nextnonterminal = 1.0 - nextdone
            delta = reward + config["GAMMA"] * nextvalues * nextnonterminal - curvalues
            advantages = delta + config["GAMMA"] * config["GAE_LAMBDA"] * nextnonterminal * advantages
            return advantages, advantages

        _, advantages = jax.lax.scan(
            gae_step, jnp.zeros((num_envs,)), (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
        )
        storage = storage.replace(advantages=advantages, returns=advantages + storage.values)

        def update_epoch(carry, _):
            agent_state, key = carry
            key, perm_key = jax.random.split(key)
            flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), storage)
            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            shuffled = jax.tree.map(
                lambda x: jnp.reshape(jax.random.permutation(perm_key, x), (num_minibatches, -1) + x.shape[1:]),
                flat,
            )

            def update_minibatch(agent_state, mb):
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params, mb.obs, mb.actions, mb.logprobs, mb.advantages, mb.returns
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl)

            agent_state, losses = jax.lax.scan(update_minibatch, agent_state, shuffled)
            return (agent_state, key), losses

        (agent_state, key), losses = jax.lax.scan(update_epoch, (agent_state, key), None, length=update_epochs)
        losses = jax.tree.map(lambda x: x[-1, -1], losses)
        learning_rate = agent_state.opt_state[1].hyperparams["learning_rate"]

        new_carry = (agent_state, env_state, next_obs, next_done, key, global_step + batch_size)
        return new_carry, (infos, losses, learning_rate)

    def save_and_eval(params, step_count):
        if config.get("SAVE_PATH") is not None:
            save_dir = os.path.join(config["SAVE_PATH"], run_name)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model')
            with open(model_path, "wb") as f:
                f.write(
                    flax.serialization.to_bytes(
                        [config, [params.network_params, params.actor_params, params.critic_params]]
                    )
                )
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")
        metrics = {}
        for mods_cfg, mod_label in eval_mod_configs(config):
            episodic_returns, env_states = evaluate(
                params,
                partial(
                    make_env,
                    mods=mods_cfg,
                    pixel_based=pixel_based,
                    native_downscaling=config.get("NATIVE_DOWNSCALING", True),
                    eval=True,
                ),
                env_id,
                eval_episodes=10,
                Model=(Network, Actor) if pixel_based else (MLP_Network, Actor),
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
        return jax.lax.scan(lambda c, _: ppo_update(c), carry, None, length=scan_steps)

    print("[ppo] start compile...")
    compile_start = time.perf_counter()
    compiled = jax.jit(train_chunk, donate_argnums=(0,)).lower(carry).compile()
    print(f"[ppo] compilation time: {time.perf_counter() - compile_start:.2f}s")

    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=num_chunks)
    rtpt.start()
    run_time = time.perf_counter()
    print(f"[ppo] starting training: {num_chunks} chunks x {steps_per_chunk} env steps")
    for chunk in range(num_chunks):
        rtpt.step()
        if config["EVAL_DURING_TRAIN"] and chunk > 0 and chunk % config["EVAL_EVERY"] == 0:
            save_and_eval(carry[0].params, int(carry[-1]))

        chunk_start = time.perf_counter()
        carry, (infos, (loss, pg_loss, v_loss, entropy_loss, approx_kl), learning_rate) = compiled(carry)
        global_step = int(carry[-1])
        metrics = {
            "charts/avg_episodic_return": float(infos["returned_episode_returns"][-1].mean()),
            "charts/avg_episodic_length": float(infos["returned_episode_lengths"][-1].mean()),
            "charts/learning_rate": learning_rate[-1].item(),
            "losses/value_loss": v_loss[-1].item(),
            "losses/policy_loss": pg_loss[-1].item(),
            "losses/entropy": entropy_loss[-1].item(),
            "losses/approx_kl": approx_kl[-1].item(),
            "losses/loss": loss[-1].item(),
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(steps_per_chunk / (time.perf_counter() - chunk_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }
        wandb.log(metrics, step=global_step)
        print(
            f"[ppo] chunk {chunk + 1}/{num_chunks} | global_step {global_step} | "
            f"avg_return {metrics['charts/avg_episodic_return']:.2f} | "
            f"avg_length {metrics['charts/avg_episodic_length']:.2f} | "
            f"loss {metrics['losses/loss']:.4f} | entropy {metrics['losses/entropy']:.4f} | "
            f"SPS {metrics['charts/SPS']}"
        )

    eval_metrics = save_and_eval(carry[0].params, global_step + 1)
    wandb.finish()
    return eval_metrics
