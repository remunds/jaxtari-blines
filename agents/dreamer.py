'''
    DreamerV3 re-implementation
    author: Rafael Rodriguez-Sanchez
    date: October 2024
    modified: October 2025
    https://github.com/rafarodsa/dreamer-v3-purejax
'''
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" 
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


from functools import partial
from typing import Tuple
from collections.abc import Callable

import wandb
import argparse
import jax
import jax.numpy as jnp
if not hasattr(jax, 'tree_map'):
    import jax.tree_util
    jax.tree_map = jax.tree_util.tree_map
import chex
import optax
import flax.nnx as nnx
import flashbax as fbx
import jaxatari
from typing import Any
from tqdm import tqdm
from jaxmodels_nnx import build_model
from utils.replayx import make_trajectory_buffer
from utils.jaxutils import symlog, symexp, scale_by_momentum, inv_twohot
from utils.jaxutils import twohot as base_twohot
from utils.loggers import JSONLogger, MultiLogger
from omegaconf import OmegaConf as oc
from utils.printarr import printarr
from omegaconf import open_dict

# oc.register_new_resolver("eval", eval)
tree_map = jax.tree.map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
vmapped_twohot = jax.vmap(base_twohot, in_axes=(0,None))
twohot = jax.vmap(vmapped_twohot, in_axes=(0, None))
swapaxes = lambda y : tree_map(lambda x: x.swapaxes(0,1), y)

@chex.dataclass(frozen=True)
class DreamerParams:
    v_min : int
    v_max : int
    n_bins : int
    lambd : float
    gamma : float
    entropy_reg : float
    imagination_horizon : int
    lr : float
    warmup: int
    grad_clip : float
    unimix : float
    free_nats: float
    rep_loss_const: float
    reconst_loss_const : float
    dynamics_loss_const : float
    critic_loss_const: float
    critic_replay_loss_const: float
    critic_ema_reg: float
    critic_ema_tau: float
    actor_loss_scale: float
    actor_entropy_reg: float
    actor_perc_low : float
    actor_perc_high: float
    actor_ret_limit: float
    actor_ret_decay: float
    n_actions : int
    replay_ratio : float
    rollout_length : int
    action_repeat : int

@chex.dataclass(frozen=True)
class DreamerState:
    buffer_state : fbx.prioritised_trajectory_buffer.PrioritisedTrajectoryBufferState
    optimizer: optax.OptState
    rng: chex.Array
    n_updates: chex.Array
    timesteps: chex.Array
    dummy_logs : dict
    rssm : Tuple[nnx.GraphDef, nnx.Param]
    agent: Tuple[nnx.GraphDef, nnx.Param]
    last_trained : int = 0

@chex.dataclass(frozen=True)
class RecurrentLatentState:
    z : chex.Array
    h : chex.Array

@chex.dataclass(frozen=True)
class Episode:
    obs: chex.Array # (T, S)
    action: chex.Array  # (T, A)
    reward: chex.Array  # (T,)
    done: chex.Array #  # (T,)
    is_first: chex.Array # marks the start of the trajectory
    latent : RecurrentLatentState

@chex.dataclass(frozen=True)
class ImaginedEpisode:
    latent : RecurrentLatentState
    action : chex.Array
    reward : chex.Array
    done : chex.Array

@chex.dataclass(frozen=True)
class EnvConfig:
    obs: chex.Array
    action: chex.Array
    done: chex.Array
    reward: chex.Array
    env_reset: Callable
    env_step: Callable
    n_actions: int
    n_envs: int
    # base_env : gymnax.environments.environment.Environment
    # env_params : gymnax.EnvParams
    base_env : Any   # Let this accept the JAXtari env object

class RSSM(nnx.Module):
    def __init__(
            self,
            rngs,
            config,
        ):
        self.config = config
        self.pixels = self.config.pixels
        self.latent_dim = config.latent_categoricals * config.latent_classes
        self.latent_categoricals = config.latent_categoricals
        self.latent_classes = config.latent_classes
        self.initial_latent = jnp.zeros((self.config.latent_categoricals, self.config.latent_classes))
        self.initial_memory = jnp.zeros((self.config.memory_dim,))
        config.embed.input_dim = self.latent_dim
        self.embed_latent = build_model(config.embed, rngs=rngs)
        config.embed.input_dim = config.n_actions
        self.embed_action = build_model(config.embed, rngs=rngs)
        config.embed.input_dim = config.memory_dim
        self.embed_memory =  build_model(config.embed, rngs=rngs)
        self.dynamics_cell = build_model(config.rnn, rngs=rngs)
        self.dynamics = build_model(config.dynamics, rngs=rngs)
        self.continuation = build_model(config.cont, rngs=rngs)
        self.reward = build_model(config.reward, rngs=rngs)

        if config.pixels:
            self.config.decoder_pixel.output_shape = self.config.input_dim
            self.config.obs_embed_pixel.input_shape = self.config.input_dim
            self.config.decoder_pixel.input_dim = self.config.state_dim
            self.obs_embed = build_model(config.obs_embed_pixel, rngs)
            self.decoder = build_model(config.decoder_pixel, rngs)
        else:
            self.config.decoder.output_dim = self.config.input_dim
            self.config.obs_embed.input_dim = self.config.input_dim
            self.obs_embed = build_model(config.obs_embed, rngs)
            self.decoder = build_model(config.decoder, rngs)

        self.encoder = build_model(config.encoder, rngs)
        self.rngs = rngs

    def init_carry(self):
        return RecurrentLatentState(
            h=self.initial_memory,
            z=self.initial_latent
        )

    def latent_to_vector(self, latent):
        batch_dims = latent.h.shape[:-1]
        z = latent.z.reshape(*batch_dims, -1)
        return jnp.concatenate([z, latent.h], axis=-1)

    def preprocess(self, obs):
        return obs if self.pixels else symlog(obs)

    def observe(
            self,
            obs_embed : chex.Array,
            prev_action : chex.Array, # onehot encoded or continuous vector
            prev_latent : RecurrentLatentState,
            is_first : chex.Array,
            params : DreamerParams,
            rng
        ):

        # embed
        batch_dims = prev_latent.h.shape[:-1]

        # reinitialize latents
        prev_action = jax.lax.select(
            jnp.broadcast_to(is_first[:, None], shape=prev_action.shape).astype(jnp.bool),
            jnp.zeros_like(prev_action),
            prev_action
        )

        init_carry = tree_map(lambda x: jnp.broadcast_to(x, shape=(*batch_dims, *x.shape)), self.init_carry())
        h = is_first[:, None] * init_carry.h + (1-is_first[:, None]) * prev_latent.h
        z = is_first[:, None, None] * init_carry.z + (1-is_first[:, None, None]) * prev_latent.z

        prev_latent = RecurrentLatentState(
            h=h,
            z=z
        )

        action_embed = self.embed_action(prev_action)
        latent_embed = self.embed_latent(prev_latent.z.reshape(*batch_dims, -1))
        memory_embed = self.embed_memory(prev_latent.h)
        embed = jnp.concatenate([memory_embed, latent_embed, action_embed], axis=-1)
        new_memory, _ = self.dynamics_cell(embed, prev_latent.h)

        new_latent_logits = self.encoder(jnp.concatenate([obs_embed, new_memory], axis=-1))

        new_latent_probs = nnx.softmax(new_latent_logits.reshape(
            *obs_embed.shape[:-1],
            self.config.latent_categoricals,
            self.config.latent_classes
        ), axis=-1)

        uniform = jnp.ones_like(new_latent_probs) / self.config.latent_classes

        new_latent_logits = jnp.log((new_latent_probs * (1-params.unimix) + uniform * params.unimix))
        new_latent = jax.random.categorical(
            rng,
            new_latent_logits,
        )
        
        new_latent = nnx.one_hot(
            new_latent,
            self.config.latent_classes,
            axis=-1
        )
        new_latent = sg(new_latent) + new_latent_probs - sg(new_latent_probs) # ST gradient estimator

        return RecurrentLatentState(
            z=new_latent,
            h=new_memory
        ), new_latent_logits

    def imagine(
            self,
            prev_action : chex.Array, # onehot encoded or continuous vector
            prev_latent : RecurrentLatentState,
            params : DreamerParams,
            rng
        ):
        # embed
        batch_dims = prev_latent.z.shape[:-2]
        # reinitialize latents

        action_embed = self.embed_action(prev_action)
        latent_embed = self.embed_latent(prev_latent.z.reshape(*batch_dims, -1))
        memory_embed = self.embed_memory(prev_latent.h)
        embed = jnp.concatenate([memory_embed, latent_embed, action_embed], axis=-1) # one memory step

        new_memory, _ = self.dynamics_cell(embed, prev_latent.h)
        new_latent_probs = nnx.softmax(
            self.dynamics(new_memory).reshape(
                *batch_dims,
                self.config.latent_categoricals,
                self.config.latent_classes
            ),
            axis=-1
        )

        uniform = jnp.ones_like(new_latent_probs) / self.config.latent_classes

        new_latent_logits = jnp.log((new_latent_probs * (1-params.unimix) + uniform * params.unimix))
        new_latent = jax.random.categorical(
            rng,
            new_latent_logits,
            axis=-1
        )
        
        new_latent = nnx.one_hot(
            new_latent,
            self.latent_classes,
            axis=-1
        )

        latent_state = RecurrentLatentState(
            z=new_latent,
            h=new_memory
        )
        
        return latent_state

class ActorCritic(nnx.Module):
    def __init__(self, config, rngs):
        self.config = config
        self.actor = build_model(config.actor, rngs)
        self.critic = build_model(config.critic, rngs)
        critic_graphdef, params = nnx.split(self.critic)
        self.target_critic = nnx.merge(critic_graphdef, tree_map(jnp.copy, params))
        self.ret_normalizer = jnp.zeros((1,))

    def __call__(self, latent_state):
        state = jnp.concatenate([latent_state.memory, latent_state.latent], axis=-1)
        return self.actor(state)

    def update_slow_params(self, params, returns):
        return_perc = jnp.percentile(returns.ravel(), jnp.array([params.actor_perc_low, params.actor_perc_high]))
        self.ret_normalizer = (return_perc[1]-return_perc[0]) * params.actor_ret_decay + self.ret_normalizer * (1 - params.actor_ret_decay)
        _, critic_params = nnx.split(self.critic)
        nnx.update(
            self.target_critic,
            optax.incremental_update(
                critic_params,
                nnx.state(self.target_critic),
                params.critic_ema_tau
            )
        )
    
    @property
    def ret_norm(self):
        return self.ret_normalizer


def build_dreamer(
    params : DreamerParams,
    config,
    env_config
):
    replay_buffer = make_trajectory_buffer(
        add_batch_size=env_config.n_envs,
        sample_batch_size=int(config.replay_buffer.batch_size),
        sample_sequence_length=config.replay_buffer.batch_length,
        period=1,
        min_length_time_axis=config.replay_buffer.batch_length,
        max_size=int(config.replay_buffer.buffer_size),
    )

    replay_buffer = replay_buffer.replace(
        init=jax.jit(replay_buffer.init),
        add=jax.jit(replay_buffer.add, donate_argnums=0),
        sample=jax.jit(replay_buffer.sample),
        can_sample=jax.jit(replay_buffer.can_sample),
        update=jax.jit(replay_buffer.update)
    )

    rng, rng_rssm, rng_agent = jax.random.split(jax.random.PRNGKey(config.seed), 3)
    config.rssm.pixels = len(env_config.obs.shape) == 3
    config.rssm.input_dim = env_config.obs.shape if config.rssm.pixels else env_config.obs.shape[0]
    rssm = RSSM(rngs=nnx.Rngs(rng_rssm), config=config.rssm)
    agent = ActorCritic(config.actor_critic, rngs=nnx.Rngs(rng_agent))

    n_params = sum(jax.tree.leaves(jax.tree.map(jnp.size, nnx.state(rssm, nnx.Param)))) + \
        sum(jax.tree.leaves(jax.tree.map(jnp.size, nnx.state(agent.actor, nnx.Param)))) + \
        sum(jax.tree.leaves(jax.tree.map(jnp.size, nnx.state(agent.critic, nnx.Param))))
    
    n_params = {
        'rssm':  sum(jax.tree.leaves(jax.tree.map(jnp.size, nnx.state(rssm, nnx.Param)))),
        'actor':  sum(jax.tree.leaves(jax.tree.map(jnp.size, nnx.state(agent.actor, nnx.Param)))),
        'critic': sum(jax.tree.leaves(jax.tree.map(jnp.size, nnx.state(agent.critic, nnx.Param))))
    }

    print(f'Trainable parameters: {tree_map(lambda x: x/1e6, n_params)} M')
    # print(tree_map(lambda x: sum(jax.tree.leaves(x))/1e3, tree_map(jnp.size, nnx.state(rssm, nnx.Param)), is_leaf=lambda x: isinstance(x, dict)))


    _sample = Episode(
        obs=env_config.obs,
        action=env_config.action,
        reward=env_config.reward,
        done=env_config.done.astype(jnp.float32),
        latent=rssm.init_carry(),
        is_first=jnp.ones_like(env_config.done, dtype=jnp.float32)
    )
    buffer_state = replay_buffer.init(_sample)

    tx = optax.chain(
        optax.adaptive_grad_clip(params.grad_clip),
        optax.scale_by_rms(0.999, 1e-20),
        scale_by_momentum(0.9, nesterov=False),
        optax.scale(-params.lr)
    ) # LaProp optimizer
    # tx = optax.adamw(1e-4)

    optstate = tx.init(
        (
            nnx.state(rssm, nnx.Param),
            nnx.state(agent.actor, nnx.Param),
            nnx.state(agent.critic, nnx.Param)
        )
    )

    agent = DreamerState(
        buffer_state=buffer_state,
        optimizer=optstate,
        rng=rng,
        rssm=nnx.split(rssm),
        agent=nnx.split(agent),
        dummy_logs={},
        timesteps=0,
        n_updates=0
    )

    return agent, tx, replay_buffer


def policy(
        rssm,
        actor,
        obs,
        carry,
        is_first,
        rng,
        params
):  
    rng, rng_obs, rng_a = jax.random.split(rng, 3)
    # observe: preprocess & embed
    obs = obs if rssm.pixels else symlog(obs)
    obs_embed = rssm.obs_embed(obs)
    # observe: compute next latent
    prev_latent, prev_action = carry
    prev_action = nnx.one_hot(prev_action, params.n_actions+1)[..., :-1]
    new_latent, _ = rssm.observe( # this expects a batch dim
        obs_embed[None],
        prev_action[None],
        tree_map(lambda x: x[None], prev_latent),
        is_first[None],
        params,
        rng_obs,
    )

    new_latent = tree_map(lambda x: x[0], new_latent)
    # compute new action with new latent
    z_h = rssm.latent_to_vector(new_latent)
    action_logits = actor(z_h)
    uniform = jnp.ones_like(action_logits) / action_logits.shape[-1]
    action_logits = jnp.log(nnx.softmax(action_logits, axis=-1) * (1-params.unimix) + uniform * params.unimix)
    # sample action
    action = jax.random.categorical(
        rng_a,
        action_logits,
    )
    carry = (new_latent, action)
    return new_latent, action, carry


def categorical_kl(p1_logits, p2_logits):
    logp1 = nnx.log_softmax(p1_logits, axis=-1)
    logp2 = nnx.log_softmax(p2_logits, axis=-1)
    kl = (jnp.exp(logp1) * (logp1-logp2)).sum(-1)
    return kl

def categorical_ent(p1_logits):
    logp1 = nnx.log_softmax(p1_logits, axis=-1)
    return -(jnp.exp(logp1) * logp1).sum(-1)

def get_dreamer_learn_fn(
        params: DreamerParams,
        optimizer, # optimizer fn
        replay_buffer, # replay buffer fn
        env_config,
        update_ratio, # number of updates per env step,
        training_starts
    ):
    
    def _train(state, i):
        # rollout the envs
        agent, env_state, rng, last_step = state
        rng, rng_s = jax.random.split(rng)
        actor = nnx.merge(*agent.agent).actor

        @partial(jax.vmap, in_axes=(0, 0, 0))
        def rollout(env_state, last_step, rng):
            def _rollout(carry, unused):
                env_state, rng, last_step = carry
                last_obs, last_carry, last_done = last_step
                rng, rng_step, rng_reset, rng_a = jax.random.split(rng, 4)
                prev_latent, prev_action = last_carry
                obs_step, env_state_step, reward, done, info = env_config.env_step(rng_step, env_state, prev_action) # execute previously computed action
                obs_res, env_state_res = env_config.env_reset(rng_reset)
                # update hidden
                env_state = tree_map(lambda reset, step: jax.lax.select(
                    last_done.astype(jnp.bool),
                    reset,
                    step
                ), env_state_res, env_state_step)

                obs = tree_map(lambda reset, step : jax.lax.select(
                    last_done.astype(jnp.bool),
                    reset,
                    step
                ), obs_res, obs_step)

                done = jax.lax.select(
                    last_done.astype(jnp.bool),
                    jnp.zeros_like(done),
                    done
                )

                latent, action, carry = policy(
                    nnx.merge(*agent.rssm),
                    actor,
                    obs,
                    last_carry,
                    last_done,
                    rng_a,
                    params
                )

                done = done.astype(jnp.float32)
                transition = Episode( # t
                    obs=obs,# o_t
                    action=sg(action), # a_t
                    reward=reward.astype(jnp.float32), # r_t
                    done=done, #d_t
                    latent=sg(latent), # h_t, z_t
                    is_first=last_done
                )
                return (env_state, rng, (obs, carry, done)), (transition, info)
        
            (env_state, rng, last_step), (experience, info) = jax.lax.scan(
                    _rollout,
                    (env_state, rng, last_step),
                    None,
                    length=params.rollout_length
            )
            return env_state, last_step, experience, info

        env_state, last_step, experience, info = rollout(env_state, last_step, jax.random.split(rng_s, env_config.n_envs))

        # add batch
        # experience = jax.tree.map(lambda x: x.swapaxes(1,0), experience)
        buffer_state = replay_buffer.add(agent.buffer_state, experience)

        # loss functions
        agent = agent.replace(
            timesteps=agent.timesteps+env_config.n_envs*params.rollout_length,
            buffer_state=buffer_state
        )
        num_updates = (agent.timesteps - agent.last_trained) * update_ratio

        def _update_dreamer(iter_state):
            half = jnp.linspace(params.v_min, 0, (params.n_bins-1)// 2 + 1, dtype=jnp.float32)
            half = symexp(half)
            bins = jnp.concatenate([half, -half[:-1][::-1]], 0)

            agent, rng, num_updates, _, _ = iter_state
            def _loss(
                    model_params,
                    model_rest_params,
                    model_graphs,
                    target_critic,
                    ret_norm,
                    data,
                    rng
                ):

                rssm_params, actor_params, critic_params = list(zip(model_graphs, model_params, model_rest_params))
                rssm = nnx.merge(*rssm_params)
                actor = nnx.merge(*actor_params)
                critic = nnx.merge(*critic_params)

                ### world model losses
                # replay
                def _observe_step(carry, transition, rssm):
                    prev_latent, prev_action, rng = carry
                    rssm = nnx.merge(*rssm)
                    obs = rssm.preprocess(transition.obs)
                    obs_embed = rssm.obs_embed(obs)
                    rng, rng_obs = jax.random.split(rng)
                    latent, posterior_logits = rssm.observe(
                        obs_embed,
                        nnx.one_hot(prev_action, params.n_actions+1)[..., :-1],
                        prev_latent,
                        transition.is_first,
                        params,
                        rng_obs,
                    )
                    return (latent, transition.action, rng), (latent, posterior_logits)
                
                _data = tree_map(lambda x: x[:, 1:], data)
                batch_dims = _data.reward.shape
                (last_latent, last_action, rng), (replayed_latents, posteriors_logits) = jax.lax.scan(
                    partial(_observe_step, rssm=rssm_params),
                    (*tree_map(lambda x: x[:, 0], (data.latent, data.action)), rng),
                    tree_map(lambda x: x.swapaxes(0, 1), _data)
                )

                replayed_latents = swapaxes(replayed_latents)
                posteriors_logits = swapaxes(posteriors_logits)

                priors_probs = nnx.softmax(rssm.dynamics(replayed_latents.h).reshape(
                    *batch_dims,
                    rssm.latent_categoricals,
                    rssm.latent_classes
                ), -1)
                uniform = jnp.ones_like(priors_probs) / priors_probs.shape[-1]
                prior_logits = jnp.log(params.unimix * uniform + (1-params.unimix) * priors_probs)

                z_h = rssm.latent_to_vector(replayed_latents)
                reward_pred = rssm.reward(z_h) # logits
                cont_pred = rssm.continuation(z_h).squeeze()
                recons_obs = rssm.decoder(z_h)

                recons_loss = optax.l2_loss(recons_obs, rssm.preprocess(_data.obs)).reshape(*_data.reward.shape, -1).mean(-1)
                cont_loss = optax.sigmoid_binary_cross_entropy(
                    cont_pred,
                    (1-_data.done) * params.gamma
                )

                reward_loss = optax.softmax_cross_entropy(
                    reward_pred, twohot(_data.reward, bins)
                )

                prediction_loss = recons_loss + cont_loss + reward_loss

                dynamics_loss = jnp.maximum(params.free_nats, categorical_kl(sg(posteriors_logits), prior_logits).sum(-1))
                rep_loss = jnp.maximum(params.free_nats, categorical_kl(posteriors_logits, sg(prior_logits)).sum(-1))

                wm_loss = (params.rep_loss_const * rep_loss + params.dynamics_loss_const*dynamics_loss + params.reconst_loss_const * prediction_loss).mean()

                rep_logs = {
                    'rep/prior_ent': categorical_ent(prior_logits).sum(-1).mean(),
                    'rep/post_ent': categorical_ent(posteriors_logits).sum(-1).mean(),
                    'rep/reward_loss': reward_loss.mean(),
                    'rep/cont_loss': cont_loss.mean(),
                    'rep/reconst_loss': recons_loss.mean(),
                }
                ####

                ############################ imagine #######
                def _imagine_step(
                        carry,
                        unused,
                        rssm_params,
                        actor_params
                    ):
                    latent, action, rng = carry
                    rng, rng_o, rng_a = jax.random.split(rng, 3)
                    rssm = nnx.merge(*rssm_params)
                    actor = nnx.merge(*actor_params)
                    latent = rssm.imagine(
                        nnx.one_hot(action, params.n_actions+1)[..., :-1],
                        latent,
                        params,
                        rng_o,
                    )
                    action_logits = actor(rssm.latent_to_vector(latent))
                    uniform = jnp.ones_like(action_logits) / action_logits.shape[-1]
                    action_logits = jnp.log(uniform * params.unimix + nnx.softmax(action_logits, axis=-1)*(1-params.unimix))
                    action = jax.random.categorical(rng_a, action_logits)
                    return (latent, action, rng), (latent, action)
                
                B, T = _data.is_first.shape[:2]
                rng, rng_a, rng_img = jax.random.split(rng, 3)
                initial_latents = tree_map(lambda x: x.reshape(B*T, *x.shape[2:]), replayed_latents)
                initial_actions = jax.random.categorical(rng_a, actor(rssm.latent_to_vector(initial_latents)))
                initial_actions = initial_actions.reshape(B*T)
                initial_rew = _data.reward.reshape(B*T)
                initial_cont = 1-_data.done.reshape(B*T)
                _, (imagined_latents, imagined_actions) = jax.lax.scan(
                    partial(_imagine_step, rssm_params=rssm_params, actor_params=actor_params),
                    (initial_latents, initial_actions, rng_img),
                    None,
                    length=params.imagination_horizon
                )
                imagined_latents = swapaxes(imagined_latents)
                imagined_actions = swapaxes(imagined_actions)
                img_states = rssm.latent_to_vector(imagined_latents)
                rewards = inv_twohot(nnx.softmax(rssm.reward(img_states), axis=-1), bins) # mean reward
                cont = nnx.sigmoid(rssm.continuation(img_states).squeeze()) # cont flag
                (
                    imagined_latents,
                    imagined_actions,
                    imagined_rewards,
                    imagined_cont
                ) = tree_map(lambda x, y: jnp.concatenate([x[:, None], y], axis=1),
                             (initial_latents, initial_actions, initial_rew, initial_cont),
                             (sg(imagined_latents), sg(imagined_actions), rewards, cont)
                            )
                
                ##############################
                # Imagination actor critic
                img_states = sg(rssm.latent_to_vector(imagined_latents))
                values_logits = critic(img_states)
                slow_values = inv_twohot(nnx.softmax(target_critic(img_states), axis=-1), bins)
                target_values = inv_twohot(nnx.softmax(values_logits, -1), bins)
                weights = sg(jnp.cumprod(imagined_cont, axis=-1))

                # \lambda-Return estimate
                def lambda_return_step(
                        next_return,
                        transition,
                        lambda_=0.95
                    ):
                    reward, value, gamma = transition
                    returns = reward + gamma * ((1-lambda_)*value + lambda_*next_return)
                    return returns, returns

                _, ret = jax.lax.scan(
                    partial(lambda_return_step, lambda_=params.lambd),
                    target_values[:, -1],
                    swapaxes((imagined_rewards[:, 1:], target_values[:, 1:], imagined_cont[:, 1:])),
                    reverse=True
                )

                ret = swapaxes(ret)
                #### Imagination critic loss
                img_critic_loss = (optax.softmax_cross_entropy(
                    values_logits[:, :-1],
                    twohot(sg(ret), bins)
                ) + params.critic_ema_reg * optax.softmax_cross_entropy(
                    values_logits[:, :-1],
                    sg(twohot(slow_values, bins))[:, :-1]
                )) * weights[:, :-1]

                img_critic_loss = img_critic_loss.mean()

                critic_logs = {
                    'imgcritic/loss': img_critic_loss,
                    'imgcritic/values':  target_values.mean(),
                    'imgcritic/length': weights.sum(-1).mean(),
                    'imgcritic/returns': ret.mean(),
                    'imgcritc/rewards': imagined_rewards.sum(-1).mean(),
                }

                #### Actor loss
                pi_probs = nnx.softmax(actor(img_states), axis=-1)
                uniform = jnp.ones_like(pi_probs) / pi_probs.shape[-1]
                pi_probs = params.unimix * uniform + (1-params.unimix) * pi_probs
                pi_ent = -(pi_probs * jnp.log(pi_probs)).sum(-1)
                adv = sg((ret - target_values[:, :-1])/jnp.maximum(params.actor_ret_limit, ret_norm))
                logpi = jnp.log(pi_probs[
                    jnp.arange(imagined_actions.shape[0])[:, None],
                    jnp.arange(imagined_actions.shape[1])[None],
                    sg(imagined_actions)
                ])
                # logpi = (jnp.log(pi_probs) * nnx.one_hot(sg(imagined_actions), params.n_actions)).sum(-1)
                actor_loss = (weights[:, :-1] * -(adv * logpi[:, :-1] + params.actor_entropy_reg * pi_ent[:, :-1])).sum(-1).mean()
                actor_logs = {
                    'actor/entropy': pi_ent.mean(),
                    'actor/actor_loss': actor_loss,
                    'actor/adv': adv.mean()
                }

                ### replay critic
                replay_value_logits = critic(z_h)
                replay_slow_values = inv_twohot(nnx.softmax(target_critic(z_h), axis=-1), bins)
                replay_target_values = ret[:, 0].reshape(*_data.reward.shape)
                discount = (1-_data.done) * params.gamma
                _, replay_ret = jax.lax.scan(
                    partial(lambda_return_step, lambda_=params.lambd),
                    replay_target_values[:, -1],
                    swapaxes((_data.reward[:, 1:], replay_target_values[:, 1:], discount[:, 1:])),
                    reverse=True
                )
                replay_ret = swapaxes(replay_ret)
                replay_critic_loss = ((
                    optax.softmax_cross_entropy(
                        replay_value_logits[:, :-1],
                        twohot(sg(replay_ret), bins)
                    ) +
                    params.critic_ema_reg * optax.softmax_cross_entropy(
                        replay_value_logits,
                        twohot(sg(replay_slow_values), bins)
                    )[:, :-1]
                )).mean()
                replay_values = inv_twohot(nnx.softmax(replay_value_logits, -1), bins)

                # ep labels

                _, sum_rewards = jax.lax.scan(
                    lambda r, x: ((1-x.is_first.astype(jnp.float32)) * r + x.reward, (1-x.is_first.astype(jnp.float32)) * r + x.reward),
                    _data.reward[:, 0],
                    swapaxes(tree_map(lambda x: x[:, 1:], _data))
                )
                # sum_rewards = swapaxes(sum_rewards)
                # sum_rewards = (sum_rewards * _data.done[:, 1:]).sum(-1) + sum_rewards[:, 0] + sum_rewards[:, -1] * (1-_data.is_first[:, -1].astype(jnp.float32))
                # sum_rewards =  sum_rewards/(_data.done[:, 1:].sum(-1) + 1 + (1-_data.is_first[:, -1].astype(jnp.float32)))

                replay_critic_logs = {
                    'replay_critic/loss': replay_critic_loss,
                    'replay_critic/returns': replay_ret.mean(),
                    'replay_critic/values': replay_values.mean(),
                    # 'replay_critic/sum_rews': sum_rewards.mean(),
                }
                loss = wm_loss + params.critic_loss_const * img_critic_loss + params.critic_replay_loss_const * replay_critic_loss + params.actor_loss_scale * actor_loss

                logs = {
                    **rep_logs,
                    **critic_logs,
                    **actor_logs,
                    **replay_critic_logs
                }

                replayed_latents = tree_map(lambda x, y: jnp.concatenate([x[:, 0:1], y], axis=1), data.latent, replayed_latents)
                return loss, (replayed_latents, ret, logs)
            
            # handle resets in imagined steps? mask out the imagines transitions after cont is done
            rng, rng_buffer = jax.random.split(rng)
            data = replay_buffer.sample(agent.buffer_state, rng_buffer)

            rng, rng_loss = jax.random.split(rng)
            actor_critic = nnx.merge(*agent.agent)
            rssm = nnx.merge(*agent.rssm)

            rssm_split = nnx.split(rssm, nnx.Param, ...)
            actor_split = nnx.split(actor_critic.actor, nnx.Param, ...)
            critic_split = nnx.split(actor_critic.critic, nnx.Param, ...)

            graphs, model_params, model_rest = list(zip(rssm_split, actor_split, critic_split))

            (loss, (replayed_latents, returns, logs)), grads = jax.value_and_grad(_loss, has_aux=True)(
                model_params,
                model_rest,
                graphs,
                actor_critic.target_critic,
                actor_critic.ret_norm,
                data.experience,
                rng_loss
            )

            # update replay buffer
            update_experience = data.experience.replace(
                latent=replayed_latents
            )
            updated_batch = data.replace(
                experience=update_experience
            )
            buffer_state = replay_buffer.update(
                agent.buffer_state,
                updated_batch
            )

            # take gradient steps
            model_params = (nnx.state(rssm, nnx.Param), nnx.state(actor_critic.actor, nnx.Param), nnx.state(actor_critic.critic, nnx.Param))
            updates, optimizer_state = optimizer.update(grads, agent.optimizer, model_params)

            # warmup
            scale = jnp.clip(agent.n_updates/params.warmup, 0, 1)

            rssm_params, actor_params, critic_params = optax.apply_updates(model_params, tree_map(lambda x: x*scale, updates))

            # update models
            nnx.update(rssm, rssm_params)
            nnx.update(actor_critic.critic, critic_params)
            nnx.update(actor_critic.actor, actor_params)
            actor_critic.update_slow_params(params, returns) # update normalization and target networks
            
            
            agent = agent.replace(
                rssm=nnx.split(rssm),
                agent=nnx.split(actor_critic),
                optimizer=optimizer_state,
                buffer_state=buffer_state,
                n_updates=agent.n_updates+1,
            )
            
            return (agent, rng, num_updates-1, loss, logs)
        

        last_trained = jax.lax.cond(
            (num_updates >= 1) & (agent.timesteps > training_starts),
            lambda : agent.timesteps,
            lambda : agent.last_trained
        )
        dummy_log = {
            'rep/prior_ent': 0.,
            'rep/post_ent': 0.,
            'rep/reward_loss': 0.,
            'rep/cont_loss': 0.,
            'rep/reconst_loss': 0.,
            'imgcritic/loss': 0.,
            'imgcritic/values':  0.,
            'imgcritic/length': 0.,
            'imgcritic/returns': 0.,
            'imgcritc/rewards': 0.,
            'actor/entropy': 0.,
            'actor/actor_loss': 0.,
            'actor/adv': 0.,
            'replay_critic/loss': 0.,
            'replay_critic/returns': 0.,
            'replay_critic/values': 0.,
            # 'replay_critic/sum_rews': 0.,
        }

        agent, rng, num_updates, loss, logs = jax.lax.while_loop(
            lambda state: (state[2] >= 1) & (agent.timesteps > training_starts),
            _update_dreamer,
            (agent, rng, num_updates, 0, dummy_log)
        )
        
        agent = agent.replace(last_trained=last_trained)
        _returns = info["returned_episode_returns"].sum() * (info["returned_episode"].sum() >= 1) /  (info["returned_episode"].sum()+1e-12)
        logs = {
            'env/return' : _returns,
            'agent/n_updates': agent.n_updates,
            'agent/timesteps': agent.timesteps,
            # 'agent/loss': loss,
            **logs,
        }
        return (agent, env_state, rng, last_step), logs

    def init_learn_fn(agent, rng):
        rng, rng_ = jax.random.split(rng)

        init_obs, env_state = jax.vmap(env_config.env_reset)(jax.random.split(rng_, env_config.n_envs))
        initial_latent = nnx.merge(*agent.rssm).init_carry()
        initial_latent = jax.tree.map(lambda x: jnp.broadcast_to(x, shape=(env_config.n_envs, *x.shape)), initial_latent)
        initial_action = jnp.zeros((env_config.n_envs,), dtype=jnp.int32) + env_config.n_actions
        initial_done = jnp.ones((env_config.n_envs,))
        return agent, env_state, rng, (init_obs, (initial_latent, initial_action), initial_done)


    def learn_fn(timesteps, agent_training_state):
        n_iters = int(timesteps / env_config.n_envs / params.rollout_length)
        agent_training_state, logs = jax.lax.scan(
            _train,
            agent_training_state,
            jnp.arange(n_iters)
        )
        # agent_training_state, logs = _train(agent_training_state, 0) 

        return agent_training_state, logs


    return init_learn_fn, learn_fn


@chex.dataclass(frozen=True)
class EvaluationState:
    last_obs : chex.Array
    action: int
    returns : float
    done : bool
    steps : int
    is_first : float
    latent : RecurrentLatentState

def get_eval_fn(
        env_config,
        eval_max_length,
        params
    ):

    def _run_eval_ep(rng, policy_fn, init_latent):
        def _env_step(state):
            episode_stats, env_state, rng = state
            last_obs = episode_stats.last_obs
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            last_action = episode_stats.action
            last_latent = episode_stats.latent
            latent, action, _ = policy_fn(
                last_obs,
                (last_latent, last_action),
                episode_stats.is_first,
                rng_a,
            )
            # obs, env_state, reward, done, info = env_config.base_env.step(
            obs, env_state, reward, done, info = env_config.env_step(
                rng_s,
                env_state,
                action
                # env_config.env_params
            )
            episode_stats = episode_stats.replace(
                last_obs=obs,
                action=action,
                returns=episode_stats.returns + reward,
                done=done,
                steps=episode_stats.steps+1,
                is_first=0.,
                latent=latent
            )
            return (episode_stats, env_state, rng)

        # initialize state
        rng, rng_s = jax.random.split(rng)

        # init_obs, env_state = env_config.base_env.reset(rng_s, env_config.env_params)
        init_obs, env_state = env_config.env_reset(rng_s)
        ep_stats = EvaluationState(
            last_obs=init_obs,
            action=0,
            returns=0.,
            done=False,
            steps=0,
            is_first=1.,
            latent=init_latent,
        )

        ep_stats, env_state, rng = jax.lax.while_loop(
            lambda state: ~state[0].done & (state[0].steps < eval_max_length),
            _env_step,
            (ep_stats, env_state, rng)
        )

        return ep_stats
    
    @partial(jax.jit, static_argnums=(1,))
    def _eval(
            agent,
            n_episodes,
            rng
    ):
        
        def policy_fn(obs, carry, is_first, rng): 
            return policy(
            nnx.merge(*agent.rssm),
            nnx.merge(*agent.agent).actor,
            obs,
            carry,
            is_first,
            rng,
            params=params
        )

        init_latent = nnx.merge(*agent.rssm).init_carry()
        rngs = jax.random.split(rng, n_episodes)
        eval_eps = jax.vmap(_run_eval_ep, in_axes=(0, None, None))(rngs, policy_fn, init_latent)
        eval_eps = jax.tree.map(partial(jnp.mean, axis=0), eval_eps)
        
        return {
            'returns': eval_eps.returns, 
            'ep_length': eval_eps.steps
        }


    return _eval

def single_run(config_dict: dict):
    config = oc.create(config_dict)
    
    env_id = config_dict.get("ENV_ID", config.get("env", "Pong"))
    seed = config_dict.get("SEED", config.get("seed", 0))
    
    run_name = f"{env_id}_DreamerV3_{seed}"
    wandb.init(
        project=config_dict.get("PROJECT", "JAXtari-Baselines"),
        entity=config_dict.get("ENTITY", None),
        config=config_dict,
        name=run_name,
        save_code=True,
    )

    from jaxatari.wrappers import AtariWrapper, PixelObsWrapper
    base_env = jaxatari.make(env_id)
    env_atari = AtariWrapper(base_env)
    env = PixelObsWrapper(env_atari)
    rng = jax.random.key(seed)

    def process_obs(raw_obs):
        img = raw_obs.astype(jnp.float32)
        if len(img.shape) == 4:
            img = img[-1] 
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img = jnp.dot(img, jnp.array([0.2989, 0.5870, 0.1140]))
        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        img = jax.image.resize(img, (64, 64), method='bilinear')
        return jnp.expand_dims(img, axis=-1)

    def jaxatari_reset(reset_rng):
        o, s = env.reset(reset_rng)
        return process_obs(o), s

    def jaxatari_step(step_rng, state, step_action):
        def scan_fn(carry, _):
            curr_state, reward_sum, done_flag = carry
            o, next_state, r, term, trunc, i = env.step(curr_state, step_action)
            d = jnp.logical_or(term, trunc)
            valid = jnp.logical_not(done_flag)
            r = jnp.where(valid, r, 0.0)
            next_done = jnp.logical_or(done_flag, d)
            next_carry = (next_state, reward_sum + r, next_done)
            return next_carry, o

        carry = (state, jnp.float32(0.0), jnp.bool_(False))
        (final_state, total_reward, final_done), obs_seq = jax.lax.scan(
            scan_fn, carry, None, length=config.params.action_repeat
        )
        final_obs = jax.tree_map(lambda x: x[-1], obs_seq)
        info_dict = {
            "returned_episode_returns": jnp.zeros_like(total_reward),
            "returned_episode": jnp.asarray(final_done, dtype=jnp.float32)
        }
        return process_obs(final_obs), final_state, total_reward, final_done, info_dict

    obs, env_state = jaxatari_reset(rng)
    n_actions = env.action_space().n
    
    with open_dict(config):
        config.params.n_actions = n_actions
        config.rssm.n_actions = n_actions
        config.actor_critic.n_actions = n_actions
        config.actor_critic.actor.output_dim = n_actions
        
    action = jax.random.randint(rng, (), 0, n_actions)
    obs, env_state, reward, done, info = jaxatari_step(rng, env_state, action)

    env_config = EnvConfig(
        obs=obs, action=action, done=done, reward=reward.astype(jnp.float32),
        n_actions=n_actions, env_reset=jaxatari_reset, env_step=jaxatari_step,
        n_envs=config.get("n_envs", 8), base_env=env
    )

    params = DreamerParams(**config.params)
    agent, optimizer, replay_buffer = build_dreamer(params, config, env_config)

    total_timesteps = config_dict.get("TOTAL_TIMESTEPS", config.get("timesteps", 10000000))
    eval_every = config_dict.get("EVAL_EVERY", config.get("eval_every", 10000))
    n_iters = int(total_timesteps // eval_every)

    batch_steps = config.replay_buffer.batch_size * (config.replay_buffer.batch_length - 1)
    update_ratio = params.replay_ratio / batch_steps 

    learn_init_fn, learn_fn = get_dreamer_learn_fn(
        params, optimizer, replay_buffer, env_config, update_ratio,
        config.replay_buffer.batch_size * config.replay_buffer.batch_length
    )
    learn_fn = jax.jit(learn_fn, static_argnums=0)
    agent_training_state = learn_init_fn(agent, rng)

    eval_fn = get_eval_fn(env_config, config.eval_max_length, params)
    
    eval_logs = {}
    for i in tqdm(range(n_iters), desc="Training DreamerV3"):
        agent_training_state, training_logs = learn_fn(eval_every, agent_training_state)

        agent = agent_training_state[0]
        rng, rng_eval = jax.random.split(agent.rng)
        agent = agent.replace(rng=rng)
        eval_logs = eval_fn(agent_training_state[0], config.eval_n_episodes, rng_eval)

        clean_train_logs = {f"train/{k}": float(v.mean()) if hasattr(v, 'mean') else float(v) for k, v in training_logs.items()}
        clean_eval_logs = {f"eval/{k}": float(v.mean()) if hasattr(v, 'mean') else float(v) for k, v in eval_logs.items()}

        wandb.log({
            **clean_train_logs,
            **clean_eval_logs,
            "charts/global_step": eval_every * (i + 1)
        })

    wandb.finish()
    return clean_eval_logs