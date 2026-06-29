"""Trajectory data collection and dataloading for C-JEPA training.

Collects random-policy trajectories from JAXtari environments using
ObjectCentricWrapper observations, then builds a JAX-native dataloader.
"""

from typing import Iterator, Tuple, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

import jaxatari
from jaxatari.wrappers import (
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    AtariWrapper,
    LogWrapper,
    FlattenObservationWrapper,
)
from jaxatari import spaces


def make_oc_env(env_id: str, frame_stack_size: int = 4, frame_skip: int = 4):
    """Create an environment with ObjectCentricWrapper (no flattening).

    Returns an env where observation_space is Box(frame_stack_size, num_features).
    """
    env = jaxatari.make(env_id, mods=None)
    env = AtariWrapper(
        env,
        sticky_actions=0.0,
        episodic_life=False,
        noop_max=30,
        first_fire=True,
        full_action_space=False,
    )
    env = ObjectCentricWrapper(
        env,
        frame_stack_size=frame_stack_size,
        frame_skip=frame_skip,
        clip_reward=False,
    )
    env = LogWrapper(env)
    return env


def make_oc_env_single_frame(env_id: str):
    """Create an environment with per-frame ObjectCentricWrapper (no stacking).

    Returns an env where observation_space is Box(num_features,).
    This is used for collecting frame-level trajectories.
    """
    env = jaxatari.make(env_id, mods=None)
    env = AtariWrapper(
        env,
        sticky_actions=0.0,
        episodic_life=False,
        noop_max=30,
        first_fire=True,
        full_action_space=False,
    )
    # Single-frame OC wrapper
    env = ObjectCentricWrapper(
        env,
        frame_stack_size=1,
        frame_skip=1,
        clip_reward=False,
    )
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    return env


def get_env_info(env_id: str = "pong") -> Dict:
    """Get environment information: num_slots, obj_attr_dim, num_actions.

    Creates a single-frame OC env and inspects its observation space.

    Returns:
        Dict with keys: num_slots, obj_attr_dim, num_actions, obs_dim
    """
    env = jaxatari.make(env_id, mods=None)
    env = AtariWrapper(
        env,
        sticky_actions=0.0,
        episodic_life=False,
        noop_max=30,
        first_fire=True,
        full_action_space=False,
    )
    env = ObjectCentricWrapper(
        env,
        frame_stack_size=1,
        frame_skip=1,
        clip_reward=False,
    )

    # Inspect the underlying env's observation space to count objects
    object_obs_space = env._env.observation_space()

    # The underlying env has a Dict space with keys like 'player', 'enemy', 'ball', etc.
    # Each object has x, y, width, height, active, visual_id, state, orientation (8 attributes)
    obj_attr_dim = 8  # x, y, width, height, active, visual_id, state, orientation
    num_slots = 0
    # Iterate over the Dict's .spaces OrderedDict
    for key, space in object_obs_space.spaces.items():
        if isinstance(space, spaces.Dict) and "x" in space.spaces:
            # It's an ObjectObservation (player, enemy, ball, etc.)
            x_space = space.spaces["x"]
            if isinstance(x_space, spaces.Box):
                shape = x_space.shape
                if len(shape) == 0 or shape[0] is None:
                    num_slots += 1
                else:
                    num_slots += shape[0]
            else:
                num_slots += 1

    num_actions = env.action_space().n

    return {
        "num_slots": num_slots,
        "obj_attr_dim": obj_attr_dim,
        "num_actions": num_actions,
        "obs_dim": num_slots * obj_attr_dim,
    }


@partial(jax.jit, static_argnums=(0, 1))
def collect_random_trajectory(
    env,
    num_steps: int,
    rng: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Collect a single random-policy trajectory.

    Args:
        env: A JAXtari environment instance (vmap-compatible reset/step).
        num_steps: Number of steps to collect.
        rng: PRNG key.

    Returns:
        obs: [num_steps, obs_dim]
        actions: [num_steps]
        dones: [num_steps]
    """
    def step_fn(carry, _):
        obs, state, rng = carry
        rng, action_key = jax.random.split(rng)
        action = jax.random.randint(action_key, (), 0, env.action_space().n)
        next_obs, state, reward, terminated, truncated, info = env.step(state, action)
        done = jnp.logical_or(terminated, truncated)
        return (next_obs, state, rng), (obs, action, done)

    rng, reset_key = jax.random.split(rng)
    obs, state = env.reset(reset_key)
    (_, _, rng), (obs_seq, actions_seq, dones_seq) = jax.lax.scan(
        step_fn, (obs, state, rng), None, length=num_steps
    )
    return obs_seq, actions_seq, dones_seq


@partial(jax.jit, static_argnums=(0, 1, 2))
def collect_batch_trajectories(
    env,
    num_envs: int,
    num_steps: int,
    rng: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Collect random-policy trajectories from multiple environments.

    Args:
        env: A JAXtari environment (will be vmap'd).
        num_envs: Number of parallel environments.
        num_steps: Steps per trajectory.
        rng: PRNG key.

    Returns:
        obs: [num_envs, num_steps, obs_dim]
        actions: [num_envs, num_steps]
        dones: [num_envs, num_steps]
    """
    vmap_reset = jax.vmap(env.reset)
    vmap_step = jax.vmap(env.step)

    def step_fn(carry, _):
        obs, state, rng = carry
        rng, *action_keys = jax.random.split(rng, num_envs + 1)
        action_keys = jnp.array(action_keys)
        actions = jax.vmap(lambda k: jax.random.randint(k, (), 0, env.action_space().n))(
            action_keys
        )
        next_obs, state, reward, terminated, truncated, info = vmap_step(state, actions)
        done = jnp.logical_or(terminated, truncated)
        return (next_obs, state, rng), (obs, actions, done)

    rng, reset_key = jax.random.split(rng)
    obs, state = vmap_reset(jax.random.split(reset_key, num_envs))
    (_, _, rng), (obs_seq, actions_seq, dones_seq) = jax.lax.scan(
        step_fn, (obs, state, rng), None, length=num_steps
    )
    # scan gives (num_steps, num_envs, ...), transpose to (num_envs, num_steps, ...)
    obs_seq = obs_seq.transpose(1, 0, 2)
    actions_seq = actions_seq.transpose(1, 0)
    dones_seq = dones_seq.transpose(1, 0)
    return obs_seq, actions_seq, dones_seq


class TrajectoryDataset:
    """JAX-native trajectory dataset for C-JEPA training.

    Stores pre-collected trajectories and provides windowed batches.
    """

    def __init__(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        dones: jnp.ndarray,
        history_frames: int = 3,
        pred_frames: int = 1,
    ):
        """
        Args:
            obs: [num_trajectories, traj_length, obs_dim]
            actions: [num_trajectories, traj_length]
            dones: [num_trajectories, traj_length]
            history_frames: Number of history frames for predictor.
            pred_frames: Number of future frames to predict.
        """
        self.obs = obs
        self.actions = actions
        self.dones = dones
        self.history_frames = history_frames
        self.pred_frames = pred_frames
        self.window_size = history_frames + pred_frames
        self.num_trajectories, self.traj_length, self.obs_dim = obs.shape

        # Precompute valid window start indices (where no done occurs in the window)
        self._compute_valid_windows()

    def _compute_valid_windows(self):
        """Compute indices of valid windows across all trajectories.

        Uses numpy (single device→host transfer) instead of per-trajectory
        jnp.where calls which trigger N separate GPU kernel launches.

        Stores self.window_indices as a JAX array [num_windows, 2] of (traj_idx, start).
        """
        # Single transfer: convert dones and obs to numpy once
        dones_np = np.array(self.dones)
        obs_np = np.array(self.obs) if hasattr(self.obs, 'device') else self.obs
        ws = self.window_size
        tl = self.traj_length

        # Ball center position — idle at reset / after a point
        BALL_CENTER_X, BALL_CENTER_Y = 78, 115
        BALL_TOL = 3

        all_starts = []
        for traj_idx in range(self.num_trajectories):
            done_pos = np.where(dones_np[traj_idx])[0]
            if len(done_pos) == 0:
                # No dones: one continuous segment
                # Filter out windows where ball is idle at the start
                starts = []
                for start in range(tl - ws + 1):
                    bx = obs_np[traj_idx, start, 16]
                    by = obs_np[traj_idx, start, 17]
                    if abs(bx - BALL_CENTER_X) >= BALL_TOL or abs(by - BALL_CENTER_Y) >= BALL_TOL:
                        starts.append(start)
                if starts:
                    seg = np.column_stack([
                        np.full(len(starts), traj_idx, dtype=np.int32),
                        np.array(starts, dtype=np.int32),
                    ])
                    all_starts.append(seg)
            else:
                # Split at done positions
                seg_starts = np.concatenate([[-1], done_pos]) + 1
                seg_ends = np.concatenate([done_pos, [tl]])
                seg_lens = seg_ends - seg_starts
                for s, l in zip(seg_starts, seg_lens):
                    if l >= ws:
                        # Filter out windows where ball is idle at the start
                        valid = []
                        for start in range(s, s + l - ws + 1):
                            bx = obs_np[traj_idx, start, 16]
                            by = obs_np[traj_idx, start, 17]
                            if abs(bx - BALL_CENTER_X) >= BALL_TOL or abs(by - BALL_CENTER_Y) >= BALL_TOL:
                                valid.append(start)
                        if valid:
                            seg = np.column_stack([
                                np.full(len(valid), traj_idx, dtype=np.int32),
                                np.array(valid, dtype=np.int32),
                            ])
                            all_starts.append(seg)

        if all_starts:
            combined = np.concatenate(all_starts, axis=0)
            self.window_indices = jnp.array(combined, dtype=jnp.int32)
            self.num_windows = combined.shape[0]
        else:
            self.window_indices = jnp.zeros((0, 2), dtype=jnp.int32)
            self.num_windows = 0

    def get_batch(
        self,
        batch_size: int,
        rng: jax.Array,
    ) -> Dict[str, jnp.ndarray]:
        """Sample a random batch of windows using JIT-compatible indexing.

        Uses vmap + dynamic_slice for GPU-friendly extraction.

        Args:
            batch_size: Number of windows to sample.
            rng: PRNG key.

        Returns:
            Dict with:
                obs: [batch_size, history_frames, obs_dim]
                actions: [batch_size, history_frames]
                target_obs: [batch_size, pred_frames, obs_dim]
                target_actions: [batch_size, pred_frames]
        """
        if self.num_windows == 0:
            raise ValueError("No valid windows in dataset.")

        idx = jax.random.randint(rng, (batch_size,), 0, self.num_windows)
        traj_idx = self.window_indices[idx, 0]  # [batch_size]
        start = self.window_indices[idx, 1]      # [batch_size]

        # Use vmap + dynamic_slice for JIT-compatible window extraction
        def extract_window(traj_idx, start):
            obs_window = jax.lax.dynamic_slice(
                self.obs, (traj_idx, start, 0),
                (1, self.window_size, self.obs_dim),
            )[0]
            act_window = jax.lax.dynamic_slice(
                self.actions, (traj_idx, start),
                (1, self.window_size),
            )[0]
            return obs_window, act_window

        obs_batch, act_batch = jax.vmap(extract_window)(traj_idx, start)

        return {
            "obs": obs_batch[:, :self.history_frames, :],        # [B, T_hist, obs_dim]
            "actions": act_batch[:, :self.history_frames],       # [B, T_hist]
            "target_obs": obs_batch[:, self.history_frames:, :],  # [B, T_pred, obs_dim]
            "target_actions": act_batch[:, self.history_frames:],  # [B, T_pred]
        }


def collect_dataset(
    env_id: str = "pong",
    num_trajectories: int = 1000,
    traj_length: int = 50,
    num_envs: int = 64,
    seed: int = 42,
) -> TrajectoryDataset:
    """Collect a dataset of random-policy trajectories.

    Args:
        env_id: Atari game ID.
        num_trajectories: Number of trajectories to collect.
        traj_length: Length of each trajectory.
        num_envs: Number of parallel environments.
        seed: Random seed.

    Returns:
        TrajectoryDataset.
    """
    # Create single-frame env for collecting per-step observations
    # This gives us (obs_dim,) per step (after FlattenObservationWrapper)
    env = make_oc_env_single_frame(env_id)
    env_info = get_env_info(env_id)
    obs_dim = env_info["obs_dim"]

    rng = jax.random.PRNGKey(seed)
    num_batches = (num_trajectories + num_envs - 1) // num_envs

    all_obs = []
    all_actions = []
    all_dones = []

    for batch_idx in range(num_batches):
        rng, batch_key = jax.random.split(rng)
        remaining = min(num_envs, num_trajectories - batch_idx * num_envs)
        obs, actions, dones = collect_batch_trajectories(
            env, remaining, traj_length, batch_key
        )
        # obs: [num_envs, traj_length, obs_dim]
        all_obs.append(obs)
        all_actions.append(actions)
        all_dones.append(dones)

        print(f"  Collected batch {batch_idx + 1}/{num_batches}: "
              f"{obs.shape[0]} trajs × {traj_length} steps")

    all_obs = jnp.concatenate(all_obs, axis=0)[:num_trajectories]
    all_actions = jnp.concatenate(all_actions, axis=0)[:num_trajectories]
    all_dones = jnp.concatenate(all_dones, axis=0)[:num_trajectories]

    # Slice observation to only object attributes (exclude score values)
    # The OC wrapper includes 2 extra dims for score_player/score_enemy
    if all_obs.shape[-1] != obs_dim:
        all_obs = all_obs[..., :obs_dim]

    print(f"Total: {all_obs.shape[0]} trajectories × {traj_length} steps, "
          f"obs_dim={all_obs.shape[2]}")

    return TrajectoryDataset(
        obs=all_obs,
        actions=all_actions,
        dones=all_dones,
    )
