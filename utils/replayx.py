import functools
import warnings
from typing import TYPE_CHECKING, Callable, Generic, Optional, Tuple, TypeVar

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax
import jax.numpy as jnp
from jax import Array

from flashbax import utils
from flashbax.buffers import sum_tree, trajectory_buffer
from flashbax.buffers.sum_tree import SumTreeState
from flashbax.buffers.trajectory_buffer import (
    BufferSample,
    BufferState,
    Experience,
    TrajectoryBufferSample,
    TrajectoryBufferState,
    can_sample,
    validate_trajectory_buffer_args,
    add,
    init,
    calculate_uniform_item_indices
)



Indices = Array

@dataclass(frozen=True)
class IndexedTrajectoryBufferSample(TrajectoryBufferSample, Generic[Experience]):
    indices: Indices

BufferState = TypeVar("BufferState", bound=TrajectoryBufferState)
BufferSample = TypeVar("BufferSample", bound=TrajectoryBufferSample)

@dataclass(frozen=True)
class IndexedTrajectoryBuffer(Generic[Experience, BufferState, BufferSample]):
    init: Callable[[Experience], BufferState]
    add: Callable[
        [BufferState, Experience],
        BufferState,
    ]
    sample: Callable[
        [BufferState, chex.PRNGKey],
        BufferSample,
    ]
    can_sample: Callable[[BufferState], Array]
    update: Callable[[BufferState, BufferSample], BufferState]



def sample(
    state: TrajectoryBufferState[Experience],
    rng_key: chex.PRNGKey,
    batch_size: int,
    sequence_length: int,
    period: int,
) -> TrajectoryBufferSample[Experience]:
    """
    Sample a batch of trajectories from the buffer.

    Args:
        state: The buffer's state.
        rng_key: Random key.
        batch_size: Batch size of sampled experience.
        sequence_length: Length of trajectory to sample.
        period: The period refers to the interval between sampled sequences. It serves to regulate
            how much overlap there is between the trajectories that are sampled. To understand the
            degree of overlap, you can calculate it as the difference between the
            sample_sequence_length and the period. For instance, if you set period=1, it means that
            trajectories will be sampled uniformly with the potential for any degree of overlap. On
            the other hand, if period is equal to sample_sequence_length - 1, then trajectories can
            be sampled in a way where only the first and last timesteps overlap with each other.
            This helps you control the extent of overlap between consecutive sequences in your
            sampling process.

    Returns:
        A batch of experience.
    """
    add_batch_size, max_length_time_axis = utils.get_tree_shape_prefix(
        state.experience, n_axes=2
    )
    # Calculate the indices of the items that will be sampled.
    item_indices = calculate_uniform_item_indices(
        state,
        rng_key,
        batch_size,
        sequence_length,
        period,
        add_batch_size,
        max_length_time_axis,
    )

    # Convert the item indices to the indices of the data buffer
    flat_data_indices = item_indices * period
    # Get the batch index and time index of the sampled items.
    batch_data_indices = flat_data_indices // max_length_time_axis
    time_data_indices = flat_data_indices % max_length_time_axis

    # The buffer is circular, so we can loop back to the start (`% max_length_time_axis`)
    # if the time index is greater than the length. We then add the sequence length to get
    # the end index of the sequence.
    time_data_indices = (
        jnp.arange(sequence_length) + time_data_indices[:, jnp.newaxis]
    ) % max_length_time_axis

    # Slice the experience in the buffer to get a batch of trajectories of length sequence_length
    batch_trajectory = jax.tree.map(
        lambda x: x.at[batch_data_indices[:, jnp.newaxis], time_data_indices].get(),
        state.experience,
    )

    return IndexedTrajectoryBufferSample(experience=batch_trajectory, indices=(batch_data_indices, time_data_indices))


def update(state: TrajectoryBufferState[Experience],
    updated_batch: TrajectoryBufferSample
) -> TrajectoryBufferSample[Experience]:
    
    batch_data_indices, time_data_indices = updated_batch.indices
    updated_experience = jax.tree.map(
        lambda new_x, x: x.at[batch_data_indices[:, jnp.newaxis], time_data_indices].set(new_x),
        updated_batch.experience,
        state.experience
    )

    state = state.replace(
        experience=updated_experience
    )
    return state


def make_trajectory_buffer(
    add_batch_size: int,
    sample_batch_size: int,
    sample_sequence_length: int,
    period: int,
    min_length_time_axis: int,
    max_size: Optional[int] = None,
    max_length_time_axis: Optional[int] = None,
) -> IndexedTrajectoryBuffer:

    validate_trajectory_buffer_args(
        max_length_time_axis=max_length_time_axis,
        min_length_time_axis=min_length_time_axis,
        add_batch_size=add_batch_size,
        sample_sequence_length=sample_sequence_length,
        period=period,
        max_size=max_size,
    )

    if sample_sequence_length > min_length_time_axis:
        min_length_time_axis = sample_sequence_length

    if max_size is not None:
        max_length_time_axis = max_size // add_batch_size

    assert max_length_time_axis is not None
    init_fn = functools.partial(
        init,
        add_batch_size=add_batch_size,
        max_length_time_axis=max_length_time_axis,
    )
    add_fn = functools.partial(
        add,
    )
    sample_fn = functools.partial(
        sample,
        batch_size=sample_batch_size,
        sequence_length=sample_sequence_length,
        period=period,
    )
    can_sample_fn = functools.partial(
        can_sample, min_length_time_axis=min_length_time_axis
    )

    return IndexedTrajectoryBuffer(
        init=init_fn,
        add=add_fn,
        sample=sample_fn,
        can_sample=can_sample_fn,
        update=update
    )