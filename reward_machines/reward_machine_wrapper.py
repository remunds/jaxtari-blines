import jax.numpy as jnp
from flax import struct


# datatclass to track current state of the rm
@struct.dataclass
class RewardMachineState():
    u: jnp.ndarray
