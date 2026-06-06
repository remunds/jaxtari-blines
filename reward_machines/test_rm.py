import jax
import jax.numpy as jnp
import jaxatari
from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper

from reward_machines.reward_machine import RewardMachine
from reward_machines.games.pong_rm import PongRm
from reward_machines.reward_machine_wrapper import RewardMachineWrapper

# Build the chain
base = jaxatari.make("pong")
env = ObjectCentricWrapper(AtariWrapper(base))
rm = RewardMachine(PongRm())
env = RewardMachineWrapper(env, rm)

# Single env, no vmap yet — clean debug output
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)
print("reset obs shape:", obs.shape)

action = jnp.array(0)  # NOOP
for i in range(5):
    obs, state, reward, done, truncated, info = env.step(state, action)
    print(f"step {i}: u={state.u} reward={reward} done={done}")
