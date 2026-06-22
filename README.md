# Baseline Implementations to evaluate on JAXtari
The goal of this repository is to provide simple baseline implementations of model-free, model-based and neuro-symbolic RL algorithms. 
All implementations have in common:
- Simple
- Fast
- JAX-native


## Installation instructions
We use the uv package manager. Follow their [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

Next, install all dependencies by running: `uv sync`.

If you have access to a GPU, make sure to install the correct JAX version, e.g., `uv add "jax[cuda12]"`.

## Training 
To start a training run, simply call `uv run train.py --config configs/<agent_name>`.


# Scoreboard
We will provide an updated table of metrics achieved by the agents.
Note, the metrics are averages across the current JAXtari-15 suite of games.
HNS: Human-normalized score, PC: Performance Change (eval agents on modifications).

### Model-Free
| Agent | HNS (RGB) | PC (RGB) | HNS (OC) |  PC (OC) 
|-------|-------|-------|-------|-------|

### Model-Based
| Agent | HNS (RGB) | PC (RGB) | HNS (OC) |  PC (OC) 
|-------|-------|-------|-------|-------|

### Neuro-Symbolic
| Agent | HNS (RGB) | PC (RGB) | HNS (OC) |  PC (OC) 
|-------|-------|-------|-------|-------|

