# LeWM — a JEPA world model for JAXtari, and an agent on top of it

Implementation of **LeWorldModel** (Maes, Le Lidec, Scieur, LeCun, Balestriero,
2026 — [arXiv:2603.19312](https://arxiv.org/abs/2603.19312), reference code
[lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)) trained on JAXtari, plus
PPO agents that learn on its representations.

LeWM is a joint-embedding predictive architecture: it predicts the *embedding* of
the next frame rather than the next frame's pixels. The usual failure mode of that
objective is representation collapse — the encoder can drive the loss to zero by
mapping every frame to the same vector. LeWM's claim is that **SIGReg** (a sketched
isotropic-Gaussian regularizer) prevents this without the stop-gradient or EMA
target that BYOL-style methods rely on, so the implementation here trains
end-to-end with **no stop-gradient**.

That claim is tested rather than assumed — see *Results* below.

## A note on the framework

The rest of this repository is JAX-native. This baseline keeps the **environment in
JAX** and writes the **networks in PyTorch**. That is a deliberate deviation and the
reason is measurable: it was developed on Apple silicon, where JAX has no GPU backend
and falls back to CPU while PyTorch reaches the GPU through Metal. One forward and
backward pass of the trunk at batch 512 took 772 ms under JAX on CPU against 37 ms
under PyTorch on Metal, which moved the agent from 8 to roughly 1300 environment
steps per second.

On the CUDA machines this repository targets, that argument does not hold, and a JAX
port is the obvious next step. `torch` and `einops` are added to `pyproject.toml` for
this baseline only. Flagging it explicitly so the maintainers can decide whether to
accept it as is or ask for the port first.

## Files

| file | what it is |
|---|---|
| `lewm.py` | this repository's entry point: `single_run(config)`, pretrains the world model then trains PPO on it |
| `world_model.py` | the world model (CNN encoder, SIGReg, AdaLN Transformer predictor), data collection, training, open-loop rollout evaluation |
| `features.py` | loads a trained encoder as a frozen feature extractor, recomputing its BatchNorm statistics first |
| `ppo_lewm.py` | PPO with a pluggable trunk: from-scratch CNN, frozen LeWM, or fine-tuned LeWM |
| `test_lewm.py` | unit tests (`pytest agents/lewm/test_lewm.py`, ~1 s) |
| `report/` | a 14-page write-up of the method, the ablations and the results |

The model itself is one self-contained file. The experiment runners used to produce
the numbers below (all 15 games, the ablation grid, the seed aggregation) live in the
accompanying report's repository and are not needed to run this baseline.

## Architecture

- **Encoder** — 3-layer CNN (the Nature-DQN trunk) → Linear → BatchNorm, giving a
  256-d embedding per frame. The paper uses a ViT; a CNN is substituted because
  Atari frames are small and the paper shows the method is encoder-agnostic
  (App. G).
- **Predictor** — 4-layer causal Transformer over the embedding sequence.
  Actions enter through **adaptive LayerNorm with a zero-initialised modulation
  head** (as in the paper), so at step 0 every gate is 0 and the predictor is
  exactly the identity. Output goes through a projector matching the encoder's
  (Linear → BatchNorm).
- **Loss** — `MSE(pred, target) + λ · SIGReg(embeddings)`, λ = 0.1. Targets come
  from the *same* encoder with gradients flowing through them.

`--action_cond add` swaps AdaLN for a simple additive action embedding, kept as an
ablation rather than deleted.

## Running

```bash
# the reported configuration: pretrain the world model, then PPO on the frozen encoder
uv run main.py +alg=lewm_rgb ENV_ID=pong

# the control arm: identical PPO, no world model
uv run main.py +alg=lewm_rgb alg.FEATURES=scratch ENV_ID=pong

# fine-tuned encoder instead of frozen
uv run main.py +alg=lewm_rgb alg.FEATURES=lewm_finetune ENV_ID=pong

# quick sanity check (~1 min)
uv run main.py +alg=lewm_rgb alg.WM_TOTAL_STEPS=30 alg.WM_INIT_SEQUENCES=40 \
    alg.TOTAL_TIMESTEPS=8192 NUM_SEEDS=1
```

World-model knobs are prefixed `WM_` in `config/alg/lewm_rgb.yaml` so they cannot
collide with PPO's. `alg.WM_SIGREG_WEIGHT=0` reproduces the collapse described below,
and `alg.WM_ACTION_COND=add` swaps AdaLN for additive action conditioning.

## Reading the results

**Falling loss does not mean learning.** Collapse also lowers the loss — in fact
it lowers it *more*. Two diagnostics guard against this:

- **Effective rank** of the embeddings (out of 256). BatchNorm pins per-dimension
  variance to 1, so collapse does not show up as shrinking norms; it shows up as
  the embeddings occupying a low-dimensional subspace. Near 1 means collapse.
- **Open-loop rollout MSE against a frozen baseline.** The predictor is seeded
  with 3 true embeddings then run autoregressively on its own outputs, and scored
  against the trivial predictor that assumes the embedding never changes. MSE in a
  learned latent space has no absolute scale, so only the comparison is meaningful.

## Results

**14 of 15 games** train without collapse and beat the frozen baseline. MsPacman
does not (rollout 0.282 vs baseline 0.272) and is reported as a failure rather
than dropped.

**Ablations** (`results/ablations`, 5000 steps, Pong / Seaquest). Rollout error is
given as a **ratio to each variant's own frozen baseline**, because latent MSE has
no absolute scale and is not comparable across variants that define different
embedding geometries. Lower is better; 1.0 means no better than assuming nothing
changes.

| variant | eff. rank | pred loss | rollout / frozen |
|---|---|---|---|
| **faithful** (paper) | 28.4 / 28.5 | **0.0087 / 0.0113** | **0.286 / 0.235** |
| no SIGReg | **1.7 / 1.9** | 0.0010 / 0.0004 | **1.006** / 0.695 |
| stop-gradient + SIGReg | 174.4 / 68.6 | 0.0730 / 0.0526 | 0.379 / 0.356 |
| stop-gradient only | 149.4 / 42.8 | 0.0960 / 0.0942 | 0.430 / 0.689 |
| additive actions | 21.2 / 21.1 | 0.0092 / 0.0115 | 0.310 / **0.188** |
| λ = 0.5 | 58.9 / 44.6 | 0.0209 / 0.0225 | 0.289 / 0.305 |
| λ = 2.0 | 89.9 / 59.5 | 0.0399 / 0.0393 | 0.330 / 0.409 |

**Removing SIGReg collapses the representation.** Effective rank falls to 1.7 / 1.9
while the prediction loss *improves* 9x / 28x — the collapse is invisible in the
training loss, which is the entire reason the rank diagnostic exists. On Pong the
collapsed model then fails to beat even its own trivial baseline.

**But SIGReg is sufficient, not necessary.** The `stop_grad_only` control (λ = 0,
stop-gradient on) does *not* collapse either: rank 149 / 43. A plain stop-gradient
prevents collapse on its own here. What SIGReg buys is a much better model — 11x /
8x lower prediction error and a better rollout ratio than either stop-gradient
variant. So the claim this work supports is narrower than "SIGReg is what prevents
collapse": **SIGReg prevents collapse without a stop-gradient, and yields a
markedly better predictor than a stop-gradient does.** Adding a stop-gradient on
top of SIGReg only makes things worse.

High rank is not itself good — `stop_grad` has the highest rank of any variant
(174) and a worse rollout ratio than faithful.

**λ = 0.1 is the right setting on Atari.** Raising it monotonically increases rank
and degrades the rollout ratio on both games (0.286→0.289→0.330 and
0.235→0.305→0.409).

**AdaLN vs additive action conditioning is a wash at this scale.** AdaLN gives
slightly lower prediction loss on both games; the additive variant gives a better
rollout ratio on Seaquest and worse on Pong. This is reported rather than omitted:
the paper's conditioning scheme shows no measurable benefit here, and is kept for
fidelity rather than because it helps. (An earlier run had 25% of action labels
corrupted by sticky actions, which would have masked any difference; these numbers
are from the corrected pipeline and the conclusion is unchanged.)

**Agents** (`results/ppo`, 1M steps, seed 1). Returns are the mean over the last
10% of iterations, as unclipped game score.

| game | PPO from scratch | on frozen LeWM | on fine-tuned LeWM |
|---|---|---|---|
| pong | 15.3 | **17.2** | −15.6 |
| seaquest | **949** | 571 | 411 |
| breakout | **40.8** | 12.2 | 12.0 |

**Pong repeated over 3 seeds** (`results/seeds`, via `aggregate_seeds.py`), because
a single Atari run says very little:

| | final return | steps to first positive return |
|---|---|---|
| PPO from scratch | 12.60 ± 7.27 | 0.61M — [0.63, 0.69, 0.51] |
| PPO on frozen LeWM | 17.36 ± 1.09 | **0.32M** — [0.27, 0.40, 0.29] |

Those two columns say different things and are kept apart deliberately.

**Final return is not significantly different.** The gap in means is +4.8 while the
scratch standard deviation alone is 7.3. "LeWM scores higher on Pong" does not
survive three seeds, and is not claimed here.

**Sample efficiency is a clean result.** LeWM features roughly halve the steps to
positive play, and the per-seed values do not overlap: the *worst* LeWM seed
(0.40M) still beats the *best* scratch seed (0.51M). This is the effect a
pretrained representation is supposed to produce, and it is what this work
supports.

**Run-to-run variance drops about sevenfold** (1.09 vs 7.27). Every LeWM seed lands
within a 2-point band; scratch spans 14 points. Pretraining makes the run
predictable as well as faster.

On Seaquest and Breakout the frozen encoder plateaus well below the scratch CNN,
and fine-tuning does not close the gap — which rules out the obvious explanation
that the frozen encoder is merely stale with respect to the improving policy.

**Against published PPO** (Schulman et al. 2017, Table 6: last-100-episode mean
after 40M frames = 10M timesteps, i.e. **10x the budget used here**).

| game | best here (1M) | published PPO (10M) | ratio |
|---|---|---|---|
| pong | 17.2 | 20.7 | 0.83 |
| seaquest | 949 | 1204.5 | 0.79 |
| breakout | 40.8 | 274.8 | **0.15** |

Pong and Seaquest reach ~4/5 of the published score on 1/10 of the interaction,
which is what a correct implementation stopped early should look like. Breakout is
the outlier, and it is also the game where the frozen representation does worst —
consistent with the frame-stacking limitation noted below, though the budget and
the representation are not separated here.

## Known limitations

- **Budget.** Agents are trained for 1M steps against a 200M-step reference. The
  curves show learning trends, not converged performance.
- **Scores are compared against the reference, curves are not.** The source the lab
  guidelines link publishes final scores only, with no per-game curve data, and LeWM's
  own reference implementation produces no game score at all (it is evaluated by latent
  planning). Overlaying the repo's own PPO would work in principle, but it is JAX and
  therefore CPU-only on Apple silicon: measured at ~17 steps/s, so one 1M-step run is
  ~15 h, and raising `NUM_ENVS` from 8 to 32 to 128 did not help. Cheap on a CUDA box,
  and the first thing worth adding.
- **Seeds.** Pong uses 3 seeds; every other agent number is a single seed and is
  indicative only. The Pong result shows why that matters: the final-return gap
  vanished into the noise once seeds were added, while the sample-efficiency gap
  survived. Seaquest and Breakout have not had that test.
- **The agent comparison is not purely a features comparison.** The scratch CNN
  mixes the four stacked frames convolutionally, while the LeWM trunk encodes each
  frame independently and concatenates the embeddings — so it has weaker access to
  precisely localised motion. That is a plausible contributor to the Breakout gap
  and is not controlled for.
- **Fine-tuning is constrained.** Pretrained encoder weights use
  `--encoder_lr_scale` (default 0.1). At the full rate training diverges; at this
  rate the encoder barely adapts. Fine-tuning here is caught between the two, so
  the fine-tuned arm should be read as "does not close the gap at this setting"
  rather than a general statement.
- **Random data-collection policy.** The world model only sees the state space a
  random agent reaches — a small slice on sparse-reward games such as
  MontezumaRevenge and Gravitar. Episodes run up to `--max_episode_steps` (500)
  with `episodic_life=False`, and windows are sampled across the whole episode, so
  coverage is not confined to the opening seconds; but it is still random-policy
  coverage.
- **MsPacman fails.** Its rollout error does not beat the frozen baseline. The
  ratio table above shows the same variant ordering holds elsewhere, so this looks
  game-specific rather than systematic, but it is not explained.
- **BatchNorm at evaluation.** `evaluate_rollout` keeps BatchNorm in
  batch-statistics mode for consistency with training. Using fixed running
  statistics instead changes rollout MSE by ~7% (measured on Pong); that is the
  size of the methodological wobble under these numbers.
- **Remaining deviations from the paper.** CNN encoder rather than ViT (argued
  above), and online random collection rather than an offline reward-free dataset.
- **The predictor is not used for control.** The paper evaluates LeWM by latent
  MPC planning; the agents here learn a policy on the encoder's representation
  instead. Planning in the latent space is the natural next step.
