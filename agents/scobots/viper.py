"""
VIPER — distill a trained SCoBots policy into a decision tree.

A SCoBots policy reads named concepts, so a tree imitating it is a list of
if-then rules over quantities a human can read. VIPER is DAgger in which every
state is weighted by how much the teacher cares about it (the spread of its
action log-probabilities), so decisive states dominate the fit:

    collect states with the teacher, label them with its greedy action
    repeat: fit a weighted tree, score it on whole episodes,
            roll it out, label the states it reaches with the *teacher's* action
    keep the best-scoring tree (the smallest one, among ties)

Credits
-------
* Bastani, Pu, Solar-Lezama — "Verifiable Reinforcement Learning via Policy
  Extraction", NeurIPS 2018 (algorithm and Q-disagreement weighting).
* SCoBots applies it the same way (https://github.com/k4ntz/SCoBots, utils/viper.py:
  weight = mean_a log pi(a|s) - min_a log pi(a|s), 25 iterations of 30k samples,
  max depth 7); this is a JAX/JAXtari re-implementation that rolls out many
  environments at once instead of one.

Usage
-----
    uv run python -m agents.scobots.viper extract models/pong_scobots_tuned_mode0_1/*.model
    uv run python -m agents.scobots.viper show models/pong_scobots_tuned_mode0_1/viper/best_+21.00.viper
    uv run python -m agents.scobots.viper show .../best_+21.00.viper --visualize --max-depth 3
"""

import argparse
import copy
import glob
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

from agents.scobots.scobots import MLP, feature_layout, make_env
from agents.scobots.scobots_eval import load_model


class VecEnv:
    """`n` environments stepped together, with reset/step compiled once."""

    def __init__(self, env):
        self.reset = jax.jit(jax.vmap(env.reset))
        self.step = jax.jit(jax.vmap(env.step))


def rollout(vec, policy, n_samples, carry):
    """States visited by `policy` until `n_samples` are collected.

    `carry` is the envs' ``(obs, state)``, continued from the previous rollout
    rather than reset: episodes then run to their end across iterations, as in
    SCoBots' single long-running env, instead of every rollout seeing only the
    first steps of each episode.
    """
    obs, state = carry
    states = []
    for _ in range(-(-n_samples // obs.shape[0])):
        states.append(np.asarray(obs))
        obs, state, *_ = vec.step(state, jnp.asarray(policy(obs)))
    return np.concatenate(states), (obs, state)


def score(vec, policy, n_episodes, max_steps, key):
    """Returns of the first episode of each of `n_episodes` parallel environments.

    Scoring needs whole episodes from a fresh start, which the continued,
    sample-budgeted rollouts above do not provide.
    """
    key, reset_key = jax.random.split(key)
    obs, state = vec.reset(jax.random.split(reset_key, n_episodes))
    returns = np.full(n_episodes, np.nan)
    for _ in range(max_steps):
        obs, state, _, _, _, info = vec.step(state, jnp.asarray(policy(obs)))
        done = np.asarray(info["returned_episode"]) & np.isnan(returns)
        returns[done] = np.asarray(info["returned_episode_returns"])[done]
        if not np.isnan(returns).any():
            break
    return returns[~np.isnan(returns)], key


def mean(returns):
    return float(np.mean(returns)) if len(returns) else float("nan")


def save(blob, path):
    with open(path, "wb") as f:
        pickle.dump(blob, f)


def tree_policy(tree):
    return lambda obs: tree.predict(np.asarray(obs))


def viper(vec, teacher, weight, args, key):
    """Returns the trees, their training accuracies and their scores."""
    key, reset_key = jax.random.split(key)
    carry = vec.reset(jax.random.split(reset_key, args.n_envs))
    states, carry = rollout(vec, teacher, args.samples_per_iter, carry)
    actions, weights = teacher(states), weight(states)
    trees, accuracies, scores = [], [], []
    t0 = time.time()
    for it in range(args.n_iter):
        tree = DecisionTreeClassifier(max_depth=args.max_depth, random_state=args.seed)
        tree.fit(states, actions, sample_weight=weights)
        returns, key = score(vec, tree_policy(tree), args.score_episodes, args.max_steps, key)
        trees.append(tree)
        accuracies.append(tree.score(states, actions, sample_weight=weights))
        scores.append(mean(returns))
        print(f"  iter {it + 1:2d}/{args.n_iter}  accuracy={accuracies[-1]:.4f}  "
              f"return={scores[-1]:+8.2f}  ({len(returns)}/{args.score_episodes} episodes, "
              f"{time.time() - t0:.0f}s)")

        # DAgger: the states the tree reaches, labelled with the teacher's action.
        new_states, carry = rollout(vec, tree_policy(tree), args.samples_per_iter, carry)
        states = np.concatenate([states, new_states])
        actions = np.concatenate([actions, teacher(new_states)])
        weights = np.concatenate([weights, weight(new_states)])
    return trees, accuracies, scores


def cmd_extract(args):
    matches = glob.glob(args.model_path)
    if not matches:
        raise SystemExit(f"no checkpoint matches {args.model_path!r}")
    model_path = max(matches, key=os.path.getmtime)  # the newest, when a glob was given
    config, actor_params, _ = load_model(model_path)
    game = config["ENV_ID"]

    env = make_env(game, eval=True)()  # imitates the policy, scores the real game
    vec = VecEnv(env)
    actor = MLP(tuple(config["HIDDEN_DIMS"]), env.action_space().n, 0.01)
    log_pi = jax.jit(lambda obs: jax.nn.log_softmax(actor.apply(actor_params, obs)))

    def teacher(obs):  # greedy teacher action
        return np.asarray(jnp.argmax(log_pi(jnp.asarray(obs)), axis=-1))

    def weight(obs):  # VIPER's Q-disagreement cost
        lp = log_pi(jnp.asarray(obs))
        return np.asarray(lp.mean(-1) - lp.min(-1))

    print(f"{model_path}\n  {game}: {env.n_features} concepts, actions {list(env.action_names)}")
    key = jax.random.PRNGKey(args.seed)
    teacher_returns, key = score(vec, teacher, args.eval_episodes, args.max_steps, key)
    print(f"teacher: {mean(teacher_returns):+.2f} over {len(teacher_returns)} episodes\n"
          f"distilling: {args.n_iter} x {args.samples_per_iter} samples, max depth {args.max_depth}")
    trees, accuracies, scores = viper(vec, teacher, weight, args, key)
    if np.all(np.isnan(scores)):
        raise SystemExit(f"no tree finished an episode within --max-steps {args.max_steps}")
    # the best score; among trees tied for it, the smallest (most readable) one
    top = np.nanmax(scores)
    best = min((i for i, s in enumerate(scores) if s == top), key=lambda i: trees[i].get_n_leaves())

    out_dir = Path(args.out_dir or Path(model_path).parent / "viper")
    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = env.image_space().shape[:2]
    _, low, high = feature_layout(game, float(width), float(height))
    meta = {"game": game, "feature_names": list(env.labels),
            "feature_low": low.tolist(), "feature_high": high.tolist(),
            "action_names": [f"{i}:{a}" for i, a in enumerate(env.action_names)]}
    for i, (tree, acc, s) in enumerate(zip(trees, accuracies, scores)):
        save({"tree": tree, "accuracy": acc, "score": s, **meta}, out_dir / f"tree_{i:02d}_{s:+.2f}.viper")
    best_path = out_dir / f"best_{scores[best]:+.2f}.viper"
    save({"tree": trees[best], "accuracy": accuracies[best], "score": scores[best], **meta}, best_path)

    tree_returns, key = score(vec, tree_policy(trees[best]), args.eval_episodes, args.max_steps, key)
    print(f"best tree: iteration {best + 1}, depth {trees[best].get_depth()}, "
          f"{trees[best].get_n_leaves()} leaves -> {best_path}\n"
          f"teacher {mean(teacher_returns):+.2f}  ->  tree {mean(tree_returns):+.2f} "
          f"over {len(tree_returns)} episodes")


def in_screen_units(tree, low, high):
    """A copy of `tree` whose thresholds read in pixels instead of the policy's [-1, 1].

    Inverts ScobotWrapper's per-feature affine map, which is monotonic, so
    every split sends each state the same way as before.
    """
    tree = copy.deepcopy(tree)
    low, high = np.asarray(low), np.asarray(high)
    split = tree.tree_.feature >= 0
    f = tree.tree_.feature[split]
    tree.tree_.threshold[split] = low[f] + (tree.tree_.threshold[split] + 1) * np.maximum(high[f] - low[f], 1e-6) / 2
    return tree


def cmd_show(args):
    with open(args.tree_path, "rb") as f:
        blob = pickle.load(f)
    tree = in_screen_units(blob["tree"], blob["feature_low"], blob["feature_high"])
    features = blob["feature_names"]
    classes = [blob["action_names"][c] for c in tree.classes_]
    print(f"{blob['game']}: return {blob['score']:+.2f}, depth {tree.get_depth()}, "
          f"{tree.get_n_leaves()} leaves\n")
    print("(thresholds in screen pixels)")
    print(export_text(tree, feature_names=features, class_names=classes,
                      max_depth=args.max_depth or tree.get_depth(), decimals=1))
    print("Most-used concepts:")
    for importance, name in sorted(zip(tree.feature_importances_, features), reverse=True)[:10]:
        if importance > 0:
            print(f"  {importance:.3f}  {name}")

    if args.visualize:
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree

        depth = args.max_depth or tree.get_depth()
        leaves = min(tree.get_n_leaves(), 2 ** depth)
        fig, ax = plt.subplots(figsize=(max(20, leaves * 3.5), max(10, (depth + 1) * 2.5)))
        plot_tree(tree, feature_names=features, class_names=classes, max_depth=args.max_depth,
                  filled=True, rounded=True, impurity=False, precision=1, fontsize=9, ax=ax)
        ax.set_title(f"{blob['game']} VIPER tree (depth {depth} of {tree.get_depth()})")
        out = args.out or args.tree_path + ".png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nsaved -> {out}")


def main():
    parser = argparse.ArgumentParser(description="VIPER for SCoBots agents")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="distill a checkpoint into a decision tree")
    p.add_argument("model_path", help="checkpoint written by agents/scobots (globs allowed)")
    p.add_argument("--out-dir", default=None, help="default: <checkpoint dir>/viper")
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--n-iter", type=int, default=25)
    p.add_argument("--samples-per-iter", type=int, default=30_000)
    p.add_argument("--n-envs", type=int, default=64, help="environments per rollout")
    p.add_argument("--score-episodes", type=int, default=3, help="episodes scoring each tree")
    p.add_argument("--eval-episodes", type=int, default=10, help="teacher vs best tree")
    p.add_argument("--max-steps", type=int, default=27_000, help="step cap per scored episode")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("show", help="print a tree as if-then rules")
    p.add_argument("tree_path")
    p.add_argument("--max-depth", type=int, default=None, help="truncate rules and drawing")
    p.add_argument("--visualize", action="store_true", help="also draw it (needs matplotlib)")
    p.add_argument("--out", default=None, help="PNG path (default: <tree_path>.png)")
    p.set_defaults(func=cmd_show)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
