#!/usr/bin/env python3
"""
run_all_dqn.py — Distribute DQN experiments across GPUs evenly.

Every (env, config) pair is run in sequence by worker threads, one per GPU.
When a GPU finishes its current experiment, it grabs the next from the queue.

Usage:
    uv run python run_all_dqn.py 0 1 2 3
    uv run python run_all_dqn.py 0,1,2,3
    uv run python run_all_dqn.py 0 1     # only two GPUs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from queue import Queue

# ═══════════════════════════════════════════════════════════════════════════════
#  Configure your experiment grid below
# ═══════════════════════════════════════════════════════════════════════════════

ENVS: list[str] = [
    "freeway", 
    "kangaroo",
    "montezumarevenge",
    "mspacman",
    "phoenix", "pong", "qbert",
    "seaquest", "skiing",
    "tennis",
    "venture",
    "timepilot", "asteroids", "breakout", 
    "frostbite", "gravitar",
    "bankheist",
    "beamrider",
    "enduro", 
]

CONFIGS: list[str] = [
    "dqn_rgb_tuned",
    "dqn_oc_tuned",
    # "dqn_rgb_original",
    # "dqn_oc_original",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  Internals — you shouldn't need to edit below this line
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Experiment:
    env: str
    config: str

    @property
    def label(self) -> str:
        return f"{self.env}/{self.config}"


def build_experiments(envs: Sequence[str], configs: Sequence[str]) -> list[Experiment]:
    return [Experiment(env=e, config=c) for e in envs for c in configs]


def run_experiment(exp: Experiment, gpu_id: int) -> int:
    """Launch a single experiment on *gpu_id* and block until it finishes."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "uv", "run", "python", "main.py",
        f"+alg={exp.config}",
        f"ENV_ID={exp.env}",
    ]

    print(f"[GPU {gpu_id:>3}] ▶  {exp.label}")
    t0 = time.perf_counter()

    proc = subprocess.Popen(cmd, env=env)
    proc.wait()

    elapsed = time.perf_counter() - t0
    if elapsed >= 60:
        ts = f"{elapsed / 60:.1f}m"
    else:
        ts = f"{elapsed:.1f}s"

    icon = "✓" if proc.returncode == 0 else f"✗ (rc={proc.returncode})"
    print(f"[GPU {gpu_id:>3}]    {icon}  {exp.label}  [{ts}]")

    return proc.returncode


def _parse_gpus(raw: Sequence[str]) -> list[int]:
    """Accept '0 1 2 3' or '0,1,2,3'."""
    ids: list[int] = []
    for token in raw:
        ids.extend(int(g) for g in token.split(","))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distribute DQN experiments across GPUs evenly."
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        help="GPU IDs, e.g. '0 1 2 3' or '0,1,2,3'",
    )
    args = parser.parse_args()
    gpu_ids = _parse_gpus(args.gpus)

    experiments = build_experiments(ENVS, CONFIGS)

    print(f"GPUs  : {gpu_ids}")
    print(f"Total : {len(experiments)} experiments  ({len(ENVS)} envs × {len(CONFIGS)} configs)")
    for e in experiments:
        print(f"        {e.label}")
    print()

    # -- shared work queue ----------------------------------------------------
    queue: Queue[Experiment | None] = Queue()
    for exp in experiments:
        queue.put(exp)

    lock = threading.Lock()
    failed: list[str] = []

    def worker(gpu_id: int) -> None:
        while True:
            try:
                exp = queue.get_nowait()
            except Exception:
                return  # no more work
            rc = run_experiment(exp, gpu_id)
            if rc != 0:
                with lock:
                    failed.append(f"{exp.label} (GPU {gpu_id}, rc={rc})")

    threads = []
    for gid in gpu_ids:
        t = threading.Thread(target=worker, args=(gid,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print()
    print("=" * 60)
    if failed:
        print(f"FAILURES ({len(failed)}/{len(experiments)}):")
        for f in failed:
            print(f"  ✗ {f}")
        sys.exit(1)
    else:
        print(f"All {len(experiments)} experiments completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
