#!/usr/bin/env python3
"""
run_multiple_exp.py — Distribute experiments across GPUs evenly.

Every (env, config) pair is run in sequence by worker threads, one per GPU.
When a GPU finishes its current experiment, it grabs the next from the queue.

A worker grabs the next experiment from the queue and runs it on the first
GPU that is free. A GPU counts as free when nvidia-smi reports no compute
process attached and no significant VRAM in use, and it is not already being
used by this process. If every GPU in the pool is busy, workers hold back
and re-check until one frees up.

Note: the guard coordinates GPUs within one process only. Two separate
invocations of this script started at the exact same instant can still both
pick the same free GPU (there is no cross-process locking).

Usage:
    uv run python run_multiple_exp.py --gpus 0 1 2 3
    uv run python run_multiple_exp.py --gpus 0,1,2,3
    uv run python run_multiple_exp.py --gpus 0 1          # only two GPUs
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
from queue import Empty, Queue

# ═══════════════════════════════════════════════════════════════════════════════
#  Configure your experiment grid below
# ═══════════════════════════════════════════════════════════════════════════════

ENVS: list[str] = [
    # JAXtari-15 split (paper Sec. 3, Fig. 4 / Fig. 8)
    "asteroids",
    "beamrider",
    "breakout",
    "enduro",
    "freeway",
    "frostbite",
    "gravitar",
    "kangaroo",
    "montezumarevenge",
    "mspacman",
    "phoenix",
    "pong",
    "seaquest",
    "skiing",
    "tennis",
]

CONFIGS: list[str] = [
    # "c51_rgb_tuned",
    "c51_oc_tuned",
    # "dqn_rgb_tuned",
    # "dqn_oc_tuned",
    # "rainbow_rgb_tuned",
    # "rainbow_oc_tuned",
    # "dqn_oc_original",
    # "dqn_rgb_original",
    # "rainbow_rgb_original",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  GPU guard — config
# ═══════════════════════════════════════════════════════════════════════════════

# A GPU is considered busy when more than this much VRAM is in use (MiB).
# JAX preallocates ~75% of device memory, so even a partially-used GPU is
# effectively unusable for a second training run.
BUSY_MEM_THRESHOLD_MIB = 1024

# How often (seconds) workers re-check the GPU pool while every GPU is busy.
POLL_INTERVAL_S = 30


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU guard — occupancy detection
# ═══════════════════════════════════════════════════════════════════════════════


def parse_gpu_table(lines: Sequence[str]) -> dict[int, tuple[str, int]]:
    """Parse `nvidia-smi --query-gpu=index,uuid,memory.used` output.

    Returns ``{index: (uuid, memory_used_mib)}``. Malformed lines are skipped.
    """
    table: dict[int, tuple[str, int]] = {}
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            table[int(parts[0])] = (parts[1], int(parts[2]))
        except ValueError:
            continue
    return table


def parse_compute_apps(lines: Sequence[str]) -> set[str]:
    """Parse `nvidia-smi --query-compute-apps=gpu_uuid,pid` output.

    Returns the set of GPU UUIDs that currently have at least one running
    compute process attached. Lines with an empty UUID are ignored.
    """
    uuids: set[str] = set()
    for line in lines:
        uuid = line.split(",", 1)[0].strip()
        if uuid:
            uuids.add(uuid)
    return uuids


def _run_nvidia_smi(args: Sequence[str]) -> list[str] | None:
    """Run nvidia-smi; return split output lines, or None on any failure."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError) as e:
        print(f"[guard] ⚠ nvidia-smi failed ({e}); treating all requested GPUs as unusable")
        return None
    return proc.stdout.strip().splitlines()


def gpu_occupancy(gpu_ids: Sequence[int]) -> dict[int, str]:
    """Classify each requested GPU as ``'free'``, ``'busy'``, or ``'missing'``.

    * ``free``    — no compute process attached and memory.used <= threshold
    * ``busy``    — a compute process is attached or significant VRAM is in use
    * ``missing`` — not present in the nvidia-smi table (bad index / no driver)
    """
    table_lines = _run_nvidia_smi(
        ["--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"]
    )
    if table_lines is None:
        return {g: "missing" for g in gpu_ids}
    table = parse_gpu_table(table_lines)

    app_lines = _run_nvidia_smi(
        ["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]
    )
    busy_uuids = parse_compute_apps(app_lines) if app_lines is not None else set()

    status: dict[int, str] = {}
    for g in gpu_ids:
        entry = table.get(g)
        if entry is None:
            status[g] = "missing"
        elif entry[0] in busy_uuids or entry[1] > BUSY_MEM_THRESHOLD_MIB:
            status[g] = "busy"
        else:
            status[g] = "free"
    return status


def acquire_free_gpu(
    gpu_ids: Sequence[int],
    in_use: set[int],
    guard: threading.Lock,
    poll_interval: float = POLL_INTERVAL_S,
) -> int:
    """Block until a GPU in *gpu_ids* is free, then atomically reserve and return it.

    A GPU is free only if (a) nvidia-smi reports no compute process and low
    VRAM usage, and (b) it is not already reserved in *in_use* by another
    worker of this process. Returns the lowest-index free GPU. The reservation
    is added to *in_use* under *guard*; the caller must discard it when the
    experiment finishes.

    Raises RuntimeError if every pool GPU is reported missing by nvidia-smi
    (driver failure / bad IDs) — that is a failure, not a waiting condition.
    """
    while True:
        with guard:
            candidates = [g for g in gpu_ids if g not in in_use]
        if not candidates:
            time.sleep(poll_interval)  # all GPUs held by our own workers
            continue

        status = gpu_occupancy(candidates)
        free = [g for g in candidates if status[g] == "free"]
        if free:
            with guard:
                for g in free:
                    if g not in in_use:
                        in_use.add(g)
                        return g
            continue  # another worker reserved them all; re-poll

        if all(status[g] == "missing" for g in candidates):
            raise RuntimeError(
                f"GPUs {[g for g in candidates if status[g] == 'missing']} "
                "reported missing by nvidia-smi"
            )
        print(
            f"[guard] all GPUs {list(gpu_ids)} busy — holding back, "
            f"rechecking every {poll_interval:.0f}s (Ctrl-C to abort)",
            flush=True,
        )
        time.sleep(poll_interval)


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
        "ENTITY=null",
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


def detect_gpus() -> list[int]:
    """Return every GPU index nvidia-smi reports, or [] if it cannot be queried."""
    lines = _run_nvidia_smi(["--query-gpu=index", "--format=csv,noheader,nounits"])
    if lines is None:
        return []
    ids: list[int] = []
    for line in lines:
        try:
            ids.append(int(line.strip()))
        except ValueError:
            continue
    return ids


def _parse_gpus(raw: Sequence[str]) -> list[int]:
    """Accept '0 1 2 3' or '0,1,2,3'."""
    ids: list[int] = []
    for token in raw:
        ids.extend(int(g) for g in token.split(","))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distribute experiments across GPUs evenly."
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        help="GPU IDs, e.g. '0 1 2 3' or '0,1,2,3'. Omit to use every GPU nvidia-smi reports.",
    )
    args = parser.parse_args()
    if args.gpus:
        gpu_ids = sorted(set(_parse_gpus(args.gpus)))  # dedupe; threads share the queue anyway
    else:
        gpu_ids = detect_gpus()
        if not gpu_ids:
            parser.error("no GPUs detected by nvidia-smi; pass --gpus explicitly")
        print(f"[guard] --gpus not given; auto-detected {gpu_ids}")

    experiments = build_experiments(ENVS, CONFIGS)
    if not experiments:
        print("No experiments configured (ENVS/CONFIGS are empty).")
        sys.exit(1)

    # -- GPU guard: fail fast on nonexistent GPUs, hold back on busy ones -----
    status = gpu_occupancy(gpu_ids)
    for g in gpu_ids:
        print(f"[guard] GPU {g}: {status[g]}", flush=True)
    missing = [g for g in gpu_ids if status[g] == "missing"]
    if missing:
        print(f"\n[guard] GPUs {missing} not found by nvidia-smi. Fix the GPU list and re-run.")
        sys.exit(1)

    print(f"GPUs  : {gpu_ids}")
    print(f"Total : {len(experiments)} experiments  ({len(ENVS)} envs × {len(CONFIGS)} configs)")
    for e in experiments:
        print(f"        {e.label}")
    print()

    # -- shared work queue ----------------------------------------------------
    queue: Queue[Experiment] = Queue()
    for exp in experiments:
        queue.put(exp)

    lock = threading.Lock()
    failed: list[str] = []
    in_use: set[int] = set()   # GPUs currently reserved by this process
    guard = threading.Lock()   # guards in_use

    def worker() -> None:
        while True:
            try:
                exp = queue.get_nowait()
            except Empty:
                return  # no more work
            try:
                gpu_id = acquire_free_gpu(gpu_ids, in_use, guard)
            except RuntimeError as e:
                with lock:
                    failed.append(f"{exp.label}: {e}")
                return
            try:
                rc = run_experiment(exp, gpu_id)
            finally:
                with guard:
                    in_use.discard(gpu_id)
            if rc != 0:
                with lock:
                    failed.append(f"{exp.label} (GPU {gpu_id}, rc={rc})")

    threads = []
    for _ in gpu_ids:
        t = threading.Thread(target=worker, daemon=True)
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
