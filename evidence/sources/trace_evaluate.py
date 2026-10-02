"""Append-only server evaluation using existing frozen DAP/PDS checkpoints.

This extension does not retrain or select on the historically accessed test split.
All algorithms use the original environment and the same registered windows.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def source_dir(family, tier, dataset, budget, seed):
    return ROOT / "results" / family / tier / dataset / f"{tier}__{dataset}__b{budget}__s{seed}"


def weighted_quantile(values, weights, quantile):
    import numpy as np
    if not values:
        return 0.0
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    return float(v[min(int(np.searchsorted(np.cumsum(w), quantile * np.sum(w))), len(v) - 1)])


def evaluate_cell(cell, smoke=False):
    import numpy as np
    import pandas as pd
    import torch
    from collections import deque
    from dap.direct_action_planning_dataset_validation.data import load_trace_dataset
    from dap.direct_action_planning_paper_closure.environment import make_calibrated_trace_env
    from dap.direct_action_planning_paper_closure.control_experiment import _load_source_components
    from dap.direct_action_planning_paper_closure.planning import make_scaled_planner
    from dap.direct_action_planning_paper_closure.controls import make_mpc_planner
    from dap.direct_action_planning_pds_adp.temporal_evaluation import load_pds_planner
    torch.set_num_threads(1)
    dataset_name, budget, seed = cell
    cfg = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))["trace"]
    output = HERE / ("trace_smoke" if smoke else "trace_runs") / f"{dataset_name}__b{budget}__s{seed}"
    if (output / "manifest.json").exists():
        m = json.loads((output / "manifest.json").read_text())
        if m.get("status") == "completed":
            return {"cell": cell, "status": "already_completed"}
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write_json(output / "started.json", {"cell": cell, "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()})
    try:
        source = source_dir("direct_action_planning_paper_closure", cfg["source_dap_tier"], *cell)
        pds_source = source_dir("direct_action_planning_pds_adp", cfg["source_pds_tier"], *cell)
        value, final_value, forecast, selected = _load_source_components(source, hidden_dim=64)
        pds, pds_meta = load_pds_planner(pds_source, gamma=cfg["gamma"], hidden_dim=64)
        methods = {
            "dap": make_scaled_planner(value=value, forecaster=forecast, gamma=cfg["gamma"], continuation_weight=selected["continuation_weight"]),
            "immediate": make_scaled_planner(value=final_value, forecaster=forecast, gamma=cfg["gamma"], continuation_weight=0.0),
            "pds_adp": pds,
            "mpc_4": make_mpc_planner(forecaster=forecast, gamma=cfg["gamma"], horizon=4),
            "mpc_8": make_mpc_planner(forecaster=forecast, gamma=cfg["gamma"], horizon=8),
        }
        dataset = load_trace_dataset(ROOT, dataset_name)
        split = "validation" if smoke else "test"
        episodes = 1 if smoke else cfg["episodes_per_domain"]
        rows, steps = [], []
        for di, domain in enumerate(dataset.domain_names):
            for episode in range(episodes):
                window_seed = seed + cfg["test_seed_offset"] + di * 1000003 + episode * 100003
                for method, planner in methods.items():
                    env, start, _ = make_calibrated_trace_env(dataset, domain, split, horizon=cfg["horizon"], budget=budget, window_seed=window_seed, quantile=cfg["capacity_training_quantile"])
                    obs, _ = env.reset(seed=window_seed)
                    fifo = deque()
                    waits, masses = [], []
                    rewards, arrivals, served, drops, queues, slos, clearances, times = [], [], [], [], [], [], [], []
                    binding = 0
                    for step in range(cfg["horizon"]):
                        prior = float(env.queue)
                        remaining = max(float(budget - env.cumulative_cost), 0.0)
                        feasible = env.action_costs <= remaining + 1e-8
                        binding += int(not bool(feasible.all()))
                        tic = time.perf_counter()
                        action, scores = planner(env, obs)
                        times.append((time.perf_counter() - tic) * 1000)
                        if not feasible[action]:
                            raise AssertionError("unaffordable selected action")
                        obs, reward, done, truncated, info = env.step(action)
                        x, y, q = float(info["arrivals"]), float(info["served"]), float(info["queue_length"])
                        overflow = max(prior + x - y - q, 0.0)
                        clearance = y / max(prior + x, 1.0)
                        if x > 0:
                            fifo.append([step, x])
                        left = y
                        while left > 1e-8 and fifo:
                            amount = min(left, fifo[0][1])
                            waits.append(step - fifo[0][0] + 1)
                            masses.append(amount)
                            fifo[0][1] -= amount
                            left -= amount
                            if fifo[0][1] <= 1e-8:
                                fifo.popleft()
                        left = overflow
                        while left > 1e-8 and fifo:
                            amount = min(left, fifo[-1][1])
                            fifo[-1][1] -= amount
                            left -= amount
                            if fifo[-1][1] <= 1e-8:
                                fifo.pop()
                        if abs(sum(entry[1] for entry in fifo) - q) > 1e-6:
                            raise AssertionError("FIFO reconstruction violates aggregate queue")
                        rewards.append(float(reward)); arrivals.append(x); served.append(y)
                        drops.append(overflow); queues.append(q); slos.append(int(info["slo_violation"])); clearances.append(clearance)
                        steps.append({"dataset": dataset_name, "budget": budget, "training_seed": seed, "domain": domain, "episode": episode, "method": method, "step": step, "window_start": start, "window_seed": window_seed, "action": int(action), "arrivals": x, "served": y, "prior_queue": prior, "queue": q, "overflow": overflow, "clearance": clearance, "cost": float(info["resource_cost"]), "remaining_budget": float(info["remaining_budget"]), "reward": float(reward), "proxy_tail_latency": float(info["tail_latency"]), "slo_violation": int(info["slo_violation"]), "decision_ms": times[-1], "mask_binding": int(not bool(feasible.all()))})
                        if done or truncated:
                            break
                    denom = max(sum(arrivals), 1.0)
                    residual = sum(arrivals) - sum(served) - sum(drops) - queues[-1]
                    if abs(residual) > 1e-6:
                        raise AssertionError(f"mass conservation residual {residual}")
                    rows.append({"dataset": dataset_name, "budget": budget, "training_seed": seed, "domain": domain, "episode": episode, "method": method, "split": split, "window_start": start, "window_seed": window_seed, "discounted_return": float(np.dot(np.power(cfg["gamma"], np.arange(len(rewards))), rewards)), "completion_ratio": sum(served) / denom, "mean_step_clearance": float(np.mean(clearances)), "slo_violation_rate": float(np.mean(slos)), "total_cost": float(env.cumulative_cost), "queue_area": sum(queues), "total_arrivals": sum(arrivals), "total_served": sum(served), "overflow": sum(drops), "overflow_ratio": sum(drops) / denom, "final_queue": queues[-1], "unfinished_ratio": queues[-1] / denom, "conservation_residual": residual, "fifo_p95_wait_steps": weighted_quantile(waits, masses, 0.95), "decision_ms_mean": float(np.mean(times)), "decision_ms_p95": float(np.quantile(times, .95)), "mask_binding_fraction": binding / len(rewards), "overspend": max(float(env.cumulative_cost) - budget, 0)})
        pd.DataFrame(rows).to_csv(output / "episodes.csv", index=False)
        pd.DataFrame(steps).to_csv(output / "steps.csv.gz", index=False, compression="gzip")
        write_json(output / "selection.json", {"dap": selected, "pds": pds_meta})
        write_json(output / "manifest.json", {"status": "completed", "cell": cell, "split": split, "new_untouched_holdout": False, "source_dap_sha256": sha(source / "models.pt"), "source_pds_sha256": sha(pds_source / "models.pt"), "protocol_sha256": sha(HERE / "protocol.json"), "script_sha256": sha(__file__), "wall_seconds": time.perf_counter() - started, "episodes": len(rows), "steps": len(steps), "completed_utc": datetime.now(timezone.utc).isoformat(), "artifacts": {name: sha(output / name) for name in ["episodes.csv", "steps.csv.gz", "selection.json"]}, "fifo_scope": "Post-hoc fluid FIFO/drop-tail diagnostic; not a measured request-level latency."})
        return {"cell": cell, "status": "completed", "seconds": round(time.perf_counter() - started, 2)}
    except Exception:
        write_json(output / "failure.json", {"status": "failed", "cell": cell, "traceback": traceback.format_exc(), "utc": datetime.now(timezone.utc).isoformat()})
        raise


def freeze_inputs(cells):
    cfg = json.loads((HERE / "protocol.json").read_text())["trace"]
    paths = [HERE / "protocol.json", Path(__file__)]
    for cell in cells:
        for family, tier in [("direct_action_planning_paper_closure", cfg["source_dap_tier"]), ("direct_action_planning_pds_adp", cfg["source_pds_tier"])]:
            directory = source_dir(family, tier, *cell)
            paths.extend(directory / name for name in ["models.pt", "diagnostics.json", "manifest.json"])
    paths.extend(sorted((ROOT / "src").rglob("*.py")))
    paths.extend(sorted((ROOT / "data/processed").rglob("*.npz")))
    inventory = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    target = HERE / "trace_frozen_inputs.json"
    if target.exists():
        old = json.loads(target.read_text())
        if old["inventory"] != inventory:
            raise RuntimeError("frozen trace inputs changed")
    else:
        write_json(target, {"frozen_utc": datetime.now(timezone.utc).isoformat(), "historical_test_access": True, "inventory": inventory})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cfg = json.loads((HERE / "protocol.json").read_text())["trace"]
    cells = [(d, b, s) for d in cfg["datasets"] for b in cfg["budgets"] for s in cfg["seeds"]]
    if args.smoke:
        cells = [cells[0]]
    else:
        freeze_inputs(cells)
    failures = []
    with ProcessPoolExecutor(max_workers=min(args.workers, cfg["maximum_workers"])) as pool:
        pending = {pool.submit(evaluate_cell, cell, args.smoke): cell for cell in cells}
        for future in as_completed(pending):
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception:
                cell = pending[future]
                failures.append({"cell": cell, "traceback": traceback.format_exc()})
                print(json.dumps({"cell": cell, "status": "failed"}), flush=True)
    write_json(HERE / ("trace_smoke_status.json" if args.smoke else "trace_status.json"), {"expected_cells": len(cells), "failed_cells": failures, "finished_utc": datetime.now(timezone.utc).isoformat()})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
