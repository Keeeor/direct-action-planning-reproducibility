#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dap.experiment import load_config  # noqa: E402
from dap.trace_experiment import run_trace_experiment  # noqa: E402


def _run(task):
    return str(run_trace_experiment(*task))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--training-domains", nargs="*", default=["combined"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--only-missing", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    tasks = []
    for training_domain in args.training_domains:
        for method in config["methods"]:
            for budget in config["budgets"]:
                for seed in config["seeds"]:
                    task = (str(ROOT), str(config_path), method, float(budget), int(seed), training_domain)
                    label = f"{float(budget):.6g}".replace(".", "p")
                    run_id = f"{config['tier']}__train_{training_domain}__{method}__b{label}__s{seed}"
                    if args.only_missing and (ROOT / "results" / "raw_logs" / config["tier"] / run_id / "manifest.json").exists():
                        continue
                    tasks.append(task)
    print(f"tasks={len(tasks)} workers={args.workers}", flush=True)
    completed, failed, started = 0, [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
                completed += 1
                print(f"PASS {task[5]} {task[2]} {task[3]} {task[4]}", flush=True)
            except Exception as exc:
                failed.append({"task": task[2:], "error": repr(exc)})
                print(f"FAIL {task[2:]} {exc!r}", flush=True)
    report = {"tasks": len(tasks), "completed": completed, "failed": failed, "elapsed_seconds": time.perf_counter() - started}
    out = ROOT / "results" / "manifests" / f"pipeline_{config['tier']}_{args.name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
