from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("dap", "baselines", "controls"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    script = {
        "dap": "scripts/run_direct_action_planning_paper_closure.py",
        "baselines": "scripts/run_direct_action_planning_paper_closure_baselines.py",
        "controls": "scripts/run_direct_action_planning_paper_closure_controls.py",
    }[args.kind]
    log_dir = args.project_root / "results/direct_action_planning_paper_closure/parallel_logs" / str(config["tier"])
    log_dir.mkdir(parents=True, exist_ok=True)

    def run_dir_for(cell):
        dataset, budget, seed = cell
        name = f"{config['tier']}__{dataset}__b{float(budget):.0f}__s{int(seed)}"
        return args.project_root / "results/direct_action_planning_paper_closure" / str(config["tier"]) / str(dataset) / name

    def completed_cell(cell):
        manifest_path = run_dir_for(cell) / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return manifest.get("status") == "completed" and manifest.get("formal_test_accessed") is False

    def run_cell(cell):
        dataset, budget, seed = cell
        label = f"{dataset}__b{budget:.0f}__s{seed}"
        log_path = log_dir / f"{label}.log"
        command = [
            sys.executable, script, "unit", "--project-root", str(args.project_root),
            "--config", str(args.config), "--dataset", str(dataset),
            "--budget", str(float(budget)), "--seed", str(int(seed)),
        ]
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=args.project_root, stdout=handle, stderr=subprocess.STDOUT, text=True)
        return label, completed.returncode

    all_cells = [(dataset, budget, seed) for dataset in config["datasets"] for budget in config["budgets"] for seed in config["seeds"]]
    cells = [cell for cell in all_cells if not completed_cell(cell)]
    skipped = [cell for cell in all_cells if completed_cell(cell)]
    for dataset, budget, seed in skipped:
        print(f"{dataset}__b{float(budget):.0f}__s{int(seed)}: SKIP completed", flush=True)
    if not cells:
        return
    failures = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = [pool.submit(run_cell, cell) for cell in cells]
        for future in as_completed(futures):
            label, code = future.result()
            print(f"{label}: {'PASS' if code == 0 else 'FAIL'}", flush=True)
            if code != 0:
                failures.append(label)
    if failures:
        raise SystemExit(f"failed cells: {', '.join(sorted(failures))}")


if __name__ == "__main__":
    main()
