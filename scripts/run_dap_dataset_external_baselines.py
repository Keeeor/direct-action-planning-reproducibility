from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dap.direct_action_planning_dataset_benchmark.experiment import (
    load_protocol,
    run_development_unit,
)
from dap.direct_action_planning_dataset_benchmark.smoke import run_smoke_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research/direct_action_planning_dataset_benchmark/configs/smoke.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    if "smoke" in config_path.name and args.dataset is None:
        for path in run_smoke_matrix(root, config_path):
            print(path)
        return
    config = load_protocol(config_path)
    datasets = [args.dataset] if args.dataset else config["datasets"]
    budgets = [args.budget] if args.budget is not None else config["budgets"]
    seeds = [args.seed] if args.seed is not None else config["seeds"]
    units = [
        (dataset, float(budget), int(seed))
        for dataset in datasets
        for budget in budgets
        for seed in seeds
    ]
    if args.workers <= 1:
        for dataset, budget, seed in units:
            print(run_development_unit(root, config_path, dataset_name=dataset, budget=budget, seed=seed))
        return
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_development_unit,
                root,
                config_path,
                dataset_name=dataset,
                budget=budget,
                seed=seed,
            ): (dataset, budget, seed)
            for dataset, budget, seed in units
        }
        for future in as_completed(futures):
            dataset, budget, seed = futures[future]
            print(f"{dataset} budget={budget:g} seed={seed}: {future.result()}", flush=True)


if __name__ == "__main__":
    main()
