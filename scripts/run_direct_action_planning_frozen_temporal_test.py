from __future__ import annotations

import argparse
from pathlib import Path

from stage2_dynamic_budget.direct_action_planning_frozen_temporal_test import (
    replay_development_unit,
    run_frozen_test_matrix,
    run_frozen_test_unit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("replay", "unit", "matrix"), required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config = Path(args.config).resolve()
    if args.mode == "matrix":
        for output in run_frozen_test_matrix(root, config):
            print(output)
        return
    if args.dataset is None or args.budget is None or args.seed is None:
        parser.error("replay and unit modes require --dataset, --budget, and --seed")
    if args.mode == "replay":
        print(
            replay_development_unit(
                root,
                config,
                dataset_name=args.dataset,
                budget=args.budget,
                seed=args.seed,
            )
        )
        return
    print(
        run_frozen_test_unit(
            root,
            config,
            dataset_name=args.dataset,
            budget=args.budget,
            seed=args.seed,
            attempt=args.attempt,
        )
    )


if __name__ == "__main__":
    main()

