from __future__ import annotations

import argparse
from pathlib import Path

from dap.direct_action_planning_dataset_specific_stabilization.experiment import (
    run_matrix,
    run_unit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("unit", "matrix"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "matrix":
        outputs = run_matrix(args.project_root, args.config)
        for output in outputs:
            print(output)
        return
    if args.dataset is None or args.budget is None or args.seed is None:
        parser.error("unit mode requires --dataset, --budget, and --seed")
    print(
        run_unit(
            args.project_root,
            args.config,
            dataset_name=args.dataset,
            budget=args.budget,
            seed=args.seed,
            attempt=args.attempt,
        )
    )


if __name__ == "__main__":
    main()
