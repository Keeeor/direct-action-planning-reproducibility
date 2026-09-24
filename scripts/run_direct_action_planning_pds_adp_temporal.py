from __future__ import annotations

import argparse
from pathlib import Path

from stage2_dynamic_budget.direct_action_planning_pds_adp.temporal_evaluation import (
    freeze_temporal_contract,
    run_temporal_test_matrix,
    run_temporal_test_unit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--template", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.freeze:
        if args.template is None:
            parser.error("--template is required with --freeze")
        print(freeze_temporal_contract(args.project_root, args.template, args.contract))
    elif args.dataset is not None:
        if args.budget is None or args.seed is None:
            parser.error("--budget and --seed are required for one unit")
        print(run_temporal_test_unit(
            args.project_root, args.contract, dataset_name=args.dataset,
            budget=args.budget, seed=args.seed,
        ))
    else:
        for output in run_temporal_test_matrix(
            args.project_root, args.contract, workers=args.workers
        ):
            print(output)


if __name__ == "__main__":
    main()
