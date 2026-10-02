from __future__ import annotations

import argparse
from pathlib import Path

from dap.direct_action_planning_paper_closure.temporal_evaluation import (
    freeze_test_contract,
    run_test_matrix,
    run_test_unit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "unit", "matrix"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--template", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "freeze":
        if args.template is None:
            parser.error("freeze requires --template")
        print(freeze_test_contract(args.project_root, args.template, args.contract))
    elif args.mode == "matrix":
        for output in run_test_matrix(args.project_root, args.contract):
            print(output)
    else:
        if args.dataset is None or args.budget is None or args.seed is None:
            parser.error("unit requires --dataset, --budget, and --seed")
        print(run_test_unit(args.project_root, args.contract, dataset_name=args.dataset, budget=args.budget, seed=args.seed, attempt=args.attempt))


if __name__ == "__main__":
    main()

