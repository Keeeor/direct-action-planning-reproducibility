from __future__ import annotations

import argparse
from pathlib import Path

from dap.direct_action_planning_pds_adp.experiment import (
    freeze_development_contract,
    run_matrix,
    run_unit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--contract")
    parser.add_argument("--freeze-contract")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    if args.freeze_contract:
        print(freeze_development_contract(root, args.config, args.freeze_contract))
        return
    if args.matrix:
        for path in run_matrix(root, args.config, contract_path=args.contract):
            print(path)
        return
    if args.dataset is None or args.budget is None or args.seed is None:
        parser.error("single-unit execution requires --dataset --budget --seed")
    print(
        run_unit(
            root,
            args.config,
            dataset_name=args.dataset,
            budget=args.budget,
            seed=args.seed,
            contract_path=args.contract,
        )
    )


if __name__ == "__main__":
    main()
