from __future__ import annotations

import argparse
from pathlib import Path

from dap.direct_action_planning_k8s_robustness.runner import (
    freeze_contract,
    run_cell,
    run_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--profile")
    parser.add_argument("--condition")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.freeze:
        if args.config is None:
            parser.error("--config is required with --freeze")
        print(freeze_contract(args.project_root, args.config, args.contract))
    elif args.profile is not None:
        if args.condition is None or args.seed is None:
            parser.error("--condition and --seed are required for one cell")
        print(run_cell(
            args.project_root, args.contract, profile=args.profile,
            condition=args.condition, seed=args.seed,
        ))
    else:
        print(run_matrix(args.project_root, args.contract))


if __name__ == "__main__":
    main()
