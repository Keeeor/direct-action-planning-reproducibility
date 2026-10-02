#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dap.direct_action_planning_paper_evidence.experiment import (
    run_core_unit,
    run_sensitivity_unit,
)
from dap.direct_action_planning_paper_evidence.analysis import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("core", "sensitivity", "analyze"))
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--factor")
    parser.add_argument("--level", type=float)
    args = parser.parse_args()
    if args.mode == "core":
        if args.dataset is None or args.budget is None or args.seed is None:
            parser.error("core requires --dataset, --budget, and --seed")
        output = run_core_unit(
            args.project_root,
            args.config,
            dataset_name=args.dataset,
            budget=args.budget,
            seed=args.seed,
            attempt=args.attempt,
        )
    elif args.mode == "sensitivity":
        if args.factor is None or args.level is None or args.seed is None:
            parser.error("sensitivity requires --factor, --level, and --seed")
        output = run_sensitivity_unit(
            args.project_root,
            args.config,
            factor=args.factor,
            level=args.level,
            seed=args.seed,
            attempt=args.attempt,
        )
    else:
        output = run_analysis(args.project_root, config_path=args.config)
    print(output)


if __name__ == "__main__":
    main()
