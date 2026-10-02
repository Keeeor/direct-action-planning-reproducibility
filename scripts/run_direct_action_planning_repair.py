from __future__ import annotations

import argparse
from pathlib import Path

from dap.direct_action_planning_repair import run_legacy_diagnostic
from dap.direct_action_planning_repair.analysis import run_analysis
from dap.direct_action_planning_repair.benchmark import (
    run_cold_planning_benchmark,
)
from dap.direct_action_planning_repair.experiment import run_minimal_repair


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated Direct Action Planning repair tasks")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--task",
        choices=("legacy-diagnostic", "minimal", "analysis", "benchmark"),
        default="legacy-diagnostic",
    )
    parser.add_argument("--run-id", default="legacy_error_diagnostic_v1")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("research/direct_action_planning_repair/configs/minimal_validation.yaml"),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.task == "legacy-diagnostic":
        print(run_legacy_diagnostic(args.project_root, args.run_id))
    elif args.task == "minimal":
        print(
            run_minimal_repair(
                args.project_root, args.config, run_id=args.run_id, smoke=args.smoke
            )
        )
    elif args.task == "analysis":
        print(run_analysis(args.project_root, args.run_id))
    elif args.task == "benchmark":
        print(run_cold_planning_benchmark(args.project_root, args.run_id))


if __name__ == "__main__":
    main()
