from __future__ import annotations

import argparse
from pathlib import Path

from dap.direct_action_planning_context_value.experiment import (
    run_alias_oracle_diagnosis,
    run_minimal_context_value,
)
from dap.direct_action_planning_context_value.analysis import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "research/direct_action_planning_context_value/configs/minimal_validation.yaml"
        ),
    )
    parser.add_argument("--run-id", default="alias_oracle_v1")
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--oracle-run-id", default="alias_oracle_v2")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        output = run_analysis(args.project_root.resolve(), args.run_id)
    elif args.minimal:
        output = run_minimal_context_value(
            args.project_root.resolve(),
            args.project_root.resolve() / args.config,
            args.run_id,
            smoke=args.smoke,
            oracle_run_id=args.oracle_run_id,
        )
    else:
        output = run_alias_oracle_diagnosis(
            args.project_root.resolve(), args.project_root.resolve() / args.config, args.run_id
        )
    print(output)


if __name__ == "__main__":
    main()
