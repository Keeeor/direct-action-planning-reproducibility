from __future__ import annotations

import argparse
from pathlib import Path

from dap.direct_action_planning_value_refresh.experiment import (
    run_frozen_causal_diagnosis,
    run_minimal_value_refresh,
)
from dap.direct_action_planning_value_refresh.analysis import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "research/direct_action_planning_value_refresh/configs/minimal_validation.yaml"
        ),
    )
    parser.add_argument("--run-id", default="minimal_v1")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--causal-run-id", default="causal_v2")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.analyze:
        output = run_analysis(root, run_id=args.run_id)
    elif args.causal:
        output = run_frozen_causal_diagnosis(root, root / args.config, run_id=args.run_id)
    else:
        output = run_minimal_value_refresh(
            root,
            root / args.config,
            run_id=args.run_id,
            smoke=args.smoke,
            causal_run_id=args.causal_run_id,
        )
    print(output)


if __name__ == "__main__":
    main()
