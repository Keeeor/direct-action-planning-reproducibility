from __future__ import annotations

import argparse
from pathlib import Path

from stage2_dynamic_budget.direct_action_planning_support_adaptation.experiment import (
    run_support_adaptation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "research/direct_action_planning_support_adaptation/configs/minimal_validation.yaml"
        ),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = run_support_adaptation(
        args.project_root,
        args.config,
        run_id=args.run_id,
        smoke=args.smoke,
    )
    print(output)


if __name__ == "__main__":
    main()
