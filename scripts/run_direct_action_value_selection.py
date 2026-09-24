#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from stage2_dynamic_budget.direct_action_value_selection.experiment import (
    run_minimal_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Direct Action Value Selection")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default="minimal_v1")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    print(
        run_minimal_validation(
            args.project_root, args.config, run_id=args.run_id, smoke=args.smoke
        )
    )


if __name__ == "__main__":
    main()
