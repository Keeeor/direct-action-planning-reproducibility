#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dap.direct_action_planning_support_adaptation.analysis import (
    run_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen Direct Action Planning support-adaptation run."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--run-id", default="minimal_v1")
    args = parser.parse_args()
    print(run_analysis(args.project_root, args.run_id))


if __name__ == "__main__":
    main()
