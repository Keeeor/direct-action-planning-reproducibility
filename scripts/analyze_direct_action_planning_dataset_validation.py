from __future__ import annotations

import argparse
from pathlib import Path

from stage2_dynamic_budget.direct_action_planning_dataset_validation.analysis import (
    run_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tier", default="gate_v1")
    parser.add_argument("--analysis-name", default="analysis_v1")
    args = parser.parse_args()
    print(run_analysis(args.root, tier=args.tier, analysis_name=args.analysis_name))


if __name__ == "__main__":
    main()
