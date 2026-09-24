from __future__ import annotations

import argparse
from pathlib import Path

from stage2_dynamic_budget.direct_action_planning_frozen_temporal_test.analysis import (
    analyze_frozen_test,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--result-root")
    parser.add_argument("--analysis-name", default="analysis_v1")
    args = parser.parse_args()
    outputs = analyze_frozen_test(
        Path(args.project_root),
        result_root=(Path(args.result_root) if args.result_root else None),
        analysis_name=args.analysis_name,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
