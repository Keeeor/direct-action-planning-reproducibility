from __future__ import annotations

import argparse
from pathlib import Path

from stage2_dynamic_budget.direct_action_planning_paper_closure.control_analysis import control_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--analysis-name", default="analysis_v1")
    args = parser.parse_args()
    print(control_analysis(args.project_root, args.config, analysis_name=args.analysis_name))


if __name__ == "__main__":
    main()
