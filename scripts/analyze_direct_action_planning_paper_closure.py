from __future__ import annotations

import argparse
from pathlib import Path

from dap.direct_action_planning_paper_closure.analysis import frontier_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--analysis-name", default="analysis_v1")
    args = parser.parse_args()
    print(frontier_analysis(args.project_root, args.contract, analysis_name=args.analysis_name))


if __name__ == "__main__":
    main()

