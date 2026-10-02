from __future__ import annotations

import argparse
from pathlib import Path

from dap.direct_action_planning_dataset_specific_stabilization.baseline_analysis import (
    run_baseline_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-tier", default="dataset_specific_baselines_core_v1")
    parser.add_argument("--dap-tier", default="dataset_specific_stabilization_core_v4")
    parser.add_argument("--analysis-name", default="analysis_v1")
    args = parser.parse_args()
    print(
        run_baseline_analysis(
            args.project_root,
            baseline_tier=args.baseline_tier,
            dap_tier=args.dap_tier,
            config_path=args.config,
            analysis_name=args.analysis_name,
        )
    )


if __name__ == "__main__":
    main()
