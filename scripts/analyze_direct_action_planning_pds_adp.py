from __future__ import annotations

import argparse
from pathlib import Path

from stage2_dynamic_budget.direct_action_planning_pds_adp.analysis import (
    analyze_temporal_results,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path,
        default=Path("research/direct_action_planning_pds_adp/configs/temporal_test_v1.yaml"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/direct_action_planning_pds_adp/pds_adp_temporal_test_v1_locked/analysis_v1"),
    )
    args = parser.parse_args()
    print(analyze_temporal_results(args.project_root, args.config, args.output))


if __name__ == "__main__":
    main()
