from __future__ import annotations

import argparse
from pathlib import Path

from stage2_dynamic_budget.direct_action_planning_dataset_specific_stabilization.analysis import (
    analyze_core,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(analyze_core(args.project_root, args.config))


if __name__ == "__main__":
    main()
