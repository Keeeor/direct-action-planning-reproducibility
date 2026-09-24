from __future__ import annotations

import argparse
from pathlib import Path

from stage2_dynamic_budget.direct_action_planning_dataset_benchmark.analysis import (
    run_analysis,
    run_claim_centered_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="external_baselines_development_v1")
    parser.add_argument("--analysis", default="analysis_v1")
    parser.add_argument("--correction-tier")
    parser.add_argument("--claim-centered", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    analysis = run_claim_centered_analysis if args.claim_centered else run_analysis
    print(analysis(root, args.tier, args.analysis, args.correction_tier))


if __name__ == "__main__":
    main()
