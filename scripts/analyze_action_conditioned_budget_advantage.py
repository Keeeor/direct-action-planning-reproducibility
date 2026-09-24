from __future__ import annotations

import argparse
from pathlib import Path

from stage2_dynamic_budget.action_conditioned_budget_advantage.analysis import run_analysis


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a completed ACBA validation run")
    parser.add_argument("--run-id", default="minimal_v2")
    args = parser.parse_args()
    output = run_analysis(
        ROOT / "results/action_conditioned_budget_advantage" / args.run_id
    )
    print(output)


if __name__ == "__main__":
    main()
