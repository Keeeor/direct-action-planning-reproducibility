from __future__ import annotations

import argparse
import faulthandler
from pathlib import Path

from stage2_dynamic_budget.action_conditioned_budget_advantage.experiment import (
    run_minimal_validation,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "research/action_conditioned_budget_advantage/configs/minimal_validation.yaml"
)


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser(description="Run the independent ACBA research branch")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", default="minimal_v1")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = run_minimal_validation(ROOT, args.config, args.run_id, smoke=args.smoke)
    print(output)


if __name__ == "__main__":
    main()
