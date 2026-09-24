from __future__ import annotations

import argparse
from pathlib import Path

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import ACBADPConfig
from stage2_dynamic_budget.direct_action_planning_controlled_aggregation.diagnosis import (
    diagnose_frozen_aggregation,
)
from stage2_dynamic_budget.direct_action_planning_controlled_aggregation.experiment import (
    run_minimal_controlled_aggregation,
)
from stage2_dynamic_budget.direct_action_planning_controlled_aggregation.analysis import (
    run_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--run-id", default="minimal_v1")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "research/direct_action_planning_controlled_aggregation/configs/minimal_validation.yaml"
        ),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.diagnose:
        output = root / "results/direct_action_planning_controlled_aggregation/diagnosis_v1"
        if output.exists():
            raise RuntimeError(f"append-only output already exists: {output}")
        output.mkdir(parents=True)
        result = diagnose_frozen_aggregation(
            root / "results/direct_action_planning_repair/minimal_v1",
            ACBADPConfig(
                horizon=16,
                max_budget=12,
                max_queue=6,
                scenario="early_burst",
                gamma=0.99,
                action_costs=(0, 1, 2, 3),
                action_capacity=(1, 2, 3, 4),
            ),
        )
        result.rounds.to_csv(output / "round_diagnostics.csv", index=False)
        result.distributions.to_csv(output / "distribution_diagnostics.csv", index=False)
        result.anchor_errors.to_csv(output / "anchor_errors.csv", index=False)
        result.mechanisms.to_csv(output / "mechanism_assessment.csv", index=False)
        print(output)
        return
    if args.analyze:
        print(run_analysis(root, run_id=args.run_id))
        return
    output = run_minimal_controlled_aggregation(
        root,
        root / args.config,
        run_id=args.run_id,
        smoke=args.smoke,
    )
    print(output)


if __name__ == "__main__":
    main()
