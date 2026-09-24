from __future__ import annotations

import argparse
from pathlib import Path

from stage2_dynamic_budget.dynamic_shadow_price.hard_coupling_experiment import (
    run_hard_coupling_experiment,
)
from stage2_dynamic_budget.dynamic_shadow_price.synthetic_experiment import (
    run_synthetic_joint_experiment,
)
from stage2_dynamic_budget.dynamic_shadow_price.trace_experiment import (
    run_trace_joint_experiment,
)
from stage2_dynamic_budget.dynamic_shadow_price.dp_experiment import (
    run_dp_joint_experiment,
    run_dp_learning_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    hard = subparsers.add_parser("hard-coupling")
    hard.add_argument("--mode", choices=("d1", "d2"), required=True)
    hard.add_argument("--budget", type=float, required=True)
    hard.add_argument("--seed", type=int, required=True)
    hard.add_argument("--total-steps", type=int)
    hard.add_argument(
        "--config",
        default="research/dynamic_shadow_price/configs/hard_coupling.yaml",
    )
    dp = subparsers.add_parser("dp-learning")
    dp.add_argument("--method", choices=("b4_budget_state", "cdba", "dsp_a", "dsp_b"), required=True)
    dp.add_argument("--scenario", choices=("early_burst", "late_burst"), required=True)
    dp.add_argument("--budget", type=int, required=True)
    dp.add_argument("--seed", type=int, required=True)
    dp.add_argument("--variant", default="default")
    dp.add_argument("--alpha", type=float)
    dp.add_argument("--delta-budget-ratio", type=float)
    dp.add_argument("--monotonic-coef", type=float)
    dp.add_argument("--total-steps", type=int)
    dp.add_argument("--config", default="research/dynamic_shadow_price/configs/dp_learning.yaml")
    joint = subparsers.add_parser("dp-joint")
    joint.add_argument("--method", choices=("b4_budget_state", "cdba", "dsp_a", "dsp_b"), required=True)
    joint.add_argument("--scenario", choices=("early_burst", "late_burst"), required=True)
    joint.add_argument("--seed", type=int, required=True)
    joint.add_argument("--variant", default="joint_default")
    joint.add_argument("--alpha", type=float)
    joint.add_argument("--delta-budget-ratio", type=float)
    joint.add_argument("--monotonic-coef", type=float)
    joint.add_argument("--total-steps", type=int)
    joint.add_argument("--config", default="research/dynamic_shadow_price/configs/dp_learning.yaml")
    synthetic = subparsers.add_parser("synthetic-joint")
    synthetic.add_argument("--method", choices=("b4_joint_hard", "dsp_a", "dsp_b"), required=True)
    synthetic.add_argument("--seed", type=int, required=True)
    synthetic.add_argument("--variant", default="formal")
    synthetic.add_argument("--total-steps", type=int)
    synthetic.add_argument("--alpha", type=float)
    synthetic.add_argument("--delta-budget-ratio", type=float)
    synthetic.add_argument("--monotonic-coef", type=float)
    synthetic.add_argument("--fixed-shadow-price", type=float)
    synthetic.add_argument("--no-actor-budget", action="store_true")
    synthetic.add_argument("--no-actor-horizon", action="store_true")
    synthetic.add_argument("--config", default="research/dynamic_shadow_price/configs/synthetic_joint.yaml")
    trace = subparsers.add_parser("trace-joint")
    trace.add_argument("--method", choices=("b4_joint_hard", "dsp_a", "dsp_b"), required=True)
    trace.add_argument("--seed", type=int, required=True)
    trace.add_argument("--variant", default="formal")
    trace.add_argument("--total-steps", type=int)
    trace.add_argument("--config", default="research/dynamic_shadow_price/configs/trace_joint.yaml")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if args.command == "hard-coupling":
        run_dir = run_hard_coupling_experiment(
            project_root,
            project_root / args.config,
            args.mode,
            args.budget,
            args.seed,
            args.total_steps,
        )
        print(run_dir)
    elif args.command == "dp-learning":
        overrides = {
            key: value
            for key, value in {
                "alpha": args.alpha,
                "delta_budget_ratio": args.delta_budget_ratio,
                "monotonic_coef": args.monotonic_coef,
                "total_steps": args.total_steps,
            }.items()
            if value is not None
        }
        run_dir = run_dp_learning_experiment(
            project_root,
            project_root / args.config,
            args.method,
            args.scenario,
            args.budget,
            args.seed,
            args.variant,
            overrides,
        )
        print(run_dir)
    elif args.command == "dp-joint":
        overrides = {
            key: value
            for key, value in {
                "alpha": args.alpha,
                "delta_budget_ratio": args.delta_budget_ratio,
                "monotonic_coef": args.monotonic_coef,
                "total_steps": args.total_steps,
            }.items()
            if value is not None
        }
        run_dir = run_dp_joint_experiment(
            project_root,
            project_root / args.config,
            args.method,
            args.scenario,
            args.seed,
            args.variant,
            overrides,
        )
        print(run_dir)
    elif args.command == "synthetic-joint":
        overrides = {
            key: value
            for key, value in {
                "total_steps": args.total_steps,
                "alpha": args.alpha,
                "delta_budget_ratio": args.delta_budget_ratio,
                "monotonic_coef": args.monotonic_coef,
                "fixed_shadow_price": args.fixed_shadow_price,
                "actor_use_budget_state": False if args.no_actor_budget else None,
                "actor_use_horizon": False if args.no_actor_horizon else None,
            }.items()
            if value is not None
        }
        run_dir = run_synthetic_joint_experiment(
            project_root,
            project_root / args.config,
            args.method,
            args.seed,
            args.variant,
            overrides,
        )
        print(run_dir)
    elif args.command == "trace-joint":
        run_dir = run_trace_joint_experiment(
            project_root,
            project_root / args.config,
            args.method,
            args.seed,
            args.variant,
            args.total_steps,
        )
        print(run_dir)


if __name__ == "__main__":
    main()
