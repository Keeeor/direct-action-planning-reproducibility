from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml

from dap.direct_action_planning_dataset_validation.data import load_trace_dataset
from dap.direct_action_planning_dataset_validation.evaluation import evaluate_methods
from dap.direct_action_planning_dataset_validation.experiment import _train_b4
from dap.utils.artifacts import sha256_file, write_json
from dap.utils.seed import set_global_seed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "research/direct_action_planning_dataset_validation/configs/b4_sensitivity.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = root / "results/direct_action_planning_dataset_validation/b4_sensitivity_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_name in config["datasets"]:
        dataset = load_trace_dataset(root, dataset_name)
        for budget in config["budgets"]:
            for seed in config["seeds"]:
                run_id = f"b4_sensitivity_v1__{dataset_name}__b{budget:.0f}__s{seed}"
                run_dir = output_root / dataset_name / run_id
                if (run_dir / "manifest.json").exists():
                    raise FileExistsError(f"append-only run exists: {run_dir}")
                run_dir.mkdir(parents=True, exist_ok=True)
                set_global_seed(int(seed), torch_threads=1)
                policy, training = _train_b4(dataset, config, float(budget), int(seed))
                episodes, _ = evaluate_methods(
                    dataset,
                    {"b4_102400": policy},
                    split="validation",
                    horizon=int(config["horizon"]),
                    budget=float(budget),
                    seed=int(seed) + 50_000_003,
                    episodes_per_domain=int(config["evaluation_episodes_per_domain"]),
                    gamma=float(config["gamma"]),
                )
                pd.DataFrame(episodes).to_csv(run_dir / "metrics.csv", index=False)
                write_json(
                    run_dir / "runtime.json",
                    {
                        "training_seconds": training.elapsed_seconds,
                        "training_steps": config["b4"]["total_steps"],
                        "global_lambda": training.global_lambda,
                    },
                )
                write_json(
                    run_dir / "manifest.json",
                    {
                        "status": "completed",
                        "run_id": run_id,
                        "validation_only": True,
                        "formal_test_accessed": False,
                        "config_sha256": sha256_file(config_path),
                        "artifacts": {
                            "metrics.csv": sha256_file(run_dir / "metrics.csv"),
                            "runtime.json": sha256_file(run_dir / "runtime.json"),
                        },
                    },
                )
                rows.extend(episodes)
    long_frame = pd.DataFrame(rows)
    gate_paths = sorted((root / "results/direct_action_planning_dataset_validation/gate_v1").glob("*/*/metrics.csv"))
    gate = pd.concat([pd.read_csv(path) for path in gate_paths], ignore_index=True)
    short = gate[(gate.split == "validation") & (gate.method == "b4_budget_state")].copy()
    keys = ["dataset", "domain", "budget", "seed", "episode", "window_seed", "window_start"]
    comparison = short.merge(
        long_frame,
        on=keys,
        suffixes=("_20480", "_102400"),
        validate="one_to_one",
    )
    metrics = ("discounted_return", "completion_ratio", "slo_violation_rate", "total_cost", "queue_area")
    for metric in metrics:
        comparison[f"{metric}_difference"] = comparison[f"{metric}_102400"] - comparison[f"{metric}_20480"]
    comparison.to_csv(output_root / "paired_validation_comparison.csv", index=False)
    difference_columns = [column for column in comparison if column.endswith("_difference")]
    summary = comparison.groupby("dataset")[difference_columns].mean().reset_index()
    summary.to_csv(output_root / "summary.csv", index=False)
    write_json(
        output_root / "manifest.json",
        {
            "status": "completed",
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "run_count": len(config["datasets"]) * len(config["budgets"]) * len(config["seeds"]),
            "validation_only": True,
            "formal_test_accessed": False,
            "artifacts": {
                "paired_validation_comparison.csv": sha256_file(output_root / "paired_validation_comparison.csv"),
                "summary.csv": sha256_file(output_root / "summary.csv"),
            },
        },
    )
    print(output_root)


if __name__ == "__main__":
    main()
