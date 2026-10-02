"""Train and select the append-only GenTD DAP cost-calibration extension."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import yaml

from dap.direct_action_planning_dataset_validation.data import (
    load_trace_dataset,
)
from dap.direct_action_planning_dataset_validation.models import (
    FeatureNormalizer,
)
from dap.direct_action_planning_paper_closure.training import (
    compute_value_target_scale,
    train_value_candidates,
)
from dap.direct_action_planning_k8s_service_repair.planner import (
    RuntimeConsistentPlanner,
)
from dap.direct_action_planning_k8s_service_repair.prototype_api import (
    ActionMapper,
    PROJECT_ROOT,
    PROTOTYPE_ROOT,
    save_checkpoint,
)
from dap.direct_action_planning_k8s_service_repair.simulation import (
    evaluate_window,
)
from dap.direct_action_planning_k8s_service_repair.training import (
    CandidateCheckpoint,
    collect_branch_dataset,
    scaled_windows,
    threshold_policy,
    train_forecaster,
)
from dap.direct_action_planning_k8s_service_repair.transition import (
    RuntimeConsistentSystemModel,
)
from dap.utils.artifacts import sha256_file, write_json

from .model import CostWeightedSystemModel
from .selection import SelectionGuards, select_cost_aware_candidate


if str(PROTOTYPE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT))
from workload.trace_converter import (  # noqa: E402
    fit_rate_scale,
    select_activity_window_start,
    transform_rate,
)


METRIC_NAMES = (
    "completion_ratio",
    "slo_violation_rate",
    "ready_cost_seconds",
    "total_reward",
    "target_changes",
    "budget_violation_seconds",
)


def stratified_windows(
    *,
    dataset_name: str,
    domain: str,
    split: str,
    horizon: int,
    activity_quantiles: tuple[float, ...],
    replicates: int,
    seeds: tuple[int, ...],
    target_peak_rps: float,
    max_rps: float,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    """Select locked activity strata using training-fit rate scaling only."""

    if split not in {"validation", "test"}:
        raise ValueError("stratified selection is restricted to validation/test")
    expected = len(activity_quantiles) * int(replicates)
    if int(replicates) <= 0 or len(seeds) != expected:
        raise ValueError("seeds must align with quantiles x replicates")
    if len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("stratified window seeds must be unique")
    dataset = load_trace_dataset(PROJECT_ROOT, dataset_name)
    if domain not in dataset.domains:
        raise ValueError(f"unknown domain {domain!r} for {dataset_name}")
    scale = fit_rate_scale(
        dataset.domains[domain]["train"],
        quantile=0.99,
        target_peak_rps=float(target_peak_rps),
        max_rps=float(max_rps),
    )
    values = np.asarray(dataset.domains[domain][split], dtype=np.float64)
    windows: list[np.ndarray] = []
    starts: list[int] = []
    labels: list[float] = []
    seed_index = 0
    for raw_quantile in activity_quantiles:
        quantile = float(raw_quantile)
        for _ in range(int(replicates)):
            start = select_activity_window_start(
                values,
                horizon=int(horizon),
                activity_quantile=quantile,
                seed=int(seeds[seed_index]),
            )
            seed_index += 1
            windows.append(transform_rate(values[start : start + int(horizon)], scale))
            starts.append(int(start))
            labels.append(quantile)
    return windows, starts, labels


def _aggregate(metrics: Iterable[Any]) -> dict[str, float]:
    rows = list(metrics)
    if not rows:
        raise ValueError("cannot aggregate zero metrics")
    return {
        name: float(np.mean([float(getattr(row, name)) for row in rows]))
        for name in METRIC_NAMES
    }


def aggregate_by_activity(
    metrics: Iterable[Any],
    activity_quantiles: tuple[float, ...] | list[float],
    *,
    low_max: float,
    high_min: float,
) -> dict[str, Any]:
    rows = list(metrics)
    labels = [float(value) for value in activity_quantiles]
    if len(rows) != len(labels) or not rows:
        raise ValueError("metrics and activity labels must be non-empty and aligned")
    low = [row for row, label in zip(rows, labels, strict=True) if label <= low_max]
    high = [row for row, label in zip(rows, labels, strict=True) if label >= high_min]
    if not low or not high:
        raise ValueError("both low- and high-activity strata must be represented")
    return {
        "global": _aggregate(rows),
        "low_activity": _aggregate(low),
        "high_activity": _aggregate(high),
        "stratum_counts": {
            "global": len(rows),
            "low_activity": len(low),
            "high_activity": len(high),
        },
    }


def _evaluate_candidate(
    *,
    checkpoint: CandidateCheckpoint,
    tie_margin: float,
    windows: list[np.ndarray],
    activity_quantiles: list[float],
    budget: float,
    model: CostWeightedSystemModel,
    mapper: ActionMapper,
    interval: float,
    guards: SelectionGuards,
) -> dict[str, Any]:
    planner = RuntimeConsistentPlanner(
        checkpoint=checkpoint,
        system_model=model,
        mapper=mapper,
        control_interval_seconds=interval,
        horizon_steps=len(windows[0]),
        total_budget_seconds=budget,
        tie_margin=tie_margin,
    )
    metrics = [
        evaluate_window(
            rates=window,
            planner=planner,
            system_model=model,
            mapper=mapper,
            total_budget_seconds=budget,
            control_interval_seconds=interval,
        )
        for window in windows
    ]
    return aggregate_by_activity(
        metrics,
        activity_quantiles,
        low_max=guards.low_activity_quantile_max,
        high_min=guards.high_activity_quantile_min,
    )


def _evaluate_threshold(
    *,
    windows: list[np.ndarray],
    activity_quantiles: list[float],
    budget: float,
    model: RuntimeConsistentSystemModel,
    mapper: ActionMapper,
    interval: float,
    queue_thresholds: tuple[float, float, float],
    guards: SelectionGuards,
) -> dict[str, Any]:
    policy = threshold_policy(
        mapper=mapper,
        model=model,
        interval=interval,
        queue_thresholds=queue_thresholds,
    )
    metrics = [
        evaluate_window(
            rates=window,
            policy=policy,
            system_model=model,
            mapper=mapper,
            total_budget_seconds=budget,
            control_interval_seconds=interval,
        )
        for window in windows
    ]
    return aggregate_by_activity(
        metrics,
        activity_quantiles,
        low_max=guards.low_activity_quantile_max,
        high_min=guards.high_activity_quantile_min,
    )


def verify_development_contract(
    *, contract_path: Path, config_path: Path
) -> dict[str, Any]:
    from .audit import verify_development_contract as verify

    return verify(
        project_root=PROJECT_ROOT,
        contract_path=contract_path,
        expected_config_path=config_path,
    )


def train(config_path: Path, contract_path: Path) -> Path:
    config_path = config_path.resolve()
    contract_path = contract_path.resolve()
    contract = verify_development_contract(
        contract_path=contract_path, config_path=config_path
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = config["profile"]
    mapper = ActionMapper(config["actions"])
    base_model = RuntimeConsistentSystemModel.load(
        PROJECT_ROOT / config["paths"]["system_model"],
        source["name"],
        slo_seconds=float(source["slo_seconds"]),
    )
    budgets = tuple(float(value) for value in config["budgets_seconds"])
    interval = float(config["control_interval_seconds"])
    horizon = int(config["horizon_steps"])

    train_windows, train_starts = scaled_windows(
        dataset_name=source["dataset"],
        domain=source["domain"],
        split="train",
        horizon=horizon,
        episodes=int(config["train_episodes"]),
        seed=int(config["seed"]),
        target_peak_rps=float(source["target_peak_rps"]),
        max_rps=float(source["max_rps"]),
    )
    validation_windows, validation_starts, validation_labels = stratified_windows(
        dataset_name=source["dataset"],
        domain=source["domain"],
        split="validation",
        horizon=horizon,
        activity_quantiles=tuple(
            float(value) for value in config["validation_activity_quantiles"]
        ),
        replicates=int(config["validation_replicates"]),
        seeds=tuple(int(value) for value in config["validation_seeds"]),
        target_peak_rps=float(source["target_peak_rps"]),
        max_rps=float(source["max_rps"]),
    )

    legacy_model = CostWeightedSystemModel(base_model, cost_weight=0.05)
    legacy_training = collect_branch_dataset(
        windows=train_windows,
        starts=train_starts,
        profile=source["name"],
        model=legacy_model,
        mapper=mapper,
        budgets=budgets,
        control_interval_seconds=interval,
        seed=int(config["seed"]) + 11,
    )
    legacy_validation = collect_branch_dataset(
        windows=validation_windows,
        starts=validation_starts,
        profile=source["name"],
        model=legacy_model,
        mapper=mapper,
        budgets=budgets,
        control_interval_seconds=interval,
        seed=int(config["seed"]) + 23,
    )
    normalizer = FeatureNormalizer.fit(legacy_training.observations)
    forecaster, forecast_history = train_forecaster(
        legacy_training,
        legacy_validation,
        normalizer,
        seed=int(config["seed"]) + 31,
        epochs=int(config["forecaster_epochs"]),
    )

    value_families: dict[float, dict[int, Any]] = {}
    fvi_histories: dict[str, list[dict[str, float]]] = {}
    dataset_counts: dict[str, dict[str, int]] = {}
    for index, raw_weight in enumerate(config["cost_weights"]):
        cost_weight = float(raw_weight)
        model = CostWeightedSystemModel(base_model, cost_weight=cost_weight)
        if cost_weight == 0.05:
            training = legacy_training
            validation = legacy_validation
        else:
            training = collect_branch_dataset(
                windows=train_windows,
                starts=train_starts,
                profile=source["name"],
                model=model,
                mapper=mapper,
                budgets=budgets,
                control_interval_seconds=interval,
                seed=int(config["seed"]) + 11,
            )
            validation = collect_branch_dataset(
                windows=validation_windows,
                starts=validation_starts,
                profile=source["name"],
                model=model,
                mapper=mapper,
                budgets=budgets,
                control_interval_seconds=interval,
                seed=int(config["seed"]) + 23,
            )
        values, history = train_value_candidates(
            training,
            validation,
            seed=int(config["seed"]) + 37 + index * 1009,
            gamma=float(config["gamma"]),
            iterations=int(config["fvi_iterations"]),
            candidate_iterations=tuple(
                int(value) for value in config["candidate_iterations"]
            ),
            epochs_per_iteration=int(config["fvi_epochs_per_iteration"]),
            learning_rate=float(config["learning_rate"]),
            hidden_dim=int(config["hidden_dim"]),
            target_scale=compute_value_target_scale(training, horizon=horizon),
            zero_initialize_output=True,
        )
        value_families[cost_weight] = values
        fvi_histories[f"{cost_weight:.8g}"] = history
        dataset_counts[f"{cost_weight:.8g}"] = {
            "training_states": training.n_states,
            "validation_states": validation.n_states,
        }

    guards = SelectionGuards.from_mapping(config["selection_guards"])
    selection_budget = float(config["selection_budget_seconds"])
    comparator = _evaluate_threshold(
        windows=validation_windows,
        activity_quantiles=validation_labels,
        budget=selection_budget,
        model=base_model,
        mapper=mapper,
        interval=interval,
        queue_thresholds=tuple(
            float(value) for value in config["baseline_parameters"]["threshold"]
        ),
        guards=guards,
    )

    candidate_rows: list[dict[str, Any]] = []
    for cost_weight, values in value_families.items():
        model = CostWeightedSystemModel(base_model, cost_weight=cost_weight)
        for iteration, value in values.items():
            for raw_continuation in config["continuation_weights"]:
                continuation = float(raw_continuation)
                checkpoint = CandidateCheckpoint(
                    value=value,
                    forecaster=forecaster,
                    gamma=float(config["gamma"]),
                    continuation_weight=continuation,
                    forecast_strategy=str(config["forecast_strategy"]),
                    forecast_multiplier=float(config["forecast_multiplier"]),
                )
                for raw_margin in config["tie_margins"]:
                    margin = float(raw_margin)
                    metrics = _evaluate_candidate(
                        checkpoint=checkpoint,
                        tie_margin=margin,
                        windows=validation_windows,
                        activity_quantiles=validation_labels,
                        budget=selection_budget,
                        model=model,
                        mapper=mapper,
                        interval=interval,
                        guards=guards,
                    )
                    candidate_rows.append({
                        "cost_weight": cost_weight,
                        "iteration": int(iteration),
                        "continuation_weight": continuation,
                        "tie_margin": margin,
                        **metrics,
                    })

    selected = select_cost_aware_candidate(candidate_rows, comparator, guards)
    selected_summary = {
        key: value for key, value in selected.items()
        if key != "selection_diagnostics"
    }
    output_dir = PROJECT_ROOT / config["paths"]["checkpoint_root"] / source["name"]
    checkpoint_path = output_dir / "models.pt"
    diagnostics_path = output_dir / "diagnostics.json"
    if checkpoint_path.exists() or diagnostics_path.exists():
        raise FileExistsError(f"calibration output is append-only: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "dap.k8s.cost_calibration_selection.v1",
        "profile": source["name"],
        "dataset": source["dataset"],
        "domain": source["domain"],
        "training_split": "train",
        "validation_split": "validation",
        "formal_v2_outcomes_previously_accessed": True,
        "new_locked_replay_outcomes_accessed": False,
        "core_method_changed": False,
        "development_contract_sha256": sha256_file(contract_path),
        "development_audit_status": contract["status"],
        "config_sha256": sha256_file(config_path),
        "selection": selected_summary,
        "cost_weight": float(selected["cost_weight"]),
        "tie_margin": float(selected["tie_margin"]),
        "forecast_strategy": str(config["forecast_strategy"]),
        "forecast_multiplier": float(config["forecast_multiplier"]),
    }
    selected_values = value_families[float(selected["cost_weight"])]
    save_checkpoint(
        checkpoint_path,
        value=selected_values[int(selected["iteration"])],
        forecaster=forecaster,
        gamma=float(config["gamma"]),
        continuation_weight=float(selected["continuation_weight"]),
        metadata=metadata,
    )
    write_json(diagnostics_path, {
        **metadata,
        "status": selected["selection_status"],
        "selection_guards": config["selection_guards"],
        "validation_activity_quantiles": validation_labels,
        "validation_window_starts": validation_starts,
        "validation_comparator": comparator,
        "selection_candidates": selected["selection_diagnostics"]["evaluated_candidates"],
        "candidate_count": selected["selection_diagnostics"]["candidate_count"],
        "survivor_count": selected["selection_diagnostics"]["survivor_count"],
        "dataset_counts": dataset_counts,
        "forecaster_history": forecast_history,
        "fvi_history_by_cost_weight": fvi_histories,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    })
    return checkpoint_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    print(train(args.config, args.contract))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

