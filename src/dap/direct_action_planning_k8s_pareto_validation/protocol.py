"""Frozen independent validation matrix and decision boundary."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


SCHEMA = "dap.k8s.pareto_validation_runtime.v1"
CANDIDATES = {"dap_cont_0p05": 0.05, "dap_cont_0p10": 0.10}
METHODS = [*CANDIDATES, "threshold"]
SEEDS = [2026081401, 2026081402, 2026081403, 2026081404, 2026081405, 2026081406]
QUANTILES = [0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
ROTATIONS = [0, 1, 2, 1, 0, 2]


def validate_runtime_config(config: dict[str, Any]) -> None:
    if config.get("schema") != SCHEMA or config.get("mode") != "validation_a3":
        raise ValueError("unregistered A3 validation schema or mode")
    expected = {
        "source_split": "validation",
        "horizon_steps": 32,
        "control_interval_seconds": 5,
        "budget_seconds": 256,
        "max_attempts": 2,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"A3 validation {name} drift")
    if list(config.get("methods", [])) != METHODS:
        raise ValueError("A3 validation method drift")
    if dict(config.get("continuation_candidates", {})) != CANDIDATES:
        raise ValueError("A3 validation candidate drift")
    if [int(value) for value in config.get("seeds", [])] != SEEDS:
        raise ValueError("A3 validation seed drift")
    if [int(value) for value in config.get("method_rotation_offsets", [])] != ROTATIONS:
        raise ValueError("A3 validation rotation drift")
    profile = dict(config.get("profile", {}))
    if [float(value) for value in profile.get("activity_quantiles", [])] != QUANTILES:
        raise ValueError("A3 validation quantile drift")
    expected_profile = {
        "name": "gentd_inference",
        "dataset": "gentd26",
        "domain": "txt2img",
        "checkpoint": "../../direct_action_planning_k8s_cost_calibration/results/checkpoints_v2/gentd_inference/models.pt",
        "slo_seconds": 1.0,
        "capacity_per_pod_rps": 29.97526124225415,
        "target_peak_rps": 80,
        "max_rps": 140,
        "training_quantile": 0.99,
        "activity_quantiles": QUANTILES,
    }
    if profile != expected_profile:
        raise ValueError("A3 validation profile/checkpoint drift")
    if config.get("screen_selection") != "../results/analysis/screen_a3_v1/screen_ranking.json":
        raise ValueError("A3 validation screen selection drift")
    if config.get("kubernetes") != {
        "context": "kind-rl-lab",
        "namespace": "dap-k8s-prototype",
        "deployment": "dap-worker",
        "image": "dap-k8s-service:local",
    }:
        raise ValueError("A3 validation Kubernetes target drift")
    if config.get("paths") != {
        "system_model": "../../direct_action_planning_k8s_prototype/results/calibration/system_model.json",
        "plan_root": "../results/request_plans/validation_a3",
        "run_root": "../results/runs/validation_a3",
    }:
        raise ValueError("A3 validation path drift")


def matrix_cells(config: dict[str, Any]) -> list[tuple[str, str, int]]:
    validate_runtime_config(config)
    output: list[tuple[str, str, int]] = []
    methods = list(config["methods"])
    for seed, offset in zip(config["seeds"], config["method_rotation_offsets"], strict=True):
        rotated = methods[int(offset):] + methods[: int(offset)]
        output.extend(("gentd_inference", method, int(seed)) for method in rotated)
    return output


def activity_quantile_for(config: dict[str, Any], seed: int) -> float:
    validate_runtime_config(config)
    index = SEEDS.index(int(seed))
    return float(config["profile"]["activity_quantiles"][index])


def select_validation_candidate(
    summaries: list[Mapping[str, Any]], *, selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return at most one candidate under all point guards plus one SESOI."""

    evaluated: list[dict[str, Any]] = []
    for raw in summaries:
        row = dict(raw)
        completion_loss = float(row["completion_loss"])
        slo_increase = float(row["slo_increase"])
        cost_increase = float(row["ready_cost_increase_seconds"])
        budget_violation = float(row["budget_violation_seconds"])
        if not np.isfinite([completion_loss, slo_increase, cost_increase, budget_violation]).all():
            raise ValueError("nonfinite A3 validation summary")
        guards = (
            completion_loss <= float(selection["completion_loss_max"]) + 1.0e-12
            and slo_increase <= float(selection["slo_increase_max"]) + 1.0e-12
            and cost_increase <= float(selection["ready_cost_increase_seconds_max"]) + 1.0e-12
            and budget_violation <= 1.0e-9
        )
        sesoi = {
            "completion": -completion_loss >= float(selection["completion_gain_sesoi"]),
            "slo": -slo_increase >= float(selection["slo_reduction_sesoi"]),
            "ready_cost": -cost_increase
            >= float(selection["ready_cost_reduction_seconds_sesoi"]),
        }
        row["passed_all_point_guards"] = bool(guards)
        row["sesoi_pass"] = sesoi
        row["sesoi_count"] = sum(bool(value) for value in sesoi.values())
        row["eligible_for_locked_replay"] = bool(guards and row["sesoi_count"] >= 1)
        margins = (
            float(selection["completion_loss_max"]),
            float(selection["slo_increase_max"]),
            float(selection["ready_cost_increase_seconds_max"]),
        )
        row["normalized_guard_violation"] = sum(
            (
                max(completion_loss - margins[0], 0.0) / margins[0],
                max(slo_increase - margins[1], 0.0) / margins[1],
                max(cost_increase - margins[2], 0.0) / margins[2],
                max(budget_violation - 1.0e-9, 0.0) / 1.0e-9,
            )
        )
        evaluated.append(row)
    eligible = [row for row in evaluated if row["eligible_for_locked_replay"]]
    eligible.sort(
        key=lambda row: (
            -int(row["sesoi_count"]),
            float(row["normalized_guard_violation"]),
            float(row["ready_cost_increase_seconds"]),
            float(row["completion_loss"]) + float(row["slo_increase"]),
            float(row["continuation_weight"]),
        )
    )
    return eligible[: int(selection["advance_count"])]
