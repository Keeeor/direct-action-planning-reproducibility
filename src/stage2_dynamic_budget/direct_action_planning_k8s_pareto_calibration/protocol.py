"""Frozen A3 stage-one protocol validation and matrix identity."""

from __future__ import annotations

from typing import Any


SCHEMA = "dap.k8s.pareto_calibration_runtime.v1"
ALLOWED_CANDIDATES = {
    "dap_cont_0p05": 0.05,
    "dap_cont_0p10": 0.10,
    "dap_cont_0p20": 0.20,
    "dap_cont_0p35": 0.35,
    "dap_cont_0p50": 0.50,
}
EXPECTED_METHODS = [*ALLOWED_CANDIDATES, "threshold"]
EXPECTED_SEEDS = [2026081301, 2026081302, 2026081303, 2026081304]
EXPECTED_QUANTILES = [0.65, 0.80, 0.90, 1.00]
EXPECTED_ROTATION_OFFSETS = [0, 2, 4, 1]


def validate_runtime_config(config: dict[str, Any]) -> None:
    """Fail closed on any method, checkpoint, cost, or system-scope drift."""

    if config.get("schema") != SCHEMA or config.get("mode") != "screen_a3":
        raise ValueError("unregistered A3 schema or mode")
    if "cost_weight" in config:
        raise ValueError("A3 cost_weight is frozen in the checkpoint and may not drift")
    expected_scalars = {
        "source_split": "validation",
        "horizon_steps": 32,
        "control_interval_seconds": 5,
        "budget_seconds": 256,
        "max_attempts": 2,
    }
    for name, expected in expected_scalars.items():
        if config.get(name) != expected:
            raise ValueError(f"A3 {name} drift")
    if list(config.get("methods", [])) != EXPECTED_METHODS:
        raise ValueError("A3 method grid drift")
    if [int(value) for value in config.get("method_rotation_offsets", [])] != EXPECTED_ROTATION_OFFSETS:
        raise ValueError("A3 method execution rotation drift")
    candidates = {
        str(name): float(value)
        for name, value in dict(config.get("continuation_candidates", {})).items()
    }
    if candidates != ALLOWED_CANDIDATES:
        raise ValueError("A3 continuation candidate grid drift")
    if [int(value) for value in config.get("seeds", [])] != EXPECTED_SEEDS:
        raise ValueError("A3 screen seed drift")
    profile = dict(config.get("profile", {}))
    if [float(value) for value in profile.get("activity_quantiles", [])] != EXPECTED_QUANTILES:
        raise ValueError("A3 activity quantile drift")
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
        "activity_quantiles": EXPECTED_QUANTILES,
    }
    if profile != expected_profile:
        raise ValueError("A3 profile or checkpoint drift")
    if config.get("inherited_development_contract") != (
        "../../direct_action_planning_k8s_cost_calibration/contracts/development_v2_contract.json"
    ):
        raise ValueError("A3 inherited development contract drift")
    if config.get("kubernetes") != {
        "context": "kind-rl-lab",
        "namespace": "dap-k8s-prototype",
        "deployment": "dap-worker",
        "image": "dap-k8s-service:local",
    }:
        raise ValueError("A3 Kubernetes target drift")
    if config.get("paths") != {
        "system_model": "../../direct_action_planning_k8s_prototype/results/calibration/system_model.json",
        "plan_root": "../results/request_plans/screen_a3",
        "run_root": "../results/runs/screen_a3",
    }:
        raise ValueError("A3 output path drift")


def matrix_cells(config: dict[str, Any]) -> list[tuple[str, str, int]]:
    validate_runtime_config(config)
    cells: list[tuple[str, str, int]] = []
    methods = [str(value) for value in config["methods"]]
    for seed, offset in zip(
        config["seeds"], config["method_rotation_offsets"], strict=True
    ):
        shift = int(offset)
        rotated = methods[shift:] + methods[:shift]
        cells.extend(
            ("gentd_inference", method, int(seed)) for method in rotated
        )
    return cells


def activity_quantile_for(config: dict[str, Any], seed: int) -> float:
    validate_runtime_config(config)
    seeds = [int(value) for value in config["seeds"]]
    try:
        index = seeds.index(int(seed))
    except ValueError as error:
        raise ValueError("unregistered A3 seed") from error
    return float(config["profile"]["activity_quantiles"][index])
