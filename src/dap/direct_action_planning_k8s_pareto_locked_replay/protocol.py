"""Frozen A3 locked-replay matrix identity."""

from __future__ import annotations

from typing import Any


SCHEMA = "dap.k8s.pareto_locked_replay_runtime.v1"
CANDIDATES = {"dap_cont_0p05": 0.05}
METHODS = ["dap_cont_0p05", "threshold"]
SEEDS = list(range(2026081501, 2026081521))
QUANTILES = [value for value in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00) for _ in range(2)]
ROTATIONS = [index % 2 for index in range(20)]


def validate_runtime_config(config: dict[str, Any]) -> None:
    if config.get("schema") != SCHEMA or config.get("mode") != "locked_replay_a3":
        raise ValueError("unregistered A3 locked-replay schema or mode")
    expected = {
        "source_split": "test",
        "horizon_steps": 32,
        "control_interval_seconds": 5,
        "budget_seconds": 256,
        "max_attempts": 2,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"A3 locked replay {name} drift")
    if list(config.get("methods", [])) != METHODS:
        raise ValueError("A3 locked replay method drift")
    if dict(config.get("continuation_candidates", {})) != CANDIDATES:
        raise ValueError("A3 locked replay candidate drift")
    if [int(value) for value in config.get("seeds", [])] != SEEDS:
        raise ValueError("A3 locked replay seed drift")
    if [int(value) for value in config.get("method_rotation_offsets", [])] != ROTATIONS:
        raise ValueError("A3 locked replay rotation drift")
    profile = dict(config.get("profile", {}))
    if [float(value) for value in profile.get("activity_quantiles", [])] != QUANTILES:
        raise ValueError("A3 locked replay quantile drift")
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
        raise ValueError("A3 locked replay profile/checkpoint drift")
    if config.get("validation_selection") != "../results/analysis/validation_a3_v1/validation_decision.json":
        raise ValueError("A3 locked replay validation selection drift")
    if config.get("kubernetes") != {
        "context": "kind-rl-lab",
        "namespace": "dap-k8s-prototype",
        "deployment": "dap-worker",
        "image": "dap-k8s-service:local",
    }:
        raise ValueError("A3 locked replay Kubernetes target drift")
    if config.get("paths") != {
        "system_model": "../../direct_action_planning_k8s_prototype/results/calibration/system_model.json",
        "plan_root": "../results/request_plans/locked_replay_a3",
        "run_root": "../results/runs/locked_replay_a3",
    }:
        raise ValueError("A3 locked replay path drift")
    if config.get("analysis") != {
        "bootstrap_draws": 20000,
        "bootstrap_seed": 2026081102,
        "completion_loss_margin": 0.01,
        "slo_increase_margin": 0.02,
        "ready_cost_increase_seconds_margin": 10.0,
        "family_correction": "holm",
    }:
        raise ValueError("A3 locked replay analysis drift")


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
    return float(config["profile"]["activity_quantiles"][SEEDS.index(int(seed))])
