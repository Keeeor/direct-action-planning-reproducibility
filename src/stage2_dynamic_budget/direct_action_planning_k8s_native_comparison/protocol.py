from __future__ import annotations

from pathlib import Path
from typing import Any


METHODS = ("dap_repaired", "hpa", "keda")
PROFILES = ("azure_http", "gentd_inference")
SEEDS = tuple(range(2026080921, 2026080931))


def rotated_methods(seed: int) -> tuple[str, ...]:
    offset = (int(seed) - SEEDS[0]) % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def matrix_cells(config: dict[str, Any]) -> list[tuple[str, str, int]]:
    return [
        (profile, method, int(seed))
        for profile in config["profiles"]
        for seed in config["seeds"]
        for method in rotated_methods(int(seed))
    ]


def resolve(config_path: Path, value: str | Path) -> Path:
    return (config_path.parent / Path(value)).resolve()


def plan_path(config: dict[str, Any], config_path: Path, profile: str, seed: int) -> Path:
    return resolve(config_path, config["paths"]["plan_root"]) / profile / (
        f"{config['source_split']}__seed{int(seed)}.jsonl"
    )


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "dap.k8s.native_comparison.v1":
        raise ValueError("unexpected native-comparison schema")
    expected = {
        "mode": "native_v1",
        "source_split": "test",
        "horizon_steps": 32,
        "control_interval_seconds": 5,
        "budget_seconds": 256,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"native-comparison config drift: {key}")
    if tuple(config.get("profiles", {})) != PROFILES:
        raise ValueError("both profiles are required in fixed order")
    defaults = config.get("controller_defaults", {})
    if float(defaults.get("native_scale_down_reserve_seconds", -1)) != 45.0:
        raise ValueError("native autoscaler reserve drift")
    if config.get("baseline_parameters", {}).get("threshold") != {
        "queue_small": 8,
        "queue_medium": 32,
        "queue_large": 96,
    }:
        raise ValueError("baseline parameter drift")

