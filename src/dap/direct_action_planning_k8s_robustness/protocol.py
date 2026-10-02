from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONDITIONS = (
    "observation_lag_1", "metric_dropout_10pct", "readiness_delay_5s"
)
PROFILES = ("azure_http", "gentd_inference")


def load_protocol(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema", "tier", "method", "source_split", "horizon_steps",
        "control_interval_seconds", "seeds", "budget_seconds", "conditions",
        "kubernetes", "controller_defaults", "paths", "actions", "profiles",
        "workload", "monitor",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing robustness protocol keys: {sorted(missing)}")
    if config["schema"] != "dap.k8s.robustness_config.v1":
        raise ValueError("invalid robustness schema")
    if config["method"] != "dap" or config["source_split"] != "test":
        raise ValueError("robustness extension is frozen to DAP and the historical test split")
    if tuple(config["profiles"]) != PROFILES:
        raise ValueError(f"profiles must be exactly {PROFILES}")
    if tuple(config["conditions"]) != CONDITIONS:
        raise ValueError(f"conditions must be exactly {CONDITIONS}")
    if tuple(int(seed) for seed in config["seeds"]) != (20260901, 20260902, 20260903):
        raise ValueError("registered seeds changed")
    if int(config["horizon_steps"]) != 64 or float(config["control_interval_seconds"]) != 10.0:
        raise ValueError("registered high-fidelity horizon or control period changed")
    if float(config["budget_seconds"]) != 1024.0:
        raise ValueError("registered medium budget changed")
    return config


def matrix_cells(config: dict[str, Any]) -> list[tuple[str, str, int]]:
    return [
        (str(profile), str(condition), int(seed))
        for profile in config["profiles"]
        for condition in config["conditions"]
        for seed in config["seeds"]
    ]


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    return (Path(config_path).resolve().parent / Path(value)).resolve()
