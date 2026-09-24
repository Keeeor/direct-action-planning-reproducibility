"""Integrity checks and descriptive analysis for the locked robustness matrix."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_completed_runs(run_root: str | Path, contract_sha256: str) -> list[Path]:
    """Select only completed attempts from one frozen robustness contract."""

    root = Path(run_root)
    runs: list[Path] = []
    seen: set[tuple[str, str, int]] = set()
    for path in sorted(root.glob("*/run_manifest.json")):
        manifest = read_json(path)
        if manifest.get("status") != "completed":
            continue
        if manifest.get("robustness_contract_sha256") != contract_sha256:
            continue
        key = (
            str(manifest.get("profile")),
            str(manifest.get("perturbation_condition")),
            int(manifest.get("seed")),
        )
        if key in seen:
            raise ValueError(f"duplicate completed robustness cell: {key}")
        seen.add(key)
        runs.append(path.parent)
    return runs


def _finite_number(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _cell(manifest: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(manifest.get("profile")),
        str(manifest.get("perturbation_condition")),
        int(manifest.get("seed")),
    )


def audit_matrix(
    runs: Iterable[Path], config: dict[str, Any], contract_sha256: str
) -> dict[str, Any]:
    """Audit exact matrix completion and all preregistered safety checks."""

    run_paths = list(runs)
    expected = {
        (str(profile), str(condition), int(seed))
        for profile in config["profiles"]
        for condition in config["conditions"]
        for seed in config["seeds"]
    }
    observed: list[tuple[str, str, int]] = []
    contract_matches = 0
    delivery_passes = 0
    restoration_passes = 0
    split_values: set[str] = set()
    applied_events = 0
    budget_violations: list[float] = []
    deadline_misses = 0
    monitor_failures = 0
    statuses: Counter[str] = Counter()

    for run in run_paths:
        manifest = read_json(run / "run_manifest.json")
        result = read_json(run / "result.json")
        delivery = read_json(run / "perturbation_delivery.json")
        restoration = read_json(run / "readiness_restoration.json")
        observed.append(_cell(manifest))
        statuses[str(manifest.get("status"))] += 1
        contract_matches += int(
            manifest.get("robustness_contract_sha256") == contract_sha256
        )
        split_values.add(str(manifest.get("plan_split")))
        delivered = delivery.get("applied_events")
        registered = delivery.get("registered_events")
        delivery_passes += int(
            delivery.get("status") == "PASS"
            and (registered is None or delivered == registered)
        )
        if isinstance(delivered, (int, float)):
            applied_events += int(delivered)
        restoration_passes += int(
            restoration.get("status") == "PASS"
            and int(restoration.get("restored_initial_delay_seconds", -1)) == 1
        )
        controller = result.get("controller") or {}
        monitor = result.get("monitor") or {}
        budget_violations.append(
            _finite_number(controller.get("budget_violation_seconds"), default=math.inf)
        )
        deadline_misses += int(controller.get("deadline_misses", 0))
        monitor_failures += int(monitor.get("failures", 0))

    observed_set = set(observed)
    duplicates = len(observed) - len(observed_set)
    maximum_violation = max(budget_violations, default=math.inf)
    checks = {
        "registered_cells": len(expected),
        "completed_cells": len(run_paths),
        "missing_cells": [list(item) for item in sorted(expected - observed_set)],
        "unexpected_cells": [list(item) for item in sorted(observed_set - expected)],
        "duplicate_cells": duplicates,
        "contract_matches": contract_matches,
        "manifest_statuses": dict(statuses),
        "plan_splits": sorted(split_values),
        "delivery_passes": delivery_passes,
        "restoration_passes": restoration_passes,
        "total_applied_events": applied_events,
        "max_budget_violation_seconds": maximum_violation,
        "total_controller_deadline_misses": deadline_misses,
        "total_monitor_failures": monitor_failures,
    }
    checks["passed"] = bool(
        len(run_paths) == len(expected)
        and not checks["missing_cells"]
        and not checks["unexpected_cells"]
        and duplicates == 0
        and contract_matches == len(expected)
        and statuses == {"completed": len(expected)}
        and split_values == {"test"}
        and delivery_passes == len(expected)
        and restoration_passes == len(expected)
        and maximum_violation <= 1.0e-9
        and deadline_misses == 0
        and monitor_failures == 0
    )
    return checks


def pair_with_historical(
    perturbed: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    """Pair perturbations to historical controls without implying randomization."""

    keys = ["profile", "seed"]
    required = set(keys + ["plan_sha256", *metrics])
    for name, frame in (("perturbed", perturbed), ("historical", historical)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} table lacks required columns: {sorted(missing)}")
    joined = perturbed.merge(
        historical,
        on=keys,
        suffixes=("_perturbed", "_historical"),
        how="left",
        validate="many_to_one",
    )
    if joined["plan_sha256_historical"].isna().any():
        raise ValueError("missing historical profile/seed pair")
    if not (
        joined["plan_sha256_perturbed"] == joined["plan_sha256_historical"]
    ).all():
        raise ValueError("historical and perturbation plan hash mismatch")
    result = joined[[*keys, "condition"]].copy()
    result["plan_sha256"] = joined["plan_sha256_perturbed"]
    result["delta_definition"] = "perturbed_minus_historical"
    for metric in metrics:
        result[f"{metric}_perturbed"] = joined[f"{metric}_perturbed"]
        result[f"{metric}_historical"] = joined[f"{metric}_historical"]
        result[f"{metric}_delta"] = (
            joined[f"{metric}_perturbed"] - joined[f"{metric}_historical"]
        )
    return result
