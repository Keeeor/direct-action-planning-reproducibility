from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

from dap.direct_action_planning_k8s_service_repair.prototype_api import (
    PROTOTYPE_ROOT,
)
from dap.utils.artifacts import sha256_file, write_json

from .audit import verify_contract
from .protocol import METHODS, PROFILES, SEEDS


if str(PROTOTYPE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT))
from analysis.aggregate import aggregate_run  # noqa: E402


METRICS = (
    ("completion_rate", "higher"),
    ("slo_violation_rate", "lower"),
    ("ready_replica_seconds", "lower"),
)
BASELINES = ("hpa", "keda")


def bootstrap_ci(
    values: Iterable[float], *, seed: int, draws: int = 20_000
) -> tuple[float, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("bootstrap requires finite paired differences")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(draws), len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def exact_sign_flip_p(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not 1 <= len(array) <= 20 or not np.isfinite(array).all():
        raise ValueError("sign-flip requires 1--20 finite paired differences")
    observed = abs(float(array.mean()))
    extreme = total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(array)):
        value = abs(float(np.mean(array * np.asarray(signs))))
        extreme += value >= observed - 1.0e-15
        total += 1
    return extreme / total


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min((len(order) - rank) * p_values[index], 1.0))
        adjusted[index] = running
    return adjusted


def _effect(values: np.ndarray) -> float:
    sd = float(values.std(ddof=1))
    if sd <= 1.0e-15:
        return math.copysign(math.inf, float(values.mean())) if values.mean() else 0.0
    return float(values.mean() / sd)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _current_run_dirs(run_root: Path, contract_hash: str) -> list[Path]:
    result: list[Path] = []
    for path in sorted(run_root.glob("native_v1__*/run_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") == "completed"
            and payload.get("native_comparison_contract_sha256") == contract_hash
        ):
            result.append(path.parent)
    return result


def analyze(
    project_root: Path, contract_path: Path, run_root: Path, output: Path
) -> dict[str, Any]:
    project_root = project_root.resolve()
    contract_path = contract_path.resolve()
    contract = verify_contract(project_root, contract_path)
    contract_hash = sha256_file(contract_path)
    run_dirs = _current_run_dirs(run_root.resolve(), contract_hash)
    if len(run_dirs) != 60:
        raise ValueError(f"analysis requires 60 completed current-contract runs, got {len(run_dirs)}")
    rows = [aggregate_run(path) for path in run_dirs]
    lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    issues: list[str] = []
    for row, run_dir in zip(rows, run_dirs, strict=True):
        key = (str(row["profile"]), str(row["method"]), int(row["seed"]))
        if key in lookup:
            issues.append(f"duplicate cell: {key}")
        lookup[key] = row
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        delivery = json.loads((run_dir / "delivery.json").read_text(encoding="utf-8"))
        cleanup = result.get("cleanup", {})
        actions = [
            json.loads(line)
            for line in (run_dir / "controller/controller_actions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if delivery.get("status") != "PASS":
            issues.append(f"delivery failed: {key}")
        if len(actions) != 32:
            issues.append(f"action count {len(actions)}: {key}")
        if float(row["budget_violation_seconds"]) > 1.0e-9:
            issues.append(f"ledger violation: {key}")
        if cleanup.get("final_ready_replicas") != 1:
            issues.append(f"cleanup Ready != 1: {key}")
    expected = {
        (profile, method, seed)
        for profile in PROFILES
        for method in METHODS
        for seed in SEEDS
    }
    if set(lookup) != expected:
        issues.append("cell identity mismatch")
    for profile in PROFILES:
        for seed in SEEDS:
            hashes = {
                str(lookup[(profile, method, seed)]["plan_sha256"])
                for method in METHODS
            }
            if len(hashes) != 1:
                issues.append(f"plan hash mismatch: {profile}/{seed}")
    if issues:
        raise ValueError("integrity audit failed: " + "; ".join(issues[:10]))

    run_rows: list[dict[str, Any]] = []
    for row in rows:
        run_rows.append({key: value for key, value in row.items() if key != "run_directory"})
    paired_rows: list[dict[str, Any]] = []
    statistics: list[dict[str, Any]] = []
    stat_indices: list[int] = []
    for profile_index, profile in enumerate(PROFILES):
        for baseline_index, baseline in enumerate(BASELINES):
            for metric_index, (metric, direction) in enumerate(METRICS):
                deltas: list[float] = []
                for seed in SEEDS:
                    dap = float(lookup[(profile, "dap_repaired", seed)][metric])
                    other = float(lookup[(profile, baseline, seed)][metric])
                    delta = dap - other
                    deltas.append(delta)
                    paired_rows.append(
                        {
                            "profile": profile,
                            "baseline": baseline,
                            "seed": seed,
                            "metric": metric,
                            "dap": dap,
                            "baseline_value": other,
                            "dap_minus_baseline": delta,
                        }
                    )
                array = np.asarray(deltas, dtype=np.float64)
                low, high = bootstrap_ci(
                    array,
                    seed=2026081701 + profile_index * 100 + baseline_index * 10 + metric_index,
                )
                row = {
                    "profile": profile,
                    "baseline": baseline,
                    "metric": metric,
                    "direction": direction,
                    "n_pairs": len(array),
                    "mean_dap_minus_baseline": float(array.mean()),
                    "std": float(array.std(ddof=1)),
                    "ci_low": low,
                    "ci_high": high,
                    "effect_dz": _effect(array),
                    "wins_positive": int(np.sum(array > 0)),
                    "ties": int(np.sum(np.isclose(array, 0.0, atol=1.0e-12))),
                    "wins_negative": int(np.sum(array < 0)),
                    "exact_sign_flip_p": exact_sign_flip_p(array),
                }
                statistics.append(row)
                stat_indices.append(len(statistics) - 1)
    adjusted = holm_adjust([statistics[index]["exact_sign_flip_p"] for index in stat_indices])
    for index, value in zip(stat_indices, adjusted, strict=True):
        statistics[index]["holm_p_12_endpoint_family"] = value

    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "run_metrics.csv", run_rows)
    _write_csv(output / "paired_plan_deltas.csv", paired_rows)
    _write_csv(output / "paired_statistics.csv", statistics)
    summary = {
        "schema": "dap.k8s.native_comparison_analysis.v1",
        "status": "PASS",
        "runs": len(rows),
        "pairs_per_comparison": 10,
        "bootstrap_draws": 20_000,
        "holm_family": 12,
        "observed_ledger_violations": 0,
        "statistics": statistics,
        "claim_boundary": (
            "paired request-plan effects on the tested local cluster; all baselines and endpoints reported"
        ),
    }
    write_json(output / "summary.json", summary)
    manifest = {
        "schema": "dap.k8s.native_comparison_analysis_manifest.v1",
        "status": "PASS",
        "contract_sha256": contract_hash,
        "source_sha256": sha256_file(Path(__file__)),
        "outputs": {
            name: sha256_file(output / name)
            for name in (
                "run_metrics.csv",
                "paired_plan_deltas.csv",
                "paired_statistics.csv",
                "summary.json",
            )
        },
    }
    write_json(output / "manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.project_root, args.contract, args.run_root, args.output)
    print(json.dumps({"status": summary["status"], "runs": summary["runs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
