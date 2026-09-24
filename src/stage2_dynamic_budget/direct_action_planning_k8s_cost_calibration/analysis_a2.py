"""Locked integrity and paired analysis for the A2 Kubernetes replay."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from stage2_dynamic_budget.utils.artifacts import write_json

from .runner import audit_calibrated_action_log


BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026081101
EXPECTED_SEEDS = tuple(range(2026081211, 2026081231))
ACTIVITY_QUANTILES = dict(
    zip(
        EXPECTED_SEEDS,
        tuple(value for value in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00) for _ in range(2)),
        strict=True,
    )
)
METHODS = ("dap_calibrated", "threshold")
METRICS = (
    "completion_rate",
    "slo_violation_rate",
    "ready_replica_seconds",
    "mean_latency_seconds",
    "p95_latency_seconds",
    "p99_latency_seconds",
    "failure_rate",
    "timeout_rate",
    "scaling_count",
    "controller_loop_seconds",
    "controller_inference_seconds",
)


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def number(row: dict[str, Any], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name} in {row.get('run_id')}")
    return value


def optional_number(row: dict[str, Any], name: str) -> float | None:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def bootstrap_ci(
    values: Iterable[float], *, seed: int, draws: int = BOOTSTRAP_DRAWS
) -> tuple[float, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if (
        array.ndim != 1
        or len(array) == 0
        or not np.isfinite(array).all()
        or int(draws) < 1
    ):
        raise ValueError("bootstrap requires finite nonempty one-dimensional values")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(draws), len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def relative_cost_bootstrap(
    left_cost: Iterable[float],
    right_cost: Iterable[float],
    *,
    seed: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[float, float, float, int]:
    """Paired ratio of means, defined even if individual right costs are zero."""
    left = np.asarray(tuple(left_cost), dtype=np.float64)
    right = np.asarray(tuple(right_cost), dtype=np.float64)
    if (
        left.shape != right.shape
        or left.ndim != 1
        or len(left) == 0
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("relative cost requires aligned finite nonempty pairs")
    denominator = float(right.mean())
    if denominator <= 1.0e-12:
        raise ValueError("mean comparator cost is zero; relative estimand undefined")
    point = float((left.mean() - denominator) / denominator)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(left), size=(int(draws), len(left)))
    left_means = left[indices].mean(axis=1)
    right_means = right[indices].mean(axis=1)
    valid = right_means > 1.0e-12
    values = (left_means[valid] - right_means[valid]) / right_means[valid]
    if len(values) < int(draws) * 0.99:
        raise ValueError("too many undefined bootstrap relative-cost replicates")
    return (
        point,
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        int((~valid).sum()),
    )


def exact_sign_flip_p(values: Iterable[float]) -> float:
    """Exact two-sided paired randomization p-value for at most 20 pairs."""
    array = np.asarray(tuple(values), dtype=np.float64)
    if (
        array.ndim != 1
        or len(array) == 0
        or len(array) > 20
        or not np.isfinite(array).all()
    ):
        raise ValueError("exact sign-flip supports 1--20 finite pairs")
    observed = abs(float(array.mean()))
    total = 1 << len(array)
    extreme = 0
    bit_positions = np.arange(len(array), dtype=np.uint64)
    for start in range(0, total, 65_536):
        stop = min(start + 65_536, total)
        ids = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = 1.0 - 2.0 * ((ids >> bit_positions) & 1).astype(np.float64)
        statistics = np.abs(signs @ array / len(array))
        extreme += int(np.sum(statistics >= observed - 1.0e-15))
    return extreme / total


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(
            running,
            min((len(p_values) - rank) * float(p_values[index]), 1.0),
        )
        adjusted[index] = running
    return adjusted


def paired_effect(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("paired effect requires at least two finite values")
    standard_deviation = float(array.std(ddof=1))
    if standard_deviation <= 1.0e-15:
        mean = float(array.mean())
        return math.copysign(math.inf, mean) if mean else 0.0
    return float(array.mean() / standard_deviation)


def action_horizon_valid(
    actions: list[dict[str, Any]], controller: dict[str, Any]
) -> bool:
    """Require 32 logged actions and cross-check controller steps when present."""
    recorded_steps = controller.get("steps")
    return len(actions) == 32 and (
        recorded_steps is None or int(recorded_steps) == 32
    )


def finite_mean_or_none(values: Iterable[float]) -> float | None:
    array = np.asarray(tuple(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else None


def count_nonempty_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(
    rows: list[dict[str, Any]], *, seeds: Iterable[int] = EXPECTED_SEEDS
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    lookup = {(str(row["method"]), int(row["seed"])): row for row in rows}
    pairs: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for seed in seeds:
        left = lookup.get(("dap_calibrated", int(seed)))
        right = lookup.get(("threshold", int(seed)))
        if left is None or right is None:
            raise ValueError(f"missing A2 pair for seed {seed}")
        if left["plan_sha256"] != right["plan_sha256"]:
            raise ValueError(f"unpaired request-plan hash for seed {seed}")
        pairs.append((int(seed), left, right))
    return pairs


def _metric_row(metric: str, values: list[float], *, seed: int) -> dict[str, Any]:
    low, high = bootstrap_ci(values, seed=seed)
    array = np.asarray(values, dtype=np.float64)
    return {
        "metric": metric,
        "estimand": "mean paired difference: calibrated DAP minus Threshold",
        "n_pairs": len(values),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "ci_low": low,
        "ci_high": high,
        "effect_dz": paired_effect(values),
        "positive": int(np.sum(array > 0)),
        "ties": int(np.sum(np.isclose(array, 0.0, atol=1.0e-12))),
        "negative": int(np.sum(array < 0)),
    }


def decide_claims(
    statistics: dict[str, dict[str, Any]], *, integrity_pass: bool
) -> dict[str, Any]:
    favorable_service = bool(
        statistics["completion_gain"]["ci_low"] > 0.0
        or statistics["slo_increase"]["ci_high"] < 0.0
    )
    bounded_checks = {
        "favorable_service_ci": favorable_service,
        "completion_loss_ucl_le_0.01": statistics["completion_loss"]["ci_high"] <= 0.01,
        "ready_cost_difference_ucl_le_10_seconds": statistics["ready_cost_difference"]["ci_high"] <= 10.0,
        "integrity_and_hard_budget_pass": bool(integrity_pass),
    }
    strict_checks = {
        "favorable_service_ci": favorable_service,
        "ready_cost_difference_ucl_le_0": statistics["ready_cost_difference"]["ci_high"] <= 0.0,
        "integrity_and_hard_budget_pass": bool(integrity_pass),
    }
    original_checks = {
        "favorable_service_ci": favorable_service,
        "relative_cost_increase_ucl_le_0.05": statistics["relative_cost_increase"]["ci_high"] <= 0.05,
        "integrity_and_hard_budget_pass": bool(integrity_pass),
    }
    noninferiority_checks = {
        "completion_loss_ucl_le_0.01": statistics["completion_loss"]["ci_high"] <= 0.01,
        "slo_increase_ucl_le_0.02": statistics["slo_increase"]["ci_high"] <= 0.02,
        "integrity_and_hard_budget_pass": bool(integrity_pass),
    }
    return {
        "bounded_service_gain_checks": bounded_checks,
        "bounded_service_gain_pass": all(bounded_checks.values()),
        "strict_pareto_checks": strict_checks,
        "strict_pareto_pass": all(strict_checks.values()),
        "original_service_gain_checks": original_checks,
        "original_service_gain_pass": all(original_checks.values()),
        "service_noninferiority_checks": noninferiority_checks,
        "service_noninferiority_pass": all(noninferiority_checks.values()),
    }


def service_cost_analysis(
    rows: list[dict[str, Any]], *, integrity_pass: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pairs = paired_rows(rows)
    left_cost = [number(left, "ready_replica_seconds") for _, left, _ in pairs]
    right_cost = [number(right, "ready_replica_seconds") for _, _, right in pairs]
    values = {
        "completion_gain": [
            number(left, "completion_rate") - number(right, "completion_rate")
            for _, left, right in pairs
        ],
        "completion_loss": [
            number(right, "completion_rate") - number(left, "completion_rate")
            for _, left, right in pairs
        ],
        "slo_increase": [
            number(left, "slo_violation_rate") - number(right, "slo_violation_rate")
            for _, left, right in pairs
        ],
        "ready_cost_difference": [
            number(left, "ready_replica_seconds") - number(right, "ready_replica_seconds")
            for _, left, right in pairs
        ],
    }
    per_seed = []
    for pair_index, (seed, left, right) in enumerate(pairs):
        per_seed.append({
            "seed": seed,
            "activity_quantile": ACTIVITY_QUANTILES[seed],
            "activity_stratum": "low" if ACTIVITY_QUANTILES[seed] <= 0.75 else "high",
            **{name: metric_values[pair_index] for name, metric_values in values.items()},
            "dap_completion": number(left, "completion_rate"),
            "threshold_completion": number(right, "completion_rate"),
            "dap_slo": number(left, "slo_violation_rate"),
            "threshold_slo": number(right, "slo_violation_rate"),
            "dap_ready_cost": number(left, "ready_replica_seconds"),
            "threshold_ready_cost": number(right, "ready_replica_seconds"),
            "plan_sha256": left["plan_sha256"],
        })

    statistics: list[dict[str, Any]] = []
    by_metric: dict[str, dict[str, Any]] = {}
    for index, (metric, metric_values) in enumerate(values.items()):
        row = _metric_row(metric, metric_values, seed=BOOTSTRAP_SEED + index)
        if metric in {"completion_gain", "slo_increase", "ready_cost_difference"}:
            row["exact_sign_flip_p"] = exact_sign_flip_p(metric_values)
        statistics.append(row)
        by_metric[metric] = row

    point, low, high, undefined = relative_cost_bootstrap(
        left_cost, right_cost, seed=BOOTSTRAP_SEED + 4
    )
    relative = {
        "metric": "relative_cost_increase",
        "estimand": "(mean calibrated-DAP cost - mean Threshold cost) / mean Threshold cost under paired resampling",
        "n_pairs": len(pairs),
        "mean": point,
        "std": None,
        "ci_low": low,
        "ci_high": high,
        "effect_dz": None,
        "undefined_bootstrap_replicates": undefined,
    }
    statistics.append(relative)
    by_metric["relative_cost_increase"] = relative

    service_indices = [0, 2]
    adjusted = holm_adjust(
        [float(statistics[index]["exact_sign_flip_p"]) for index in service_indices]
    )
    for index, adjusted_p in zip(service_indices, adjusted, strict=True):
        statistics[index]["holm_p_service_family"] = adjusted_p

    decision = decide_claims(by_metric, integrity_pass=integrity_pass)
    decision.update({
        "analysis_unit": "one complete paired request-plan run",
        "n_pairs": len(pairs),
        "service_sign_flip_family": ["completion_gain", "slo_increase"],
        "cost_sign_flip_is_secondary_and_outside_service_family": True,
    })
    return statistics, per_seed, decision


def stratified_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for stratum_index, (stratum, seeds) in enumerate((
        ("low", EXPECTED_SEEDS[:10]),
        ("high", EXPECTED_SEEDS[10:]),
    )):
        pairs = paired_rows(rows, seeds=seeds)
        metrics = {
            "completion_gain": [number(left, "completion_rate") - number(right, "completion_rate") for _, left, right in pairs],
            "slo_increase": [number(left, "slo_violation_rate") - number(right, "slo_violation_rate") for _, left, right in pairs],
            "ready_cost_difference": [number(left, "ready_replica_seconds") - number(right, "ready_replica_seconds") for _, left, right in pairs],
        }
        for metric_index, (metric, values) in enumerate(metrics.items()):
            row = _metric_row(
                metric,
                values,
                seed=BOOTSTRAP_SEED + 100 + stratum_index * 10 + metric_index,
            )
            row.update({
                "activity_stratum": stratum,
                "quantile_range": "0.55--0.75" if stratum == "low" else "0.80--1.00",
                "analysis_role": "supplementary_not_used_for_primary_decision",
            })
            result.append(row)
    return result


def group_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for method_index, method in enumerate(METHODS):
        group = [row for row in rows if row["method"] == method]
        for metric_index, metric in enumerate(METRICS):
            values = [value for row in group if (value := optional_number(row, metric)) is not None]
            if not values:
                continue
            low, high = bootstrap_ci(
                values, seed=BOOTSTRAP_SEED + 200 + method_index * 20 + metric_index
            )
            array = np.asarray(values, dtype=np.float64)
            result.append({
                "method": method,
                "metric": metric,
                "n": len(values),
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)),
                "ci_low": low,
                "ci_high": high,
                "min": float(array.min()),
                "max": float(array.max()),
            })
    return result


def pareto_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = []
    for seed, left, right in paired_rows(rows):
        left_weak = (
            number(left, "completion_rate") >= number(right, "completion_rate")
            and number(left, "slo_violation_rate") <= number(right, "slo_violation_rate")
            and number(left, "ready_replica_seconds") <= number(right, "ready_replica_seconds")
        )
        right_weak = (
            number(right, "completion_rate") >= number(left, "completion_rate")
            and number(right, "slo_violation_rate") <= number(left, "slo_violation_rate")
            and number(right, "ready_replica_seconds") <= number(left, "ready_replica_seconds")
        )
        left_strict = left_weak and any((
            number(left, "completion_rate") > number(right, "completion_rate"),
            number(left, "slo_violation_rate") < number(right, "slo_violation_rate"),
            number(left, "ready_replica_seconds") < number(right, "ready_replica_seconds"),
        ))
        right_strict = right_weak and any((
            number(right, "completion_rate") > number(left, "completion_rate"),
            number(right, "slo_violation_rate") < number(left, "slo_violation_rate"),
            number(right, "ready_replica_seconds") < number(left, "ready_replica_seconds"),
        ))
        label = "dap_dominates" if left_strict else "threshold_dominates" if right_strict else "tradeoff"
        labels.append({"seed": seed, "label": label})
    return [{
        "n_pairs": len(labels),
        "dap_dominates": sum(row["label"] == "dap_dominates" for row in labels),
        "threshold_dominates": sum(row["label"] == "threshold_dominates" for row in labels),
        "tradeoff": sum(row["label"] == "tradeoff" for row in labels),
    }]


def mechanism_analysis(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for manifest_path in sorted(run_root.glob("formal_a2__*/run_manifest.json")):
        run = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actions = [
            json.loads(line)
            for line in (run / "controller/controller_actions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        targets = np.asarray([float(row["target_replicas"]) for row in actions])
        current = np.asarray([
            float(row.get("hard_mask_current_ready_replicas", np.nan)) for row in actions
        ])
        output = {
            "run_id": run.name,
            "method": manifest["method"],
            "seed": int(manifest["seed"]),
            "activity_quantile": float(manifest["activity_quantile"]),
            "mean_target_replicas": float(targets.mean()),
            "target_changes": int(np.sum(targets[1:] != targets[:-1])),
            "mean_current_ready_replicas": finite_mean_or_none(current),
            "mean_loop_seconds": float(np.mean([float(row["control_loop_latency_seconds"]) for row in actions])),
        }
        if manifest["method"] == "dap_calibrated":
            q_margins = []
            for action in actions:
                feasible = sorted(
                    [float(value) for name, value in action["q_values"].items() if action["feasible"].get(name)],
                    reverse=True,
                )
                q_margins.append(feasible[0] - feasible[1] if len(feasible) > 1 else np.nan)
            output.update({
                "mean_q_margin": float(np.nanmean(q_margins)),
                "mean_inference_seconds": float(np.mean([float(row["dap_inference_latency_seconds"]) for row in actions])),
            })
        rows.append(output)
    return rows


def formal_integrity_audit(
    rows: list[dict[str, Any]], run_root: Path, contract_path: Path
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    counters: Counter[str] = Counter()
    observed: Counter[tuple[str, int]] = Counter()
    plan_hashes: dict[int, set[str]] = defaultdict(set)
    metric_ids = {str(row["run_id"]) for row in rows}
    metric_by_id = {str(row["run_id"]): row for row in rows}
    contract_digest = sha256(contract_path.resolve())
    run_directories = sorted(run_root.resolve().glob("formal_a2__*"))

    def fail(run_id: str, check: str, detail: str) -> None:
        issues.append({"run_id": run_id, "check": check, "detail": detail})

    for run in run_directories:
        run_id = run.name
        required = (
            "run_manifest.json",
            "result.json",
            "delivery.json",
            "readiness_restoration.json",
            "request_plan.manifest.json",
            "requests.jsonl",
            "controller/controller_result.json",
            "controller/controller_actions.jsonl",
        )
        missing = [name for name in required if not (run / name).is_file()]
        if missing:
            fail(run_id, "required_artifacts", ",".join(missing))
            continue
        try:
            manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            delivery = json.loads((run / "delivery.json").read_text(encoding="utf-8"))
            restoration = json.loads((run / "readiness_restoration.json").read_text(encoding="utf-8"))
            plan = json.loads((run / "request_plan.manifest.json").read_text(encoding="utf-8"))
            controller = json.loads((run / "controller/controller_result.json").read_text(encoding="utf-8"))
            actions = [
                json.loads(line)
                for line in (run / "controller/controller_actions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            fail(run_id, "artifact_parse", repr(error))
            continue

        method = str(manifest.get("method"))
        seed = int(manifest.get("seed", -1))
        observed[(method, seed)] += 1
        counters["parsed_runs"] += 1
        if (run / "a2_failure.json").exists():
            fail(run_id, "failure_artifact", "completed directory contains a2_failure.json")
        if run_id not in metric_ids:
            fail(run_id, "aggregate_pairing", "run absent from aggregate")
            continue
        if (
            manifest.get("run_id") != run_id
            or manifest.get("matrix_row_id") != run_id
            or manifest.get("profile") != "gentd_inference"
            or seed not in EXPECTED_SEEDS
            or method not in METHODS
        ):
            fail(run_id, "run_identity", "directory or registered identity differs")
        if manifest.get("status") != "completed" or result.get("status") != "completed":
            fail(run_id, "completion", "manifest or result is not completed")
        if manifest.get("runtime_audit_sha256") != contract_digest:
            fail(run_id, "runtime_contract", "runtime contract digest differs")
        if manifest.get("plan_sha256") != plan.get("plan_sha256"):
            fail(run_id, "plan_manifest", "plan hash differs within run")
        if (
            plan.get("split") != "test"
            or int(plan.get("seed", -1)) != seed
            or float(manifest.get("activity_quantile", -1)) != ACTIVITY_QUANTILES.get(seed)
        ):
            fail(run_id, "plan_registration", "split, seed, or activity quantile differs")
        plan_hashes[seed].add(str(manifest.get("plan_sha256")))

        replay = result.get("replay", {})
        scheduled = int(replay.get("scheduled", -1))
        sent = int(replay.get("sent", -2))
        completed = int(replay.get("completed", -3))
        failed = int(replay.get("failed", -4))
        request_lines = count_nonempty_lines(run / "requests.jsonl")
        if not (
            scheduled == sent == request_lines == int(plan.get("request_count", -5))
            and completed + failed == scheduled
        ):
            fail(run_id, "request_accounting", f"scheduled={scheduled},sent={sent},lines={request_lines},completed={completed},failed={failed}")
        else:
            counters["request_accounting_pass"] += 1

        budget = float(manifest.get("budget_seconds", math.nan))
        ready_cost = float(controller.get("ready_cost_seconds", math.nan))
        violation = float(controller.get("budget_violation_seconds", math.nan))
        if not (
            math.isfinite(budget)
            and math.isfinite(ready_cost)
            and ready_cost <= budget + 1.0e-9
            and abs(violation) <= 1.0e-9
        ):
            fail(run_id, "hard_budget", f"budget={budget},cost={ready_cost},violation={violation}")
        else:
            counters["hard_budget_pass"] += 1
        if int(controller.get("deadline_misses", -1)) != 0:
            fail(run_id, "controller_deadlines", str(controller.get("deadline_misses")))
        else:
            counters["deadline_pass"] += 1
        missing_rate = float(metric_by_id[run_id].get("metric_missing_rate", math.nan))
        monitor = result.get("monitor", {})
        if (
            int(monitor.get("failures", -1)) != 0
            or int(monitor.get("samples", 0)) < 1
            or not math.isfinite(missing_rate)
            or missing_rate >= 0.01
        ):
            fail(run_id, "monitor", f"failures={monitor.get('failures')},missing_rate={missing_rate}")
        else:
            counters["monitor_pass"] += 1
        if delivery.get("status") != "PASS":
            fail(run_id, "delivery", json.dumps(delivery, sort_keys=True))
        else:
            counters["delivery_pass"] += 1
        if (
            restoration.get("status") != "PASS"
            or int(restoration.get("restored_initial_delay_seconds", -1)) != 1
        ):
            fail(run_id, "readiness_restoration", json.dumps(restoration, sort_keys=True))
        else:
            counters["restoration_pass"] += 1
        cleanup = result.get("cleanup", {})
        if int(cleanup.get("final_desired_replicas", -1)) != 1 or int(cleanup.get("final_ready_replicas", -1)) != 1:
            fail(run_id, "cleanup", json.dumps(cleanup, sort_keys=True))
        else:
            counters["cleanup_pass"] += 1
        if not action_horizon_valid(actions, controller):
            fail(run_id, "action_horizon", f"actions={len(actions)},steps={controller.get('steps')}")
        else:
            counters["action_horizon_pass"] += 1
        if method == "dap_calibrated":
            try:
                evidence = audit_calibrated_action_log(run)
                counters["calibrated_action_steps"] += evidence["steps"]
                counters["calibrated_q_argmax_verified"] += evidence["q_argmax_verified"]
                counters["calibrated_independent_ready_recorded"] += evidence["independent_ready_channels_recorded"]
            except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as error:
                fail(run_id, "calibrated_action_audit", repr(error))

    expected = Counter((method, seed) for seed in EXPECTED_SEEDS for method in METHODS)
    raw_ids = {path.name for path in run_directories}
    if len(run_directories) != 40:
        fail("FORMAL_MATRIX", "run_directory_count", str(len(run_directories)))
    if metric_ids != raw_ids:
        fail("FORMAL_MATRIX", "aggregate_run_ids", "raw and aggregate run-id sets differ")
    if observed != expected:
        fail("FORMAL_MATRIX", "registered_cells", f"observed={observed},expected={expected}")
    inconsistent = {str(seed): sorted(hashes) for seed, hashes in plan_hashes.items() if len(hashes) != 1}
    if inconsistent or set(plan_hashes) != set(EXPECTED_SEEDS):
        fail("FORMAL_MATRIX", "paired_plan_hashes", json.dumps({"groups": len(plan_hashes), "inconsistent": inconsistent}, sort_keys=True))
    execution_path = run_root / "execution_summary.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8")) if execution_path.is_file() else {}
    if execution.get("registered_cells") != 40 or execution.get("failed") != 0:
        fail("FORMAL_MATRIX", "execution_summary", json.dumps(execution, sort_keys=True))
    return {
        "schema": "dap.k8s.cost_calibration_formal_integrity.v1",
        "status": "PASS" if not issues else "FAIL",
        "expected_runs": 40,
        "observed_run_directories": len(run_directories),
        "observed_aggregate_rows": len(rows),
        "paired_plan_groups": len(plan_hashes),
        "counters": dict(sorted(counters.items())),
        "contract_sha256": contract_digest,
        "execution_summary": execution,
        "issues": issues,
    }


def _write_claim_table(path: Path, statistics: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    by_metric = {row["metric"]: row for row in statistics}
    lines = [
        "# A2 Claim–Evidence Table",
        "",
        "All effects are calibrated DAP minus Threshold over 20 paired request plans. Intervals are 20,000-draw paired percentile-bootstrap 95% intervals. Exact sign-flip tests are two-sided; the two service endpoints use Holm correction.",
        "",
        "| Endpoint | Mean | 95% CI | d_z | exact p | Holm service p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in ("completion_gain", "slo_increase", "ready_cost_difference", "relative_cost_increase"):
        row = by_metric[metric]
        effect = "NA" if row.get("effect_dz") is None else f"{row['effect_dz']:.6f}"
        exact = "NA" if row.get("exact_sign_flip_p") is None else f"{row['exact_sign_flip_p']:.8g}"
        holm = "NA" if row.get("holm_p_service_family") is None else f"{row['holm_p_service_family']:.8g}"
        lines.append(f"| {metric} | {row['mean']:.8f} | [{row['ci_low']:.8f}, {row['ci_high']:.8f}] | {effect} | {exact} | {holm} |")
    lines.extend([
        "",
        "| Registered decision | Result |",
        "|---|---|",
        f"| Bounded service gain | {'PASS' if decision['bounded_service_gain_pass'] else 'FAIL'} |",
        f"| Strict Pareto | {'PASS' if decision['strict_pareto_pass'] else 'FAIL'} |",
        f"| Original +5% service-gain branch | {'PASS' if decision['original_service_gain_pass'] else 'FAIL'} |",
        f"| Service noninferiority | {'PASS' if decision['service_noninferiority_pass'] else 'FAIL'} |",
        "",
        "Evidence is exploratory, GenTD-profile and local-testbed bounded. The deployed candidate has zero continuation weight, so these results do not support a learned continuation-value claim.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.metrics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 40 or any(row["run_status"] != "completed" for row in rows):
        raise ValueError("A2 analysis requires exactly 40 completed aggregate rows")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    integrity = formal_integrity_audit(rows, args.run_root.resolve(), args.contract.resolve())
    write_json(output / "integrity_report.json", integrity)
    if integrity["status"] != "PASS":
        raise ValueError(f"A2 integrity audit failed with {len(integrity['issues'])} issues")
    statistics, per_seed, decision = service_cost_analysis(rows, integrity_pass=True)
    strata = stratified_analysis(rows)
    groups = group_summaries(rows)
    pareto = pareto_analysis(rows)
    mechanisms = mechanism_analysis(args.run_root.resolve())
    write_csv(output / "paired_statistics.csv", statistics)
    write_csv(output / "paired_seed_deltas.csv", per_seed)
    write_csv(output / "stratified_statistics.csv", strata)
    write_csv(output / "group_summary.csv", groups)
    write_csv(output / "pareto_pair_counts.csv", pareto)
    write_csv(output / "mechanism_seed_metrics.csv", mechanisms)
    formal_decision = {
        "schema": "dap.k8s.cost_calibration_formal_decision.v1",
        "evidence_label": "authorized_exploratory_locked_replay",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "decision": decision,
        "claim_boundary": "frozen immediate DAP on registered GenTD profile and local Kubernetes testbed",
        "learned_continuation_value_claim_supported": False,
    }
    write_json(output / "formal_decision.json", formal_decision)
    _write_claim_table(output / "claim_evidence_table.md", statistics, decision)
    audit = {
        "schema": "dap.k8s.cost_calibration_analysis_audit.v1",
        "status": "PASS",
        "analysis_source_sha256": sha256(Path(__file__).resolve()),
        "metrics_sha256": sha256(args.metrics.resolve()),
        "runtime_contract_sha256": sha256(args.contract.resolve()),
        "integrity_report_sha256": sha256(output / "integrity_report.json"),
        "input_rows": len(rows),
        "paired_units": len(per_seed),
        "comparison_family": "two registered service endpoints; Holm adjusted",
        "no_exclusions": True,
        "supplementary_strata_not_used_for_primary_decision": True,
    }
    write_json(output / "analysis_audit.json", audit)
    print(json.dumps({"status": "PASS", "decision": formal_decision}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
