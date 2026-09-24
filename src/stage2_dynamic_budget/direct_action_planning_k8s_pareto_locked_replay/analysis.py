"""Fail-closed integrity and frozen inference for the A3 locked replay.

The analysis code was added only after all 40 locked Kubernetes cells had
finished.  It reads the frozen artifacts, audits them before exposing endpoint
effects, and applies the decision rule recorded in LOCKED_REPLAY_PROTOCOL.md.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from stage2_dynamic_budget.direct_action_planning_k8s_cost_calibration.analysis_a2 import (
    action_horizon_valid,
    bootstrap_ci,
    count_nonempty_lines,
    exact_sign_flip_p,
    holm_adjust,
    paired_effect,
)
from stage2_dynamic_budget.direct_action_planning_k8s_cost_calibration.runtime import (
    FROZEN_CHECKPOINT_SHA256,
)
from stage2_dynamic_budget.direct_action_planning_k8s_pareto_calibration.runner import (
    audit_candidate_action_log,
)
from stage2_dynamic_budget.direct_action_planning_k8s_service_repair.prototype_api import (
    PROJECT_ROOT,
)
from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json

from .protocol import CANDIDATES, activity_quantile_for, matrix_cells
from .runtime_audit import resolve, verify_runtime_contract

from analysis.aggregate import aggregate_run  # type: ignore  # noqa: E402


SCHEMA = "dap.k8s.pareto_locked_replay_analysis.v1"
PRIMARY_METRICS = (
    "completion_difference",
    "slo_violation_difference",
    "ready_cost_difference_seconds",
)
GROUP_METRICS = (
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


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _number(row: Mapping[str, Any], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name} in {row.get('run_id')}")
    return value


def _optional_number(row: Mapping[str, Any], name: str) -> float | None:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(
    rows: list[dict[str, Any]], *, seeds: Iterable[int]
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    """Return candidate/Threshold pairs after checking plan identity."""

    lookup = {(str(row["method"]), int(row["seed"])): row for row in rows}
    pairs: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for raw_seed in seeds:
        seed = int(raw_seed)
        candidate = lookup.get(("dap_cont_0p05", seed))
        threshold = lookup.get(("threshold", seed))
        if candidate is None or threshold is None:
            raise ValueError(f"missing A3 locked pair for seed {seed}")
        if candidate["plan_sha256"] != threshold["plan_sha256"]:
            raise ValueError(f"unpaired A3 locked request plan for seed {seed}")
        pairs.append((seed, candidate, threshold))
    return pairs


def decide_locked_replay(
    statistics: Mapping[str, Mapping[str, Any]],
    *,
    analysis_config: Mapping[str, Any],
    integrity_pass: bool,
) -> dict[str, Any]:
    """Apply the frozen three-guard-plus-one-strict-CI rule exactly."""

    completion = statistics["completion_difference"]
    slo = statistics["slo_violation_difference"]
    cost = statistics["ready_cost_difference_seconds"]
    guards = {
        "completion_lcl_ge_minus_margin": float(completion["ci_low"])
        >= -float(analysis_config["completion_loss_margin"]),
        "slo_ucl_le_margin": float(slo["ci_high"])
        <= float(analysis_config["slo_increase_margin"]),
        "ready_cost_ucl_le_margin": float(cost["ci_high"])
        <= float(analysis_config["ready_cost_increase_seconds_margin"]),
        "integrity_and_hard_budget_pass": bool(integrity_pass),
    }
    strict = {
        "completion_lcl_gt_zero": float(completion["ci_low"]) > 0.0,
        "slo_ucl_lt_zero": float(slo["ci_high"]) < 0.0,
        "ready_cost_ucl_lt_zero": float(cost["ci_high"]) < 0.0,
    }
    return {
        "primary_guard_checks": guards,
        "strict_favorable_checks": strict,
        "all_primary_guards_pass": all(guards.values()),
        "at_least_one_strict_favorable_ci": any(strict.values()),
        "locked_replay_success": all(guards.values()) and any(strict.values()),
    }


def _metric_row(
    metric: str, values: list[float], *, seed: int, draws: int
) -> dict[str, Any]:
    low, high = bootstrap_ci(values, seed=seed, draws=draws)
    array = np.asarray(values, dtype=np.float64)
    return {
        "metric": metric,
        "estimand": "mean paired difference: dap_cont_0p05 minus Threshold",
        "n_pairs": len(values),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "ci_low": low,
        "ci_high": high,
        "effect_dz": paired_effect(values),
        "positive": int(np.sum(array > 0)),
        "ties": int(np.sum(np.isclose(array, 0.0, atol=1.0e-12))),
        "negative": int(np.sum(array < 0)),
        "exact_sign_flip_p": exact_sign_flip_p(values),
    }


def paired_analysis(
    rows: list[dict[str, Any]], *, config: Mapping[str, Any], integrity_pass: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pairs = paired_rows(rows, seeds=config["seeds"])
    values = {
        "completion_difference": [
            _number(candidate, "completion_rate")
            - _number(threshold, "completion_rate")
            for _, candidate, threshold in pairs
        ],
        "slo_violation_difference": [
            _number(candidate, "slo_violation_rate")
            - _number(threshold, "slo_violation_rate")
            for _, candidate, threshold in pairs
        ],
        "ready_cost_difference_seconds": [
            _number(candidate, "ready_replica_seconds")
            - _number(threshold, "ready_replica_seconds")
            for _, candidate, threshold in pairs
        ],
    }
    draws = int(config["analysis"]["bootstrap_draws"])
    bootstrap_seed = int(config["analysis"]["bootstrap_seed"])
    statistics = [
        _metric_row(metric, metric_values, seed=bootstrap_seed + index, draws=draws)
        for index, (metric, metric_values) in enumerate(values.items())
    ]
    adjusted = holm_adjust(
        [float(row["exact_sign_flip_p"]) for row in statistics]
    )
    for row, adjusted_p in zip(statistics, adjusted, strict=True):
        row["holm_p_primary_family"] = adjusted_p
    by_metric = {str(row["metric"]): row for row in statistics}

    per_seed: list[dict[str, Any]] = []
    for index, (seed, candidate, threshold) in enumerate(pairs):
        per_seed.append(
            {
                "seed": seed,
                "activity_quantile": activity_quantile_for(dict(config), seed),
                **{metric: metric_values[index] for metric, metric_values in values.items()},
                "candidate_completion": _number(candidate, "completion_rate"),
                "threshold_completion": _number(threshold, "completion_rate"),
                "candidate_slo": _number(candidate, "slo_violation_rate"),
                "threshold_slo": _number(threshold, "slo_violation_rate"),
                "candidate_ready_cost": _number(candidate, "ready_replica_seconds"),
                "threshold_ready_cost": _number(threshold, "ready_replica_seconds"),
                "plan_sha256": candidate["plan_sha256"],
            }
        )
    decision = decide_locked_replay(
        by_metric,
        analysis_config=config["analysis"],
        integrity_pass=integrity_pass,
    )
    decision.update(
        {
            "analysis_unit": "one complete paired 32-cycle request-plan replay",
            "n_pairs": len(pairs),
            "primary_family": list(PRIMARY_METRICS),
            "family_correction": "Holm over all three registered endpoints",
            "screen_and_validation_pairs_pooled": False,
        }
    )
    return statistics, per_seed, decision


def integrity_audit(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    config_path: Path,
    run_root: Path,
    runtime_contract_path: Path,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    counters: Counter[str] = Counter()
    expected = Counter(
        (str(method), int(seed))
        for _, method, seed in matrix_cells(config)
    )
    observed: Counter[tuple[str, int]] = Counter()
    plan_hashes: dict[int, set[str]] = defaultdict(set)
    metric_by_id = {str(row["run_id"]): row for row in rows}
    contract_digest = sha256_file(runtime_contract_path)
    run_directories = sorted(path.parent for path in run_root.glob("*/run_manifest.json"))

    def fail(run_id: str, check: str, detail: str) -> None:
        issues.append({"run_id": run_id, "check": check, "detail": detail})

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
    for run in run_directories:
        run_id = run.name
        missing = [name for name in required if not (run / name).is_file()]
        if missing:
            fail(run_id, "required_artifacts", ",".join(missing))
            continue
        try:
            manifest = _json(run / "run_manifest.json")
            result = _json(run / "result.json")
            delivery = _json(run / "delivery.json")
            restoration = _json(run / "readiness_restoration.json")
            plan = _json(run / "request_plan.manifest.json")
            controller = _json(run / "controller/controller_result.json")
            actions = _jsonl(run / "controller/controller_actions.jsonl")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            fail(run_id, "artifact_parse", repr(error))
            continue

        method = str(manifest.get("method"))
        seed = int(manifest.get("seed", -1))
        observed[(method, seed)] += 1
        counters["parsed_runs"] += 1
        if (run / "locked_replay_a3_failure.json").exists():
            fail(run_id, "failure_artifact", "completed run contains failure artifact")
        metric = metric_by_id.get(run_id)
        if metric is None:
            fail(run_id, "aggregate_pairing", "run absent from aggregate")
            continue
        expected_id = f"locked_replay_a3__gentd_inference__seed{seed}__{method}"
        if (
            run_id != expected_id
            or manifest.get("run_id") != run_id
            or manifest.get("matrix_row_id") != run_id
            or manifest.get("profile") != "gentd_inference"
            or (method, seed) not in expected
        ):
            fail(run_id, "run_identity", "directory or frozen identity differs")
        if manifest.get("status") != "completed" or result.get("status") != "completed":
            fail(run_id, "completion", "manifest or result not completed")
        if manifest.get("runtime_audit_sha256") != contract_digest:
            fail(run_id, "runtime_contract", "contract hash differs")
        if manifest.get("config_sha256") != sha256_file(config_path):
            fail(run_id, "config", "config hash differs")
        if manifest.get("base_checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256:
            fail(run_id, "checkpoint", "base checkpoint differs")
        if (
            manifest.get("plan_sha256") != plan.get("plan_sha256")
            or plan.get("split") != "test"
            or int(plan.get("seed", -1)) != seed
            or not np.isclose(
                float(manifest.get("activity_quantile", -1)),
                activity_quantile_for(config, seed),
            )
        ):
            fail(run_id, "plan_registration", "plan hash/split/seed/quantile differs")
        plan_hashes[seed].add(str(manifest.get("plan_sha256")))

        replay = result.get("replay", {})
        scheduled = int(replay.get("scheduled", -1))
        sent = int(replay.get("sent", -2))
        completed = int(replay.get("completed", -3))
        failed = int(replay.get("failed", -4))
        request_lines = count_nonempty_lines(run / "requests.jsonl")
        if not (
            scheduled
            == sent
            == request_lines
            == int(plan.get("request_count", -5))
            and completed + failed == scheduled
        ):
            fail(
                run_id,
                "request_accounting",
                f"scheduled={scheduled},sent={sent},lines={request_lines},completed={completed},failed={failed}",
            )
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
        monitor = result.get("monitor", {})
        missing_rate = float(metric.get("metric_missing_rate", math.nan))
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
        if (
            int(cleanup.get("final_desired_replicas", -1)) != 1
            or int(cleanup.get("final_ready_replicas", -1)) != 1
        ):
            fail(run_id, "cleanup", json.dumps(cleanup, sort_keys=True))
        else:
            counters["cleanup_pass"] += 1
        if not action_horizon_valid(actions, controller):
            fail(run_id, "action_horizon", f"actions={len(actions)},steps={controller.get('steps')}")
        else:
            counters["action_horizon_pass"] += 1
        if method in CANDIDATES:
            try:
                evidence = audit_candidate_action_log(
                    run,
                    candidate_id=method,
                    continuation_weight=float(CANDIDATES[method]),
                )
                counters["candidate_action_steps"] += int(evidence["steps"])
                counters["candidate_q_argmax_verified"] += int(evidence["q_argmax_verified"])
            except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as error:
                fail(run_id, "candidate_action_audit", repr(error))

    raw_ids = {path.name for path in run_directories}
    expected_count = sum(expected.values())
    if len(run_directories) != expected_count:
        fail("LOCKED_MATRIX", "run_directory_count", str(len(run_directories)))
    if set(metric_by_id) != raw_ids:
        fail("LOCKED_MATRIX", "aggregate_run_ids", "raw and aggregate run IDs differ")
    if observed != expected:
        fail("LOCKED_MATRIX", "registered_cells", f"observed={observed},expected={expected}")
    inconsistent = {
        str(seed): sorted(hashes)
        for seed, hashes in plan_hashes.items()
        if len(hashes) != 1
    }
    if inconsistent or set(plan_hashes) != {int(seed) for seed in config["seeds"]}:
        fail("LOCKED_MATRIX", "paired_plan_hashes", json.dumps(inconsistent, sort_keys=True))
    execution_path = run_root / "execution_summary.json"
    execution = _json(execution_path) if execution_path.is_file() else {}
    if (
        int(execution.get("registered_cells", -1)) != expected_count
        or int(execution.get("completed_this_invocation", -1)) != expected_count
        or int(execution.get("failed", -1)) != 0
    ):
        fail("LOCKED_MATRIX", "execution_summary", json.dumps(execution, sort_keys=True))
    return {
        "schema": "dap.k8s.pareto_locked_replay_integrity.v1",
        "status": "PASS" if not issues else "FAIL",
        "expected_runs": expected_count,
        "observed_run_directories": len(run_directories),
        "observed_aggregate_rows": len(rows),
        "paired_plan_groups": len(plan_hashes),
        "counters": dict(sorted(counters.items())),
        "contract_sha256": contract_digest,
        "execution_summary": execution,
        "issues": issues,
    }


def group_summaries(
    rows: list[dict[str, Any]], *, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seed = int(config["analysis"]["bootstrap_seed"])
    draws = int(config["analysis"]["bootstrap_draws"])
    for method_index, method in enumerate(config["methods"]):
        group = [row for row in rows if str(row["method"]) == str(method)]
        for metric_index, metric in enumerate(GROUP_METRICS):
            values = [
                value
                for row in group
                if (value := _optional_number(row, metric)) is not None
            ]
            if not values:
                continue
            low, high = bootstrap_ci(
                values,
                seed=seed + 100 + method_index * len(GROUP_METRICS) + metric_index,
                draws=draws,
            )
            array = np.asarray(values, dtype=np.float64)
            output.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(array.mean()),
                    "std": float(array.std(ddof=1)),
                    "ci_low": low,
                    "ci_high": high,
                    "min": float(array.min()),
                    "max": float(array.max()),
                }
            )
    return output


def pareto_counts(per_seed: list[dict[str, Any]]) -> dict[str, int]:
    labels = Counter()
    for row in per_seed:
        candidate_weak = (
            float(row["completion_difference"]) >= 0.0
            and float(row["slo_violation_difference"]) <= 0.0
            and float(row["ready_cost_difference_seconds"]) <= 0.0
        )
        threshold_weak = (
            float(row["completion_difference"]) <= 0.0
            and float(row["slo_violation_difference"]) >= 0.0
            and float(row["ready_cost_difference_seconds"]) >= 0.0
        )
        candidate_strict = candidate_weak and any(
            (
                float(row["completion_difference"]) > 0.0,
                float(row["slo_violation_difference"]) < 0.0,
                float(row["ready_cost_difference_seconds"]) < 0.0,
            )
        )
        threshold_strict = threshold_weak and any(
            (
                float(row["completion_difference"]) < 0.0,
                float(row["slo_violation_difference"]) > 0.0,
                float(row["ready_cost_difference_seconds"]) > 0.0,
            )
        )
        label = (
            "candidate_dominates"
            if candidate_strict
            else "threshold_dominates"
            if threshold_strict
            else "tradeoff"
        )
        labels[label] += 1
    return {
        "n_pairs": len(per_seed),
        "candidate_dominates": labels["candidate_dominates"],
        "threshold_dominates": labels["threshold_dominates"],
        "tradeoff": labels["tradeoff"],
    }


def _claim_table(
    statistics: list[dict[str, Any]], decision: Mapping[str, Any]
) -> str:
    lines = [
        "# A3 Locked-Replay Claim–Evidence Table",
        "",
        "All effects are frozen DAP continuation 0.05 minus Threshold over 20 fresh paired chronological-test request plans. Intervals are 20,000-draw paired percentile-bootstrap 95% intervals. Exact sign-flip tests are two-sided and Holm-adjusted as one family across all three registered endpoints.",
        "",
        "| Endpoint | Mean | SD | 95% CI | d_z | exact p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in statistics:
        lines.append(
            f"| {row['metric']} | {row['mean']:.8f} | {row['std']:.8f} | "
            f"[{row['ci_low']:.8f}, {row['ci_high']:.8f}] | "
            f"{row['effect_dz']:.6f} | {row['exact_sign_flip_p']:.8g} | "
            f"{row['holm_p_primary_family']:.8g} |"
        )
    lines.extend(
        [
            "",
            "| Frozen decision component | Result |",
            "|---|---|",
            f"| All three noninferiority guards plus integrity | {'PASS' if decision['all_primary_guards_pass'] else 'FAIL'} |",
            f"| At least one strict favorable interval | {'PASS' if decision['at_least_one_strict_favorable_ci'] else 'FAIL'} |",
            f"| Overall locked-replay rule | {'PASS' if decision['locked_replay_success'] else 'FAIL'} |",
            "",
            "Evidence remains local-testbed, one-profile, one-budget, and authorized exploratory evidence. Screen and validation pairs are not pooled with this test replay.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_analysis(
    *,
    project_root: Path,
    config_path: Path,
    runtime_contract_path: Path,
    output: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    runtime_contract_path = runtime_contract_path.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"A3 locked analysis is append-only: {output}")
    output.mkdir(parents=True, exist_ok=True)
    verify_runtime_contract(
        project_root=project_root,
        contract_path=runtime_contract_path,
        expected_config_path=config_path,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_root = resolve(config_path, config["paths"]["run_root"])
    rows = [
        aggregate_run(path.parent)
        for path in sorted(run_root.glob("*/run_manifest.json"))
    ]
    integrity = integrity_audit(
        rows,
        config=config,
        config_path=config_path,
        run_root=run_root,
        runtime_contract_path=runtime_contract_path,
    )
    write_json(output / "integrity_report.json", integrity)
    if integrity["status"] != "PASS":
        raise ValueError(
            f"A3 locked integrity failed with {len(integrity['issues'])} issues"
        )

    _write_csv(output / "run_metrics.csv", rows)
    statistics, per_seed, decision = paired_analysis(
        rows, config=config, integrity_pass=True
    )
    _write_csv(output / "paired_statistics.csv", statistics)
    _write_csv(output / "paired_seed_deltas.csv", per_seed)
    _write_csv(output / "group_summary.csv", group_summaries(rows, config=config))
    write_json(output / "pareto_pair_counts.json", pareto_counts(per_seed))
    formal_decision = {
        "schema": SCHEMA,
        "evidence_label": "authorized_local_exploratory_locked_test_replay",
        "candidate": "dap_cont_0p05",
        "comparator": "threshold",
        "bootstrap_draws": int(config["analysis"]["bootstrap_draws"]),
        "bootstrap_seed": int(config["analysis"]["bootstrap_seed"]),
        "decision": decision,
        "claim_boundary": "GenTD inference profile, one Ready budget, one local Kubernetes testbed",
    }
    write_json(output / "formal_decision.json", formal_decision)
    (output / "claim_evidence_table.md").write_text(
        _claim_table(statistics, decision), encoding="utf-8"
    )
    audit = {
        "schema": "dap.k8s.pareto_locked_replay_analysis_audit.v1",
        "status": "PASS",
        "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(config_path),
        "runtime_contract_sha256": sha256_file(runtime_contract_path),
        "integrity_report_sha256": sha256_file(output / "integrity_report.json"),
        "run_metrics_sha256": sha256_file(output / "run_metrics.csv"),
        "paired_statistics_sha256": sha256_file(output / "paired_statistics.csv"),
        "paired_seed_deltas_sha256": sha256_file(output / "paired_seed_deltas.csv"),
        "formal_decision_sha256": sha256_file(output / "formal_decision.json"),
        "input_rows": len(rows),
        "paired_units": len(per_seed),
        "comparison_family": "three registered primary endpoints; Holm adjusted",
        "no_exclusions": True,
        "screen_validation_not_pooled": True,
    }
    write_json(output / "analysis_audit.json", audit)
    return formal_decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision = run_analysis(
        project_root=args.project_root,
        config_path=args.config,
        runtime_contract_path=args.runtime_contract,
        output=args.output,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
