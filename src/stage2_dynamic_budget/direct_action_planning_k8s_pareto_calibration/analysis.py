"""Fail-closed integrity and descriptive ranking for the frozen A3 screen."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from stage2_dynamic_budget.direct_action_planning_k8s_service_repair.prototype_api import (
    PROJECT_ROOT,
)
from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json

from .protocol import activity_quantile_for, validate_runtime_config
from .runner import audit_candidate_action_log
from .runtime_audit import resolve, verify_runtime_contract
from .selection import rank_screen_candidates

# prototype_api installs the frozen prototype root on sys.path.
from analysis.aggregate import aggregate_run  # type: ignore  # noqa: E402


SCHEMA = "dap.k8s.pareto_calibration_screen_analysis.v1"


def action_horizon_valid(
    actions: list[dict[str, Any]], controller: Mapping[str, Any]
) -> bool:
    """Require 32 logs and cross-check the optional DAP-only summary field."""

    recorded_steps = controller.get("steps")
    return len(actions) == 32 and (
        recorded_steps is None or int(recorded_steps) == 32
    )


def count_nonempty_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _finite(row: Mapping[str, Any], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {name} in {row.get('run_id', row.get('method'))}")
    return value


def paired_candidate_rows(
    rows: Iterable[Mapping[str, Any]], *, candidate_methods: Iterable[str]
) -> list[dict[str, Any]]:
    """Build registered candidate-minus-Threshold contrasts by request plan."""

    materialized = [dict(row) for row in rows]
    lookup = {
        (str(row["method"]), int(row["seed"])): row for row in materialized
    }
    seeds = sorted(
        int(row["seed"])
        for row in materialized
        if str(row["method"]) == "threshold"
    )
    output: list[dict[str, Any]] = []
    for method in candidate_methods:
        for seed in seeds:
            candidate = lookup.get((str(method), seed))
            threshold = lookup.get(("threshold", seed))
            if candidate is None or threshold is None:
                raise ValueError(f"missing paired A3 row: method={method}, seed={seed}")
            completion_gain = _finite(candidate, "completion_rate") - _finite(
                threshold, "completion_rate"
            )
            output.append(
                {
                    "method": str(method),
                    "seed": seed,
                    "completion_gain": completion_gain,
                    "completion_loss": -completion_gain,
                    "slo_increase": _finite(candidate, "slo_violation_rate")
                    - _finite(threshold, "slo_violation_rate"),
                    "ready_cost_increase_seconds": _finite(
                        candidate, "ready_replica_seconds"
                    )
                    - _finite(threshold, "ready_replica_seconds"),
                    "budget_violation_seconds": _finite(
                        candidate, "budget_violation_seconds"
                    ),
                }
            )
    return output


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def summarize_screen_candidates(
    paired: Iterable[Mapping[str, Any]],
    *,
    continuation_candidates: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Compute frozen stage-one point summaries; no inferential claim at n=4."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in paired:
        groups[str(row["method"])].append(row)
    if set(groups) != set(continuation_candidates):
        raise ValueError("A3 candidate summary does not match frozen candidate grid")
    output: list[dict[str, Any]] = []
    for method in continuation_candidates:
        rows = groups[method]
        record: dict[str, Any] = {
            "method": method,
            "continuation_weight": float(continuation_candidates[method]),
            "n_pairs": len(rows),
            "budget_violation_seconds": max(
                _finite(row, "budget_violation_seconds") for row in rows
            ),
        }
        for metric in (
            "completion_gain",
            "completion_loss",
            "slo_increase",
            "ready_cost_increase_seconds",
        ):
            values = [_finite(row, metric) for row in rows]
            record[metric] = float(np.mean(values))
            record[f"{metric}_sd"] = _sample_std(values)
            record[f"{metric}_min"] = min(values)
            record[f"{metric}_max"] = max(values)
        output.append(record)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def integrity_audit(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    config_path: Path,
    run_root: Path,
    runtime_contract_path: Path,
) -> dict[str, Any]:
    """Check every registered raw cell before exposing endpoint contrasts."""

    issues: list[dict[str, str]] = []
    counters: Counter[str] = Counter()
    expected = Counter(
        (str(method), int(seed))
        for seed in config["seeds"]
        for method in config["methods"]
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
            manifest = json.loads((run / "run_manifest.json").read_text())
            result = json.loads((run / "result.json").read_text())
            delivery = json.loads((run / "delivery.json").read_text())
            restoration = json.loads((run / "readiness_restoration.json").read_text())
            plan = json.loads((run / "request_plan.manifest.json").read_text())
            controller = json.loads(
                (run / "controller/controller_result.json").read_text()
            )
            actions = _jsonl(run / "controller/controller_actions.jsonl")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            fail(run_id, "artifact_parse", repr(error))
            continue
        method = str(manifest.get("method"))
        seed = int(manifest.get("seed", -1))
        observed[(method, seed)] += 1
        counters["parsed_runs"] += 1
        if (run / "a3_failure.json").exists():
            fail(run_id, "failure_artifact", "completed directory has a3_failure.json")
        metric = metric_by_id.get(run_id)
        if metric is None:
            fail(run_id, "aggregate_pairing", "run absent from aggregate")
            continue
        expected_id = f"screen_a3__gentd_inference__seed{seed}__{method}"
        if (
            run_id != expected_id
            or manifest.get("run_id") != run_id
            or manifest.get("matrix_row_id") != run_id
            or manifest.get("profile") != "gentd_inference"
            or (method, seed) not in expected
        ):
            fail(run_id, "run_identity", "directory or frozen identity differs")
        if manifest.get("status") != "completed" or result.get("status") != "completed":
            fail(run_id, "completion", "manifest or result is not completed")
        if manifest.get("runtime_audit_sha256") != contract_digest:
            fail(run_id, "runtime_contract", "contract digest differs")
        if manifest.get("config_sha256") != sha256_file(config_path):
            fail(run_id, "config", "runtime config digest differs")
        if (
            manifest.get("plan_sha256") != plan.get("plan_sha256")
            or plan.get("split") != "validation"
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
        if method in config["continuation_candidates"]:
            try:
                evidence = audit_candidate_action_log(
                    run,
                    candidate_id=method,
                    continuation_weight=float(config["continuation_candidates"][method]),
                )
                counters["candidate_action_steps"] += int(evidence["steps"])
                counters["candidate_q_argmax_verified"] += int(
                    evidence["q_argmax_verified"]
                )
            except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as error:
                fail(run_id, "candidate_action_audit", repr(error))

    raw_ids = {path.name for path in run_directories}
    if len(run_directories) != sum(expected.values()):
        fail("SCREEN_MATRIX", "run_directory_count", str(len(run_directories)))
    if set(metric_by_id) != raw_ids:
        fail("SCREEN_MATRIX", "aggregate_run_ids", "raw and aggregate run ids differ")
    if observed != expected:
        fail("SCREEN_MATRIX", "registered_cells", f"observed={observed},expected={expected}")
    inconsistent = {
        str(seed): sorted(hashes)
        for seed, hashes in plan_hashes.items()
        if len(hashes) != 1
    }
    if inconsistent or set(plan_hashes) != {int(seed) for seed in config["seeds"]}:
        fail("SCREEN_MATRIX", "paired_plan_hashes", json.dumps(inconsistent, sort_keys=True))
    execution_path = run_root / "execution_summary.json"
    execution = json.loads(execution_path.read_text()) if execution_path.is_file() else {}
    if (
        int(execution.get("registered_cells", -1)) != sum(expected.values())
        or int(execution.get("completed_this_invocation", -1)) != sum(expected.values())
        or int(execution.get("failed", -1)) != 0
    ):
        fail("SCREEN_MATRIX", "execution_summary", json.dumps(execution, sort_keys=True))
    return {
        "schema": "dap.k8s.pareto_calibration_screen_integrity.v1",
        "status": "PASS" if not issues else "FAIL",
        "expected_runs": sum(expected.values()),
        "observed_run_directories": len(run_directories),
        "observed_aggregate_rows": len(rows),
        "paired_plan_groups": len(plan_hashes),
        "counters": dict(sorted(counters.items())),
        "contract_sha256": contract_digest,
        "execution_summary": execution,
        "issues": issues,
    }


def _screen_report(ranked: list[dict[str, Any]], integrity: dict[str, Any]) -> str:
    lines = [
        "# A3 Stage-One Screen Report",
        "",
        "This is an authorized local exploratory validation screen (`n=4` paired plans per candidate), not a paper-level inferential result.",
        "",
        f"Integrity: **{integrity['status']}**.",
        "",
        "| Rank | Candidate | continuation | completion gain | SLO increase | Ready-cost increase (s) | point guards | normalized violation |",
        "|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for index, row in enumerate(ranked, start=1):
        lines.append(
            f"| {index} | {row['method']} | {row['continuation_weight']:.2f} | "
            f"{row['completion_gain']:.6f} | {row['slo_increase']:.6f} | "
            f"{row['ready_cost_increase_seconds']:.3f} | "
            f"{'PASS' if row['passed_all_point_guards'] else 'FAIL'} | "
            f"{row['normalized_guard_violation']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Frozen advancement: the first two candidates advance to six fresh validation plans. No equivalence, improvement, or publication claim is made from this screen.",
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
        raise FileExistsError(f"A3 analysis output is append-only: {output}")
    output.mkdir(parents=True, exist_ok=True)
    verify_runtime_contract(
        project_root=project_root,
        contract_path=runtime_contract_path,
        expected_config_path=config_path,
    )
    config = yaml.safe_load(config_path.read_text())
    validate_runtime_config(config)
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
        raise ValueError(f"A3 screen integrity failed with {len(integrity['issues'])} issues")
    _write_csv(output / "run_metrics.csv", rows)
    candidate_methods = tuple(config["continuation_candidates"])
    paired = paired_candidate_rows(rows, candidate_methods=candidate_methods)
    _write_csv(output / "paired_deltas.csv", paired)
    summaries = summarize_screen_candidates(
        paired,
        continuation_candidates=config["continuation_candidates"],
    )
    _write_csv(output / "candidate_summary.csv", summaries)
    margins = config["selection"]
    ranked = rank_screen_candidates(
        summaries,
        completion_loss_max=float(margins["completion_loss_max"]),
        slo_increase_max=float(margins["slo_increase_max"]),
        ready_cost_increase_seconds_max=float(
            margins["ready_cost_increase_seconds_max"]
        ),
    )
    advance_count = min(int(margins["advance_count"]), len(ranked))
    decision = {
        "schema": SCHEMA,
        "evidence_label": "authorized_exploratory_validation_screen",
        "inference_allowed": False,
        "ranked_candidates": ranked,
        "advanced_candidates": [row["method"] for row in ranked[:advance_count]],
        "selection_rule": "frozen_normalized_three_endpoint_point_guard_ranking",
    }
    write_json(output / "screen_ranking.json", decision)
    (output / "SCREEN_REPORT.md").write_text(
        _screen_report(ranked, integrity), encoding="utf-8"
    )
    audit = {
        "schema": "dap.k8s.pareto_calibration_analysis_audit.v1",
        "status": "PASS",
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(config_path),
        "runtime_contract_sha256": sha256_file(runtime_contract_path),
        "integrity_report_sha256": sha256_file(output / "integrity_report.json"),
        "run_metrics_sha256": sha256_file(output / "run_metrics.csv"),
        "paired_deltas_sha256": sha256_file(output / "paired_deltas.csv"),
        "candidate_summary_sha256": sha256_file(output / "candidate_summary.csv"),
        "screen_ranking_sha256": sha256_file(output / "screen_ranking.json"),
    }
    write_json(output / "analysis_audit.json", audit)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_analysis(
        project_root=args.project_root,
        config_path=args.config,
        runtime_contract_path=args.runtime_contract,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
