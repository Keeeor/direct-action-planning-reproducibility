"""Fail-closed integrity and frozen point-rule analysis for A3 validation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from dap.direct_action_planning_k8s_pareto_calibration.analysis import (
    action_horizon_valid,
    count_nonempty_lines,
    paired_candidate_rows,
    summarize_screen_candidates,
)
from dap.direct_action_planning_k8s_pareto_calibration.runner import (
    audit_candidate_action_log,
)
from dap.direct_action_planning_k8s_service_repair.prototype_api import (
    PROJECT_ROOT,
)
from dap.utils.artifacts import sha256_file, write_json

from .protocol import (
    CANDIDATES,
    activity_quantile_for,
    select_validation_candidate,
    validate_runtime_config,
)
from .runtime_audit import resolve, verify_runtime_contract

from analysis.aggregate import aggregate_run  # type: ignore  # noqa: E402


SCHEMA = "dap.k8s.pareto_validation_analysis.v1"


def evaluate_candidate(
    summary: Mapping[str, Any], *, selection: Mapping[str, Any]
) -> dict[str, Any]:
    """Expose every guard and SESOI without weakening the frozen conjunction."""

    row = dict(summary)
    completion_loss = float(row["completion_loss"])
    slo_increase = float(row["slo_increase"])
    cost_increase = float(row["ready_cost_increase_seconds"])
    budget_violation = float(row["budget_violation_seconds"])
    if not np.isfinite(
        [completion_loss, slo_increase, cost_increase, budget_violation]
    ).all():
        raise ValueError("nonfinite A3 validation summary")
    guards = {
        "completion": completion_loss
        <= float(selection["completion_loss_max"]) + 1.0e-12,
        "slo": slo_increase <= float(selection["slo_increase_max"]) + 1.0e-12,
        "ready_cost": cost_increase
        <= float(selection["ready_cost_increase_seconds_max"]) + 1.0e-12,
        "hard_budget": budget_violation <= 1.0e-9,
    }
    sesoi = {
        "completion": -completion_loss
        >= float(selection["completion_gain_sesoi"]),
        "slo": -slo_increase >= float(selection["slo_reduction_sesoi"]),
        "ready_cost": -cost_increase
        >= float(selection["ready_cost_reduction_seconds_sesoi"]),
    }
    row["point_guard_pass"] = guards
    row["passed_all_point_guards"] = all(guards.values())
    row["sesoi_pass"] = sesoi
    row["sesoi_count"] = sum(bool(value) for value in sesoi.values())
    row["eligible_for_locked_replay"] = bool(
        row["passed_all_point_guards"] and row["sesoi_count"] >= 1
    )
    return row


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
        if (run / "validation_a3_failure.json").exists():
            fail(run_id, "failure_artifact", "completed run contains failure artifact")
        metric = metric_by_id.get(run_id)
        if metric is None:
            fail(run_id, "aggregate_pairing", "run absent from aggregate")
            continue
        expected_id = f"validation_a3__gentd_inference__seed{seed}__{method}"
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
        if manifest.get("base_checkpoint_sha256") != (
            "sha256:8c9ab4d5419856e23d4d71b1bba5efcc7e0cc639f7d7d9bc87b799c65ba62a2c"
        ):
            fail(run_id, "checkpoint", "base checkpoint differs")
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
        if method in CANDIDATES:
            try:
                evidence = audit_candidate_action_log(
                    run,
                    candidate_id=method,
                    continuation_weight=CANDIDATES[method],
                )
                counters["candidate_action_steps"] += int(evidence["steps"])
                counters["candidate_q_argmax_verified"] += int(
                    evidence["q_argmax_verified"]
                )
            except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as error:
                fail(run_id, "candidate_action_audit", repr(error))
    raw_ids = {path.name for path in run_directories}
    expected_count = sum(expected.values())
    if len(run_directories) != expected_count:
        fail("VALIDATION_MATRIX", "run_directory_count", str(len(run_directories)))
    if set(metric_by_id) != raw_ids:
        fail("VALIDATION_MATRIX", "aggregate_run_ids", "raw and aggregate run IDs differ")
    if observed != expected:
        fail("VALIDATION_MATRIX", "registered_cells", f"observed={observed},expected={expected}")
    inconsistent = {
        str(seed): sorted(hashes)
        for seed, hashes in plan_hashes.items()
        if len(hashes) != 1
    }
    if inconsistent or set(plan_hashes) != {int(seed) for seed in config["seeds"]}:
        fail("VALIDATION_MATRIX", "paired_plan_hashes", json.dumps(inconsistent, sort_keys=True))
    execution_path = run_root / "execution_summary.json"
    execution = _json(execution_path) if execution_path.is_file() else {}
    if (
        int(execution.get("registered_cells", -1)) != expected_count
        or int(execution.get("completed_this_invocation", -1)) != expected_count
        or int(execution.get("failed", -1)) != 0
    ):
        fail("VALIDATION_MATRIX", "execution_summary", json.dumps(execution, sort_keys=True))
    return {
        "schema": "dap.k8s.pareto_validation_integrity.v1",
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


def _report(evaluated: list[dict[str, Any]], integrity: dict[str, Any], selected: list[str]) -> str:
    lines = [
        "# A3 Independent Validation Report",
        "",
        "This is an authorized exploratory selection analysis over six paired validation plans per candidate. It is not a publication-level inferential claim.",
        "",
        f"Integrity: **{integrity['status']}**.",
        "",
        "| Candidate | completion gain | SLO increase | Ready-cost increase (s) | all point guards | SESOI count | eligible |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for row in evaluated:
        lines.append(
            f"| {row['method']} | {row['completion_gain']:.6f} | "
            f"{row['slo_increase']:.6f} | {row['ready_cost_increase_seconds']:.3f} | "
            f"{'PASS' if row['passed_all_point_guards'] else 'FAIL'} | "
            f"{row['sesoi_count']} | "
            f"{'YES' if row['eligible_for_locked_replay'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "Frozen selected candidate(s): " + (", ".join(selected) if selected else "none"),
            "",
            "The validation means select whether a candidate may enter the planned 20-pair locked exploratory replay; they do not establish noninferiority or superiority.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_analysis(
    *, project_root: Path, config_path: Path, runtime_contract_path: Path, output: Path
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    runtime_contract_path = runtime_contract_path.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"A3 validation analysis is append-only: {output}")
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
        raise ValueError(
            f"A3 validation integrity failed with {len(integrity['issues'])} issues"
        )
    _write_csv(output / "run_metrics.csv", rows)
    paired = paired_candidate_rows(rows, candidate_methods=tuple(CANDIDATES))
    _write_csv(output / "paired_deltas.csv", paired)
    summaries = summarize_screen_candidates(
        paired, continuation_candidates=CANDIDATES
    )
    _write_csv(output / "candidate_summary.csv", summaries)
    evaluated = [
        evaluate_candidate(row, selection=config["selection"]) for row in summaries
    ]
    selected_rows = select_validation_candidate(
        summaries, selection=config["selection"]
    )
    selected = [str(row["method"]) for row in selected_rows]
    eligible = [
        str(row["method"])
        for row in evaluated
        if row["eligible_for_locked_replay"]
    ]
    if set(selected) - set(eligible):
        raise RuntimeError("selection implementation disagrees with exposed guards")
    decision = {
        "schema": SCHEMA,
        "evidence_label": "authorized_exploratory_stage_two_validation",
        "inference_allowed": False,
        "evaluated_candidates": evaluated,
        "eligible_candidates": eligible,
        "selected_candidate": selected[0] if selected else None,
        "locked_replay_authorized_by_frozen_rule": bool(selected),
    }
    write_json(output / "validation_decision.json", decision)
    (output / "VALIDATION_REPORT.md").write_text(
        _report(evaluated, integrity, selected), encoding="utf-8"
    )
    audit = {
        "schema": "dap.k8s.pareto_validation_analysis_audit.v1",
        "status": "PASS",
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(config_path),
        "runtime_contract_sha256": sha256_file(runtime_contract_path),
        "integrity_report_sha256": sha256_file(output / "integrity_report.json"),
        "run_metrics_sha256": sha256_file(output / "run_metrics.csv"),
        "paired_deltas_sha256": sha256_file(output / "paired_deltas.csv"),
        "candidate_summary_sha256": sha256_file(output / "candidate_summary.csv"),
        "validation_decision_sha256": sha256_file(output / "validation_decision.json"),
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
