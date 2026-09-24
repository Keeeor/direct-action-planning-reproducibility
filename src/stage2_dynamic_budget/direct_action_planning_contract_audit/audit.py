from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ACTION_ORDER = ("no_op", "scale_small", "scale_medium", "scale_large")
TARGET_BY_ACTION = {
    "no_op": 1,
    "scale_small": 2,
    "scale_medium": 3,
    "scale_large": 5,
}
TOTAL_BUDGET_SECONDS = 256.0
EXPECTED_MUTATION_CATEGORY = {
    "delivered_action": "execution_identity",
    "greedy_action": "score_maximizer",
    "target_mapping": "actuator_mapping",
    "shared_forecast": "shared_context",
    "selected_feasibility": "hard_feasibility",
    "ledger_closure": "ledger_closure",
}


def _different_action(action: str) -> str:
    return next(candidate for candidate in ACTION_ORDER if candidate != action)


def verify_record(record: dict[str, Any]) -> set[str]:
    """Return the violated runtime-contract invariant categories."""

    findings: set[str] = set()
    feasible = record.get("feasible")
    q_values = record.get("q_values")
    branches = record.get("branches")
    if not isinstance(feasible, dict) or set(feasible) != set(ACTION_ORDER):
        return {"record_schema"}
    if not isinstance(q_values, dict) or set(q_values) != set(ACTION_ORDER):
        return {"record_schema"}
    if not isinstance(branches, dict):
        return {"record_schema"}
    expected_branches = {
        candidate for candidate in ACTION_ORDER if feasible[candidate] is True
    }
    if set(branches) != expected_branches:
        findings.add("record_schema")

    action = str(record.get("action"))
    greedy = str(record.get("greedy_action"))
    if feasible.get("no_op") is not True or feasible.get(action) is not True:
        findings.add("hard_feasibility")

    feasible_q = {
        candidate: float(q_values[candidate])
        for candidate in ACTION_ORDER
        if feasible[candidate] is True
    }
    if not feasible_q:
        findings.add("hard_feasibility")
    else:
        expected_greedy = max(
            ACTION_ORDER,
            key=lambda candidate: feasible_q.get(candidate, float("-inf")),
        )
        if greedy != expected_greedy:
            findings.add("score_maximizer")
        best = max(feasible_q.values())
        if action not in feasible_q or feasible_q[action] < best - 1.0e-10:
            findings.add("execution_identity")

    if record.get("target_replicas") != TARGET_BY_ACTION.get(action):
        findings.add("actuator_mapping")

    try:
        common_forecast = float(record["predicted_load_rps"])
        branch_forecasts = {
            float(branch["forecast_arrival_rps"])
            for branch in branches.values()
        }
        if branch_forecasts != {common_forecast}:
            findings.add("shared_context")
    except (KeyError, TypeError, ValueError):
        findings.add("shared_context")

    try:
        used = float(record["budget_sample"]["cumulative_ready_cost"])
        remaining = float(record["budget_remaining_seconds"])
        if (
            used < -1.0e-9
            or used > TOTAL_BUDGET_SECONDS + 1.0e-9
            or abs(remaining - (TOTAL_BUDGET_SECONDS - used)) > 1.0e-6
        ):
            findings.add("ledger_closure")
    except (KeyError, TypeError, ValueError):
        findings.add("ledger_closure")
    return findings


def mutate_record(record: dict[str, Any], mutation: str) -> dict[str, Any]:
    """Apply one deterministic, single-field contract fault."""

    if mutation not in EXPECTED_MUTATION_CATEGORY:
        raise KeyError(mutation)
    output = copy.deepcopy(record)
    if mutation == "delivered_action":
        feasible = [name for name in ACTION_ORDER if output["feasible"][name]]
        output["action"] = min(
            feasible,
            key=lambda name: float(output["q_values"][name]),
        )
        if output["action"] == record["action"]:
            output["action"] = _different_action(str(record["action"]))
    elif mutation == "greedy_action":
        output["greedy_action"] = _different_action(str(record["greedy_action"]))
    elif mutation == "target_mapping":
        output["target_replicas"] = int(record["target_replicas"]) + 1
    elif mutation == "shared_forecast":
        first = ACTION_ORDER[0]
        output["branches"][first]["forecast_arrival_rps"] = (
            float(output["branches"][first]["forecast_arrival_rps"]) + 1.0
        )
    elif mutation == "selected_feasibility":
        output["feasible"][str(record["action"])] = False
    elif mutation == "ledger_closure":
        output["budget_remaining_seconds"] = (
            float(record["budget_remaining_seconds"]) + 1.0
        )
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def discover_logs(project_root: Path) -> list[Path]:
    service = sorted(
        (
            project_root
            / "research/direct_action_planning_k8s_service_repair/results/runs/formal_v2"
        ).glob("formal__*__*__seed*__dap_repaired/controller/controller_actions.jsonl")
    )
    replay = sorted(
        (
            project_root
            / "research/direct_action_planning_k8s_pareto_calibration/results/runs/locked_replay_a3"
        ).glob(
            "locked_replay_a3__gentd_inference__seed*__dap_cont_0p05/"
            "controller/controller_actions.jsonl"
        )
    )
    if len(service) != 80 or len(replay) != 20:
        raise AssertionError(
            f"expected 80 registered and 20 replay logs, found {len(service)} and {len(replay)}"
        )
    return service + replay


def _input_digest(project_root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def run_audit(project_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = discover_logs(project_root)
    clean_records = 0
    clean_findings = 0
    mutation_rows: list[dict[str, Any]] = []
    by_type = {
        mutation: {"attempted": 0, "detected": 0, "localized": 0}
        for mutation in EXPECTED_MUTATION_CATEGORY
    }
    for path in paths:
        rows = _read_jsonl(path)
        if len(rows) != 32 or [int(row["step"]) for row in rows] != list(range(32)):
            raise AssertionError(f"{path}: expected ordered steps 0..31")
        for row in rows:
            clean_records += 1
            original_findings = verify_record(row)
            clean_findings += int(bool(original_findings))
            if original_findings:
                raise AssertionError(
                    f"clean contract finding in {path} step {row['step']}: {sorted(original_findings)}"
                )
            for mutation, expected in EXPECTED_MUTATION_CATEGORY.items():
                findings = verify_record(mutate_record(row, mutation))
                detected = bool(findings)
                localized = expected in findings
                by_type[mutation]["attempted"] += 1
                by_type[mutation]["detected"] += int(detected)
                by_type[mutation]["localized"] += int(localized)
                mutation_rows.append(
                    {
                        "mutation": mutation,
                        "expected_category": expected,
                        "detected": int(detected),
                        "localized": int(localized),
                    }
                )
    attempted = sum(row["attempted"] for row in by_type.values())
    detected = sum(row["detected"] for row in by_type.values())
    localized = sum(row["localized"] for row in by_type.values())
    summary = {
        "schema": "dap.contract_replay_audit.v1",
        "status": "pass"
        if clean_findings == 0 and attempted == detected == localized == 19_200
        else "fail",
        "source": "all frozen DAP Kubernetes controller action records",
        "input_log_count": len(paths),
        "input_digest": _input_digest(project_root, paths),
        "clean_records": clean_records,
        "clean_records_with_findings": clean_findings,
        "mutations_attempted": attempted,
        "mutations_detected": detected,
        "mutations_localized": localized,
        "detection_rate": detected / attempted,
        "localization_rate": localized / attempted,
        "by_mutation": by_type,
        "interpretation": (
            "Deterministic single-field mutation validation of the executable "
            "decision-record contract; not an estimate of real fault frequency "
            "or human debugging time."
        ),
    }
    if summary["status"] != "pass":
        raise AssertionError(f"contract replay audit failed: {summary}")
    return summary, mutation_rows


def write_results(project_root: Path, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, mutation_rows = run_audit(project_root)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_path = output_dir / "mutation_outcomes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mutation_rows[0]))
        writer.writeheader()
        writer.writerows(mutation_rows)
    plan_path = project_root / "research/direct_action_planning_contract_audit/PLAN.md"
    code_path = Path(__file__).resolve()
    manifest = {
        "schema": "dap.contract_replay_manifest.v1",
        "status": summary["status"],
        "inputs": {
            "aggregate_log_digest": summary["input_digest"],
            "log_count": summary["input_log_count"],
            "plan": _sha256(plan_path),
            "code": _sha256(code_path),
        },
        "outputs": {
            "summary.json": _sha256(summary_path),
            "mutation_outcomes.csv": _sha256(csv_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_results(args.project_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
