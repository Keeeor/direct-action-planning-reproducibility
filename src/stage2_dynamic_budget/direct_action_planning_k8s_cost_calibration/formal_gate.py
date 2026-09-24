"""Bind A2 formal execution to the pre-effect operational pilot gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import yaml

from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json

from .runtime_audit import resolve, validate_runtime_config


SCHEMA = "dap.k8s.cost_calibration_formal_gate.v1"


def validate_pilot_gate(*, config_path: Path) -> dict[str, Any]:
    """Reject formal execution unless the frozen operational pilot passed."""
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_runtime_config(config)
    if config.get("mode") != "formal_a2":
        raise ValueError("pilot-to-formal gate requires formal_a2 config")
    binding = config.get("pilot_gate")
    if not isinstance(binding, dict):
        raise ValueError("formal config lacks pilot gate binding")
    if binding.get("required_status") != "PASS":
        raise ValueError("formal pilot gate status requirement drift")
    report_path = resolve(config_path, binding.get("report", ""))
    expected_sha = binding.get("report_sha256")
    actual_sha = sha256_file(report_path)
    if expected_sha != actual_sha:
        raise ValueError("pilot integrity report hash drift")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_checks = {
        "completed": "4/4",
        "runtime_contract_binding": "4/4",
        "plan_manifest_binding": "4/4",
        "request_accounting": "4/4",
        "hard_ready_budget": "4/4",
        "controller_deadline": "4/4",
        "monitor_missing_rate_below_0_01": "4/4",
        "delivery": "4/4",
        "readiness_restoration": "4/4",
        "action_horizon_32": "4/4",
        "calibrated_feasible_q_argmax": "64/64",
    }
    if (
        report.get("schema") != "dap.k8s.cost_calibration_pilot_integrity.v1"
        or report.get("status") != "PASS"
        or int(report.get("expected_runs", -1)) != 4
        or int(report.get("observed_runs", -1)) != 4
        or int(report.get("failed_runs", -1)) != 0
        or report.get("issues") != []
        or report.get("checks") != expected_checks
        or report.get("service_metrics_not_used_for_tuning") is not True
    ):
        raise ValueError("pilot integrity report does not satisfy formal gate")
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "formal_config_path": config_path.as_posix(),
        "formal_config_sha256": sha256_file(config_path),
        "pilot_report_path": report_path.as_posix(),
        "pilot_report_sha256": actual_sha,
        "pilot_runtime_contract_sha256": report["runtime_contract_sha256"],
        "effect_selection_from_pilot": False,
        "formal_execution_authorized": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = validate_pilot_gate(config_path=args.config)
    write_json(args.output, evidence)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
