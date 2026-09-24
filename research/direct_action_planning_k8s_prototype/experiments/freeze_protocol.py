from __future__ import annotations

"""Freeze the formal Kubernetes protocol only after real Pilot integrity gates."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.aggregate import aggregate_run
from experiments.run_matrix import matrix_rows, validate_matrix_config


# Kept local to avoid making the freeze protocol depend on a runtime module's
# private helper. The hash covers exactly the files that determine an execution.
def _source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for directory_name in ("app", "controller", "baselines", "workload", "calibration", "experiments", "analysis", "kubernetes", "tests"):
        directory = root / directory_name
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(path.read_bytes())
    makefile = root / "Makefile"
    if makefile.exists():
        digest.update("Makefile".encode("utf-8"))
        digest.update(makefile.read_bytes())
    return "sha256:" + digest.hexdigest()


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(config_path: Path, value: str | Path) -> Path:
    return (config_path.parent / Path(value)).resolve()


def _completed_pilot_rows(pilot_root: Path, *, source_tree_sha256: str) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for path in sorted(pilot_root.rglob("run_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "completed"
            or not manifest.get("run_id")
            or manifest.get("source_tree_sha256") != source_tree_sha256
        ):
            continue
        completed[str(manifest["run_id"])] = aggregate_run(path.parent)
    return completed


def evaluate_pilot(pilot_config: dict[str, Any], *, pilot_config_path: Path) -> dict[str, Any]:
    validate_matrix_config(pilot_config, config_path=pilot_config_path)
    if pilot_config["mode"] != "pilot":
        raise ValueError("pilot gate requires a pilot config")
    root = _relative(pilot_config_path, pilot_config["paths"]["run_root"])
    expected = {row["run_id"] for row in matrix_rows(pilot_config)}
    completed = _completed_pilot_rows(root, source_tree_sha256=_source_tree_sha256(ROOT)) if root.exists() else {}
    missing = sorted(expected - set(completed))
    failures: list[dict[str, Any]] = []
    for run_id, row in sorted(completed.items()):
        if row["budget_violation_seconds"] > 1.0e-9:
            failures.append({"run_id": run_id, "gate": "hard_budget", "value": row["budget_violation_seconds"]})
        missing_rate = row["metric_missing_rate"]
        if not (missing_rate >= 0.0 and missing_rate < 0.01):
            failures.append({"run_id": run_id, "gate": "metrics", "value": missing_rate})
        if row["total_requests"] <= 0 or row["total_requests"] != row["completed_requests"] + row["failed_requests"]:
            failures.append({"run_id": run_id, "gate": "replay_accounting", "value": row["total_requests"]})
        if row["method"] == "dap" and row["controller_deadline_misses"] > 1:
            failures.append({"run_id": run_id, "gate": "dap_deadline", "value": row["controller_deadline_misses"]})
    return {
        "schema": "dap.k8s.pilot_gate.v1",
        "pilot_root": str(root),
        "expected_rows": len(expected),
        "completed_rows": len(completed),
        "missing_run_ids": missing,
        "failures": failures,
        "status": "passed" if not missing and not failures else "not_ready",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def freeze(formal_config_path: Path, *, pilot_config_path: Path) -> dict[str, Any]:
    formal_config_path = formal_config_path.resolve()
    pilot_config_path = pilot_config_path.resolve()
    formal = yaml.safe_load(formal_config_path.read_text(encoding="utf-8"))
    pilot = yaml.safe_load(pilot_config_path.read_text(encoding="utf-8"))
    validate_matrix_config(formal, config_path=formal_config_path)
    if formal["mode"] != "formal":
        raise ValueError("formal config must have mode=formal")
    gate = evaluate_pilot(pilot, pilot_config_path=pilot_config_path)
    gate_path = ROOT / "results" / "pilot_gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if gate["status"] != "passed":
        raise RuntimeError(f"pilot gates are not ready; see {gate_path}")

    test_plan_root = _relative(formal_config_path, formal["paths"]["plan_root"])
    if test_plan_root.exists() and any(test_plan_root.rglob("*.jsonl")):
        raise RuntimeError("sealed test request plans already exist; formal freeze must precede test-plan generation")
    manifest_path = _relative(formal_config_path, formal["frozen_config_manifest"])
    if manifest_path.exists() and "Status: **FROZEN**" in manifest_path.read_text(encoding="utf-8"):
        raise RuntimeError(f"formal configuration is already frozen: {manifest_path}")

    artifacts: dict[str, str] = {
        "formal_config_sha256": _sha256(formal_config_path),
        "pilot_config_sha256": _sha256(pilot_config_path),
        "pilot_gate_sha256": _sha256(gate_path),
        "system_model_sha256": _sha256(_relative(formal_config_path, formal["paths"]["system_model"])),
        "source_tree_sha256": _source_tree_sha256(ROOT),
    }
    for profile, values in formal["profiles"].items():
        artifacts[f"checkpoint_{profile}_sha256"] = _sha256(_relative(formal_config_path, values["checkpoint"]))
    for path in sorted((ROOT / "kubernetes").rglob("*.yaml")):
        artifacts[f"kubernetes_{path.relative_to(ROOT)}_sha256"] = _sha256(path)
    frozen_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Frozen Configuration Manifest",
        "",
        "Status: **FROZEN**.",
        "",
        f"Frozen at: `{frozen_at}`",
        "",
        "The Pilot integrity gates passed before this formal configuration was frozen. "
        "Formal request plans are generated only after this timestamp from the registered test split.",
        "",
        "## Bound Artifacts",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in sorted(artifacts.items()))
    lines.extend([
        "",
        "## Fixed Protocol",
        "",
        f"- Profiles: `{', '.join(formal['profiles'])}`",
        f"- Budgets (Ready-replica seconds): `{formal['budgets']}`",
        f"- Methods: `{', '.join(formal['methods'])}`",
        f"- Repetitions: `{formal['repetitions']}`",
        f"- Controller interval / horizon: `{formal['control_interval_seconds']} s / {formal['horizon_steps']} steps`",
        f"- Test plan directory: `{test_plan_root}`",
        "",
        "No formal run may alter these bindings, select a checkpoint, or regenerate an existing request plan.",
    ])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    preregistration = ROOT / "docs" / "PREREGISTRATION.md"
    text = preregistration.read_text(encoding="utf-8")
    text = text.replace("Status: **DRAFT - NOT FROZEN**. It becomes frozen only after all pilot gates\npass.", f"Status: **FROZEN** on `{frozen_at}` after registered Pilot integrity gates passed.")
    preregistration.write_text(text, encoding="utf-8")
    return {"status": "frozen", "manifest": str(manifest_path), "pilot_gate": gate}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="formal config")
    parser.add_argument("--pilot-config", type=Path, default=ROOT / "experiments" / "configs" / "pilot.yaml")
    args = parser.parse_args()
    result = freeze(args.config, pilot_config_path=args.pilot_config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
