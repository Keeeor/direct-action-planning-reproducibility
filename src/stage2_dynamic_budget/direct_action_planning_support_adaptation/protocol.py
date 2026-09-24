from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


PARAMETER_COLUMNS = (
    "base_load",
    "burst_start",
    "burst_amplitude",
    "burst_duration",
    "period",
    "periodic_amplitude",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parameter_key(row: dict[str, object]) -> tuple[float, ...]:
    return tuple(round(float(row[column]), 12) for column in PARAMETER_COLUMNS)


def assert_parameter_splits_disjoint(
    splits: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    ids: dict[str, list[str]] = {}
    parameters: dict[tuple[float, ...], list[str]] = {}
    for split, rows in splits.items():
        for row in rows:
            row_id = str(row["id"])
            ids.setdefault(row_id, []).append(split)
            parameters.setdefault(_parameter_key(row), []).append(f"{split}:{row_id}")
    duplicate_ids = sorted(key for key, owners in ids.items() if len(owners) > 1)
    duplicate_parameters = sorted(
        owners for owners in parameters.values() if len(owners) > 1
    )
    audit = {
        "schema": "direct_action_planning_support_adaptation.parameter_audit.v1",
        "passed": not duplicate_ids and not duplicate_parameters,
        "rows": int(sum(len(rows) for rows in splits.values())),
        "duplicate_ids": duplicate_ids,
        "duplicate_parameter_rows": duplicate_parameters,
    }
    if not audit["passed"]:
        raise ValueError(f"scenario parameter split leakage: {audit}")
    return audit


def split_target_prefix_suffix(
    frame: pd.DataFrame,
    *,
    ratio: float,
    horizon: int,
    prefix_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required = {"trajectory_id", "t", "row_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"target trajectories require columns {sorted(required)}")
    if not 0.0 <= ratio <= prefix_fraction <= 1.0:
        raise ValueError("ratio must be in [0, prefix_fraction]")
    cutoff = int(np.ceil(horizon * prefix_fraction))
    prefix = frame[frame.t < cutoff].copy()
    suffix = frame[frame.t >= cutoff].copy()
    requested = int(np.floor(ratio * len(frame) + 1.0e-12))
    if ratio > 0.0 and requested == 0:
        requested = 1
    if requested > len(prefix):
        raise ValueError("requested calibration fraction exceeds prefix pool")
    order = np.random.default_rng(seed).permutation(len(prefix))
    calibration = prefix.iloc[order[:requested]].sort_values("row_id").reset_index(drop=True)
    suffix = suffix.sort_values("row_id").reset_index(drop=True)
    overlap = sorted(set(calibration.row_id).intersection(suffix.row_id))
    audit = {
        "schema": "direct_action_planning_support_adaptation.prefix_suffix_audit.v1",
        "passed": not overlap and (calibration.empty or int(calibration.t.max()) < cutoff),
        "requested_ratio": float(ratio),
        "realized_ratio": float(len(calibration) / max(len(frame), 1)),
        "prefix_cutoff_t": cutoff,
        "calibration_rows": int(len(calibration)),
        "suffix_rows": int(len(suffix)),
        "overlap_row_ids": overlap,
    }
    if not audit["passed"]:
        raise RuntimeError(f"target prefix/suffix leakage: {audit}")
    return calibration, suffix, audit


@dataclass
class FinalTestLedger:
    path: Path
    selection: dict[str, object]
    test_evaluations_completed: int = 0
    test_status: str = "not_started"

    @classmethod
    def create(cls, path: str | Path, selection: dict[str, object]) -> "FinalTestLedger":
        path = Path(path)
        if path.exists():
            raise RuntimeError(f"selection ledger already exists: {path}")
        ledger = cls(path, dict(selection))
        ledger._write()
        return ledger

    @classmethod
    def load(cls, path: str | Path) -> "FinalTestLedger":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            selection=dict(payload["selection"]),
            test_evaluations_completed=int(payload["test_evaluations_completed"]),
            test_status=str(payload["test_status"]),
        )

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "direct_action_planning_support_adaptation.selection.v1",
            "updated_at": _now(),
            "selection_source": "validation_only",
            "selection": self.selection,
            "test_used_for_selection": False,
            "test_evaluations_completed": self.test_evaluations_completed,
            "test_status": self.test_status,
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def mark_test_started(self) -> None:
        if self.test_status != "not_started" or self.test_evaluations_completed != 0:
            raise RuntimeError("final test already started or completed")
        self.test_status = "started"
        self.test_evaluations_completed = 1
        self._write()

    def mark_test_completed(self) -> None:
        if self.test_status != "started" or self.test_evaluations_completed != 1:
            raise RuntimeError("final test was not started exactly once")
        self.test_status = "completed"
        self._write()
