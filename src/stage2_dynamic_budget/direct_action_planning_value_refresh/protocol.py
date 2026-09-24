from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FinalTestLedger:
    path: Path
    cells: list[dict[str, object]]
    test_evaluations_completed: int = 0
    test_status: str = "not_started"

    @classmethod
    def create(cls, path: str | Path, cells: list[dict[str, object]]) -> "FinalTestLedger":
        path = Path(path)
        if path.exists():
            raise RuntimeError(f"selection ledger already exists: {path}")
        ledger = cls(path=path, cells=cells)
        ledger._write()
        return ledger

    @classmethod
    def load(cls, path: str | Path) -> "FinalTestLedger":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            cells=list(payload["cells"]),
            test_evaluations_completed=int(payload["test_evaluations_completed"]),
            test_status=str(payload["test_status"]),
        )

    def _write(self) -> None:
        payload = {
            "schema": "direct_action_planning_value_refresh.selection.v1",
            "updated_at": _now(),
            "test_evaluations_completed": self.test_evaluations_completed,
            "test_status": self.test_status,
            "selection_source": "validation_only",
            "test_used_for_selection": False,
            "cells": self.cells,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
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

