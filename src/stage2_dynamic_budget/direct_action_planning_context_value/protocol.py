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
    selection: dict[str, object]
    validation_sha256: str
    test_evaluations_completed: int = 0
    test_status: str = "not_started"

    @classmethod
    def create(
        cls,
        path: str | Path,
        selection: dict[str, object],
        validation_sha256: str,
    ) -> "FinalTestLedger":
        path = Path(path)
        if path.exists():
            raise RuntimeError(f"selection ledger already exists: {path}")
        ledger = cls(path, dict(selection), validation_sha256)
        ledger._write()
        return ledger

    @classmethod
    def load(cls, path: str | Path) -> "FinalTestLedger":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            selection=dict(payload["selection"]),
            validation_sha256=str(payload["validation_sha256"]),
            test_evaluations_completed=int(payload["test_evaluations_completed"]),
            test_status=str(payload["test_status"]),
        )

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "direct_action_planning_context_value.selection.v1",
            "updated_at": _now(),
            "selection_source": "validation_only",
            "selection": self.selection,
            "validation_sha256": self.validation_sha256,
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
