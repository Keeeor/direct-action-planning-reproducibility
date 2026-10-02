from __future__ import annotations

from pathlib import Path

import pytest

from dap.direct_action_planning_k8s_service_repair.audit import (
    build_inventory,
    verify_inventory,
)


def test_inventory_detects_content_drift(tmp_path: Path) -> None:
    path = tmp_path / "source.py"
    path.write_text("value = 1\n", encoding="utf-8")
    frozen = build_inventory([path], root=tmp_path)
    verify_inventory(frozen, root=tmp_path)
    path.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory drift"):
        verify_inventory(frozen, root=tmp_path)


def test_inventory_rejects_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "source.py"
    path.write_text("value = 1\n", encoding="utf-8")
    frozen = build_inventory([path], root=tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match="missing inventory file"):
        verify_inventory(frozen, root=tmp_path)

