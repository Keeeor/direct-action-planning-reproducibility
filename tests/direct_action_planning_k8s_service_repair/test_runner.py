from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage2_dynamic_budget.direct_action_planning_k8s_service_repair.runner import (
    _audit_repaired_action_log,
    matrix_cells,
)


def test_pilot_matrix_contains_only_registered_profile_specific_comparators() -> None:
    config = {
        "profiles": {"azure_http": {}, "gentd_inference": {}},
        "seeds": [7],
        "methods_by_profile": {
            "azure_http": ["dap_repaired", "mpc_4"],
            "gentd_inference": ["dap_repaired", "threshold"],
        },
        "perturbations": [],
    }
    assert matrix_cells(config) == [
        ("azure_http", "dap_repaired", "control", 7),
        ("azure_http", "mpc_4", "control", 7),
        ("gentd_inference", "dap_repaired", "control", 7),
        ("gentd_inference", "threshold", "control", 7),
    ]


def test_action_log_audit_rejects_non_argmax_greedy_action(tmp_path: Path) -> None:
    controller = tmp_path / "controller"
    controller.mkdir()
    row = {
        "step": 0, "model_ready_replicas": 1,
        "hard_mask_current_ready_replicas": 1,
        "greedy_action": "no_op",
        "q_values": {"no_op": 0.0, "scale_small": 1.0},
        "feasible": {"no_op": True, "scale_small": True},
    }
    (controller / "controller_actions.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="not recorded Q argmax"):
        _audit_repaired_action_log(tmp_path)

