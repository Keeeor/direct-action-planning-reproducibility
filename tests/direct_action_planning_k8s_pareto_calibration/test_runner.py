from __future__ import annotations

import json
from pathlib import Path

from dap.direct_action_planning_k8s_pareto_calibration.runner import (
    audit_candidate_action_log,
    run_directory_matches,
)


def test_run_directory_identity_is_exact_or_append_only_attempt() -> None:
    run_id = "screen_a3__gentd_inference__seed2026081301__dap_cont_0p20"
    assert run_directory_matches(run_id, run_id)
    assert run_directory_matches(run_id + "__attempt2", run_id)
    assert not run_directory_matches(run_id + "_extra", run_id)
    assert not run_directory_matches(
        "screen_a3__gentd_inference__seed2026081301__dap_cont_0p2", run_id
    )


def test_candidate_action_audit_requires_32_feasible_argmax_actions(
    tmp_path: Path,
) -> None:
    controller = tmp_path / "controller"
    controller.mkdir()
    rows = []
    for step in range(32):
        rows.append({
            "step": step,
            "action": "scale_small",
            "greedy_action": "scale_small",
            "q_values": {"no_op": 0.0, "scale_small": 1.0},
            "feasible": {"no_op": True, "scale_small": True},
            "hard_mask_current_ready_replicas": 1,
        })
    (controller / "controller_actions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (controller / "a3_candidate.json").write_text(
        json.dumps({
            "candidate_id": "dap_cont_0p20",
            "continuation_weight": 0.2,
            "cost_weight": 1.5,
            "core_method_changed": False,
        }),
        encoding="utf-8",
    )
    evidence = audit_candidate_action_log(
        tmp_path, candidate_id="dap_cont_0p20", continuation_weight=0.2
    )
    assert evidence["q_argmax_verified"] == 32
    assert evidence["continuation_weight"] == 0.2

