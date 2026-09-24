from __future__ import annotations

import copy

from stage2_dynamic_budget.direct_action_planning_contract_audit.audit import (
    ACTION_ORDER,
    EXPECTED_MUTATION_CATEGORY,
    mutate_record,
    verify_record,
)


def _record() -> dict:
    return {
        "action": "scale_small",
        "greedy_action": "scale_small",
        "target_replicas": 2,
        "predicted_load_rps": 7.0,
        "feasible": {name: True for name in ACTION_ORDER},
        "q_values": {
            "no_op": 0.0,
            "scale_small": 3.0,
            "scale_medium": 2.0,
            "scale_large": 1.0,
        },
        "branches": {
            name: {"forecast_arrival_rps": 7.0} for name in ACTION_ORDER
        },
        "budget_remaining_seconds": 246.0,
        "budget_sample": {"cumulative_ready_cost": 10.0},
    }


def test_clean_record_has_no_contract_violation() -> None:
    assert verify_record(_record()) == set()


def test_each_registered_mutation_is_detected_and_localized() -> None:
    clean = _record()
    for mutation, category in EXPECTED_MUTATION_CATEGORY.items():
        mutated = mutate_record(copy.deepcopy(clean), mutation)
        findings = verify_record(mutated)
        assert findings
        assert category in findings
