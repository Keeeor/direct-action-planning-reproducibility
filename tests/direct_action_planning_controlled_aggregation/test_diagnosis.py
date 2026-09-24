from __future__ import annotations

import pandas as pd

from stage2_dynamic_budget.direct_action_planning_controlled_aggregation.diagnosis import (
    duplicate_and_coverage,
    empirical_anchor_labels,
)


def test_empirical_anchor_uses_only_visited_cells_and_normalizes_probabilities() -> None:
    rows = []
    for action in (0, 1):
        rows.append(
            {
                "split": "validation",
                "t": 0,
                "load": 1,
                "queue": 0,
                "remaining_budget": 2,
                "action": action,
                "q_lv": 1.0 + action,
                "q_star": 1.0 + action,
                "next_load_prob_0": 1.0,
                "next_load_prob_1": 0.0,
                "next_load_prob_2": 0.0,
            }
        )
    samples = pd.DataFrame(
        [
            {"t": 0, "load": 1, "queue": 0, "action": action, "true_next_load": nxt}
            for action in (0, 1)
            for nxt in (0, 1, 1)
        ]
    )
    anchor = empirical_anchor_labels(pd.DataFrame(rows), samples)
    assert len(anchor) == 2
    assert (anchor[["next_load_prob_0", "next_load_prob_1", "next_load_prob_2"]].sum(axis=1) == 1).all()


def test_duplicate_and_coverage_separates_grid_coverage_from_revisits() -> None:
    d0 = pd.DataFrame(
        [
            {"t": 0, "load": load, "queue": 0, "action": 0, "true_next_load": load}
            for load in (0, 1)
        ]
    )
    samples = pd.concat([d0[d0.load == 0]] * 3, ignore_index=True)
    metrics = duplicate_and_coverage(samples, d0)
    assert metrics["grid_coverage"] == 0.5
    assert metrics["transition_duplicate_rate"] > 0.0
