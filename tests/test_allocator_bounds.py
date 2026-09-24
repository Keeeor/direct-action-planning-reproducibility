import numpy as np
import pytest

from stage2_dynamic_budget.models.budget_allocator import local_budget_from_q


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_local_budget_respects_dynamic_bounds(q: float) -> None:
    budget = local_budget_from_q(
        mean_remaining_rate=2.0,
        q=q,
        eta=2.0,
        min_multiplier=0.25,
        max_multiplier=4.0,
    )
    assert 0.5 <= budget <= 8.0


def test_local_budget_midpoint_is_anchor() -> None:
    assert local_budget_from_q(3.0, 0.5, 1.7, 0.25, 4.0) == pytest.approx(3.0)


def test_local_budget_is_monotone_in_allocator_output() -> None:
    values = [local_budget_from_q(2.0, q, 1.0, 0.25, 4.0) for q in np.linspace(0, 1, 21)]
    assert all(a <= b for a, b in zip(values, values[1:]))
