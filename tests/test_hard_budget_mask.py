import numpy as np

from stage2_dynamic_budget.envs.wrappers import hard_budget_action_mask


def test_over_budget_actions_are_masked_and_noop_remains() -> None:
    mask = hard_budget_action_mask(np.asarray([0.0, 1.0, 2.0, 4.0]), remaining_budget=1.5)
    assert mask.tolist() == [True, True, False, False]
    assert bool(mask[0])


def test_negative_remaining_budget_still_keeps_minimum_cost_action() -> None:
    mask = hard_budget_action_mask(np.asarray([0.0, 1.0, 2.0]), remaining_budget=-1.0)
    assert mask.tolist() == [True, False, False]
