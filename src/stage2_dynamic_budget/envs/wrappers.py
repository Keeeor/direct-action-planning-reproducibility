from __future__ import annotations

import numpy as np


def hard_budget_action_mask(action_costs: np.ndarray, remaining_budget: float) -> np.ndarray:
    costs = np.asarray(action_costs, dtype=np.float64)
    if costs.ndim != 1 or costs.size == 0 or np.any(costs < 0):
        raise ValueError("action_costs must be a non-empty non-negative vector")
    mask = costs <= max(float(remaining_budget), 0.0) + 1e-12
    mask[int(np.argmin(costs))] = True
    return mask.astype(bool)
