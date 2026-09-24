"""Action-conditioned budget advantage research branch.

This package is additive. The stopped CDBA and dynamic-shadow-price packages remain
unchanged and are only consumed through read-only checkpoints where explicitly recorded.
"""

from .dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    ActionDPResult,
    solve_action_dp,
)

__all__ = [
    "ACBADPConfig",
    "ActionConditionedBudgetMDP",
    "ActionDPResult",
    "solve_action_dp",
]

