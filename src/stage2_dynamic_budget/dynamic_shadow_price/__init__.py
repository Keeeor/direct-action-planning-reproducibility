"""Independent Dynamic Shadow Pricing research branch.

The frozen CDBA implementation remains in the original modules.  Everything in
this package is additive so that old run manifests remain reproducible.
"""

from .dp_reference import DiscreteBudgetMDP, DiscreteDPConfig, solve_backward_dp

__all__ = ["DiscreteBudgetMDP", "DiscreteDPConfig", "solve_backward_dp"]
