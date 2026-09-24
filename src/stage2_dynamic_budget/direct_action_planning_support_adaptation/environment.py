from __future__ import annotations

from stage2_dynamic_budget.action_conditioned_budget_advantage.branching import (
    BranchableDiscreteEnv,
)
from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
)


def make_branchable_env(
    mdp: ActionConditionedBudgetMDP,
    initial_budget: int,
) -> BranchableDiscreteEnv:
    env = BranchableDiscreteEnv(
        mdp.config,
        initial_budget=initial_budget,
        budget_scale=mdp.config.max_budget,
    )
    # BranchableDiscreteEnv reconstructs its canonical MDP. Replacing that
    # instance is required for continuous parameterized transition kernels.
    env.mdp = mdp
    return env
