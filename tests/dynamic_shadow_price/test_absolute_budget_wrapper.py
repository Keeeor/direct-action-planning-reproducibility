import numpy as np

from dap.dynamic_shadow_price.synthetic_joint import (
    AbsoluteBudgetObservationWrapper,
)
from dap.envs.synthetic_queue_env import (
    DynamicBudgetSchedulingEnv,
    SyntheticQueueConfig,
)


def test_absolute_budget_wrapper_uses_shared_scale_without_changing_accounting():
    base = DynamicBudgetSchedulingEnv(
        SyntheticQueueConfig(horizon=8, budget=10, scenario="stable")
    )
    env = AbsoluteBudgetObservationWrapper(base, budget_scale=20)
    obs, _ = env.reset(seed=1)
    assert np.isclose(obs[-2], 0.5)
    next_obs, _, _, _, info = env.step(2)
    assert info["resource_cost"] == 2.0
    assert info["remaining_budget"] == 8.0
    assert np.isclose(next_obs[-2], 0.4)
