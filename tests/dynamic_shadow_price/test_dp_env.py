import numpy as np
import pytest

from dap.dynamic_shadow_price.dp_env import DiscreteBudgetGymEnv
from dap.dynamic_shadow_price.dp_reference import DiscreteDPConfig


def test_dp_gym_env_matches_budget_accounting_and_canonical_observation():
    env = DiscreteBudgetGymEnv(
        DiscreteDPConfig(horizon=8, max_budget=6, scenario="early_burst"),
        initial_budget=4,
    )
    obs, _ = env.reset(seed=7)
    assert obs.shape == (14,)
    assert obs[-2] == 1.0
    assert obs[-1] == 1.0
    assert env.valid_action_mask().tolist() == [True, True, True, True]
    next_obs, _, terminated, truncated, info = env.step(3)
    assert info["resource_cost"] == 3.0
    assert info["remaining_budget"] == 1.0
    assert np.isclose(next_obs[-2], 0.25)
    assert not terminated and not truncated
    assert env.valid_action_mask().tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match="remaining budget"):
        env.step(2)


def test_dp_env_can_expose_absolute_budget_on_shared_maximum_scale():
    config = DiscreteDPConfig(horizon=8, max_budget=12, scenario="late_burst")
    env = DiscreteBudgetGymEnv(config, initial_budget=4, budget_scale=12)
    obs, _ = env.reset(seed=2)
    assert np.isclose(obs[-2], 4 / 12)
    env.step(1)
    assert np.isclose(env._observation()[-2], 3 / 12)
