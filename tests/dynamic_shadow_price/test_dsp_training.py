import numpy as np
import pytest
import torch

from stage2_dynamic_budget.dynamic_shadow_price.dp_env import DiscreteBudgetGymEnv
from stage2_dynamic_budget.dynamic_shadow_price.dp_reference import DiscreteDPConfig
from stage2_dynamic_budget.dynamic_shadow_price.dsp_trainer import (
    DSPTrainer,
    DSPTrainerConfig,
)
from stage2_dynamic_budget.dynamic_shadow_price.shadow_policy import (
    DSPPolicyConfig,
    DynamicShadowPricePolicy,
)


@pytest.mark.parametrize("variant", ["dsp_a", "dsp_b"])
def test_dsp_training_smoke_is_finite_and_budget_feasible(variant):
    env_config = DiscreteDPConfig(
        horizon=8, max_budget=6, scenario="early_burst"
    )
    policy = DynamicShadowPricePolicy(
        DSPPolicyConfig(
            variant=variant,
            episode_budget=6.0,
            horizon=8,
            action_costs=(0.0, 1.0, 2.0, 3.0),
            hidden_dim=16,
            hard_global_budget=True,
        )
    )
    trainer = DSPTrainer(
        policy,
        DSPTrainerConfig(
            total_steps=256,
            rollout_steps=64,
            update_epochs=1,
            minibatch_size=32,
        ),
        device=torch.device("cpu"),
        budget=6.0,
        seed=3,
    )
    result = trainer.train(lambda seed: DiscreteBudgetGymEnv(env_config, 6))
    assert result.update_history
    numeric = [
        value
        for row in result.update_history
        for value in row.values()
        if isinstance(value, float) and not np.isnan(value)
    ]
    assert np.isfinite(numeric).all()
    assert max(result.episode_costs) <= 6
