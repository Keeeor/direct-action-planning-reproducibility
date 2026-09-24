import numpy as np
import pytest
import torch

from stage2_dynamic_budget.agents.ppo import PPOConfig, PPOTrainer
from stage2_dynamic_budget.envs.synthetic_queue_env import (
    DynamicBudgetSchedulingEnv,
    SyntheticQueueConfig,
)
from stage2_dynamic_budget.models.policy import ConstrainedSchedulingPolicy, PolicyConfig


@pytest.mark.parametrize("method", ["ppo", "lagrangian", "budget_state", "fixed_local", "cdba", "cdba_discrete"])
def test_each_learning_method_completes_a_small_update(method: str) -> None:
    torch.manual_seed(11)
    policy = ConstrainedSchedulingPolicy(
        PolicyConfig(method=method, action_dim=4, hidden_dim=16, episode_budget=32.0, horizon=32)
    )
    before = torch.cat([parameter.detach().flatten().clone() for parameter in policy.parameters()])
    trainer = PPOTrainer(
        policy,
        PPOConfig(
            total_steps=64,
            rollout_steps=64,
            update_epochs=1,
            minibatch_size=32,
        ),
        device=torch.device("cpu"),
        budget=32.0,
        seed=11,
    )

    def factory(seed: int):
        return DynamicBudgetSchedulingEnv(
            SyntheticQueueConfig(horizon=32, budget=32.0, scenario="late_burst")
        )

    result = trainer.train(factory)
    after = torch.cat([parameter.detach().flatten() for parameter in policy.parameters()])
    assert not torch.equal(before, after)
    assert len(result.episode_costs) == 2
    assert result.update_history
    assert all(np.isfinite(row["policy_loss"]) for row in result.update_history)
