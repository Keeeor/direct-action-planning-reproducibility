import torch

from dap.envs.synthetic_queue_env import DynamicBudgetSchedulingEnv, SyntheticQueueConfig
from dap.evaluation.rollout import evaluate_agent
from dap.models.policy import ConstrainedSchedulingPolicy, PolicyConfig


def test_seeded_stochastic_evaluation_is_reproducible() -> None:
    policy = ConstrainedSchedulingPolicy(
        PolicyConfig(method="budget_state", action_dim=4, hidden_dim=16, episode_budget=16, horizon=16)
    )

    def factory(seed: int):
        return DynamicBudgetSchedulingEnv(SyntheticQueueConfig(horizon=16, budget=16))

    first = evaluate_agent(policy, factory, 16, 9, 2, torch.device("cpu"), deterministic=False)
    second = evaluate_agent(policy, factory, 16, 9, 2, torch.device("cpu"), deterministic=False)
    assert first[0] == second[0]
    comparable_first = [{k: v for k, v in row.items() if k != "decision_latency_ms"} for row in first[1]]
    comparable_second = [{k: v for k, v in row.items() if k != "decision_latency_ms"} for row in second[1]]
    assert comparable_first == comparable_second
