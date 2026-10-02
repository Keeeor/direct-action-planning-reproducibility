from dap.envs.synthetic_queue_env import (
    ActionSpec,
    DynamicBudgetSchedulingEnv,
    SyntheticQueueConfig,
)


def test_action_costs_match_configuration() -> None:
    actions = (
        ActionSpec("no_op", 0.0, 0.0),
        ActionSpec("small", 1.0, 1.0),
        ActionSpec("medium", 2.0, 2.2),
        ActionSpec("large", 4.0, 4.8),
    )
    env = DynamicBudgetSchedulingEnv(SyntheticQueueConfig(actions=actions))
    assert env.action_costs.tolist() == [0.0, 1.0, 2.0, 4.0]
    assert env.capacity_deltas.tolist() == [0.0, 1.0, 2.2, 4.8]


def test_alternative_cost_benefit_curve_preserves_action_space() -> None:
    actions = (
        ActionSpec("no_op", 0.0, 0.0),
        ActionSpec("small", 1.0, 0.8),
        ActionSpec("medium", 2.0, 1.7),
        ActionSpec("large", 4.0, 3.4),
    )
    env = DynamicBudgetSchedulingEnv(SyntheticQueueConfig(actions=actions))
    assert env.action_space.n == 4
    assert env.action_costs.tolist() == [0.0, 1.0, 2.0, 4.0]
    assert env.capacity_deltas.tolist() == [0.0, 0.8, 1.7, 3.4]
