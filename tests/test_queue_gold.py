import pytest

from dap.envs.synthetic_queue_env import (
    DynamicBudgetSchedulingEnv,
    SyntheticQueueConfig,
)


def test_queue_update_matches_hand_calculation() -> None:
    config = SyntheticQueueConfig(horizon=2, base_capacity=5.0, budget=10.0)
    env = DynamicBudgetSchedulingEnv(config)
    env.reset(seed=0, options={"arrival_trace": [8.0, 1.0]})
    _, _, _, _, first = env.step(0)
    assert first["arrivals"] == pytest.approx(8.0)
    assert first["served"] == pytest.approx(5.0)
    assert first["queue_length"] == pytest.approx(3.0)
    _, _, _, _, second = env.step(1)
    assert second["served"] == pytest.approx(4.0)
    assert second["queue_length"] == pytest.approx(0.0)


def test_budget_is_metamorphic_for_queue_dynamics() -> None:
    configs = [SyntheticQueueConfig(horizon=3, budget=b) for b in (5.0, 50.0)]
    envs = [DynamicBudgetSchedulingEnv(config) for config in configs]
    for env in envs:
        env.reset(seed=3, options={"arrival_trace": [8.0, 8.0, 8.0]})
    for action in [1, 0, 2]:
        transitions = [env.step(action) for env in envs]
        infos = [transition[4] for transition in transitions]
        assert infos[0]["queue_length"] == pytest.approx(infos[1]["queue_length"])
        # Base-state baselines use the first 12 fields; budget may only affect
        # the two explicitly registered budget-state fields.
        assert transitions[0][0][:-2] == pytest.approx(transitions[1][0][:-2])
