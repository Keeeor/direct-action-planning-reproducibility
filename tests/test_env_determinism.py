import numpy as np

from dap.envs.synthetic_queue_env import (
    DynamicBudgetSchedulingEnv,
    SyntheticQueueConfig,
)


def rollout(seed: int):
    env = DynamicBudgetSchedulingEnv(
        SyntheticQueueConfig(horizon=12, scenario="periodic", budget=20.0)
    )
    obs, _ = env.reset(seed=seed)
    rows = [obs.copy()]
    for action in [0, 1, 2, 3] * 3:
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            np.r_[obs, reward, info["resource_cost"], info["queue_length"]]
        )
        if terminated or truncated:
            break
    return rows


def test_same_seed_and_actions_are_deterministic() -> None:
    first, second = rollout(123), rollout(123)
    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0)


def test_different_seed_changes_poisson_arrivals() -> None:
    assert any(not np.array_equal(a, b) for a, b in zip(rollout(1), rollout(2)))
