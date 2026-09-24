import numpy as np

from stage2_dynamic_budget.envs.trace_driven_env import TraceDrivenQueueEnv
from stage2_dynamic_budget.envs.synthetic_queue_env import SyntheticQueueConfig


def test_observation_depends_only_on_trace_prefix() -> None:
    prefix = [2.0, 3.0, 4.0]
    traces = [prefix + [100.0, 100.0], prefix + [0.0, 0.0]]
    envs = [
        TraceDrivenQueueEnv(np.asarray(trace), SyntheticQueueConfig(horizon=5, budget=10.0))
        for trace in traces
    ]
    observations = [env.reset(seed=9)[0] for env in envs]
    np.testing.assert_allclose(observations[0], observations[1])
    for action in [0, 0]:
        observations = [env.step(action)[0] for env in envs]
        np.testing.assert_allclose(observations[0], observations[1])


def test_time_split_preserves_order() -> None:
    from stage2_dynamic_budget.data.split_by_time import split_by_time

    trace = np.arange(10)
    train, validation, test = split_by_time(trace, 0.6, 0.2)
    assert train.tolist() == list(range(6))
    assert validation.tolist() == [6, 7]
    assert test.tolist() == [8, 9]
