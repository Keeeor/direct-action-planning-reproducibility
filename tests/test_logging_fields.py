from dap.envs.synthetic_queue_env import (
    DynamicBudgetSchedulingEnv,
    SyntheticQueueConfig,
)


REQUIRED = {
    "resource_cost",
    "cumulative_cost",
    "remaining_budget",
    "remaining_budget_ratio",
    "remaining_horizon_ratio",
    "slo_violation",
    "queue_length",
    "tail_latency",
    "load_level",
    "risk_level",
    "action_name",
}


def test_step_info_has_required_fields() -> None:
    env = DynamicBudgetSchedulingEnv(SyntheticQueueConfig(horizon=2))
    env.reset(seed=0, options={"arrival_trace": [5.0, 6.0]})
    _, _, _, _, info = env.step(1)
    assert REQUIRED <= info.keys()
