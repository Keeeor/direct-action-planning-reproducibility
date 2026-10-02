from __future__ import annotations

import numpy as np
import pytest
import torch

from dap.direct_action_planning_dataset_validation.data import TraceDataset
from dap.direct_action_planning_dataset_validation.training import (
    collect_branch_dataset,
)
from dap.envs.synthetic_queue_env import SyntheticQueueConfig
from dap.envs.trace_driven_env import TraceDrivenQueueEnv

from dap.direct_action_planning_dataset_specific_stabilization.planning import (
    make_scaled_planner,
)
from dap.direct_action_planning_dataset_specific_stabilization.selection import (
    SelectionGuardrails,
    select_planning_candidate,
)
from dap.direct_action_planning_dataset_specific_stabilization.training import (
    compute_value_target_scale,
    train_value_candidates,
)
from dap.direct_action_planning_dataset_specific_stabilization.models import (
    ScaledEvidenceValueNetwork,
)
from dap.direct_action_planning_dataset_specific_stabilization.environment import (
    DomainActionCalibration,
    calibrate_domain_actions,
)
from dap.direct_action_planning_dataset_specific_stabilization.baseline_adapter import (
    TrainingConstraintEnv,
    calibrated_baseline_runtime,
)
from dap.direct_action_planning_dataset_specific_stabilization.baseline_experiment import (
    _policy_train_config,
    load_baseline_protocol,
)
from dap.direct_action_planning_dataset_specific_stabilization.experiment import (
    build_checkpoint_payload,
    candidate_grid,
    load_protocol,
)
from dap.direct_action_planning_dataset_specific_stabilization.analysis import (
    assess_dataset_gate,
)


class _ConstantValue:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return np.full(len(observations), self.value, dtype=np.float32)


class _ConstantForecaster:
    def __init__(self, load: float):
        self.load = float(load)

    def predict(self, observation: np.ndarray) -> float:
        del observation
        return self.load


def _dataset() -> TraceDataset:
    trace = np.tile(np.asarray([2.0, 5.0, 9.0, 4.0]), 16)
    return TraceDataset(
        name="test",
        domains={"d": {"train": trace, "validation_fit": trace}},
        split_contract="test-only",
    )


def test_scaled_planner_applies_registered_continuation_weight() -> None:
    env = TraceDrivenQueueEnv(
        np.asarray([5.0, 9.0, 2.0]),
        SyntheticQueueConfig(horizon=3, budget=8.0),
    )
    observation, _ = env.reset(seed=0)
    immediate = make_scaled_planner(
        value=_ConstantValue(10.0),
        forecaster=_ConstantForecaster(7.0),
        gamma=0.99,
        continuation_weight=0.0,
    )
    half = make_scaled_planner(
        value=_ConstantValue(10.0),
        forecaster=_ConstantForecaster(7.0),
        gamma=0.99,
        continuation_weight=0.5,
    )
    _, immediate_q = immediate(env, observation)
    _, half_q = half(env, observation)
    np.testing.assert_allclose(half_q, immediate_q + 0.99 * 0.5 * 10.0)


def test_scaled_planner_rejects_out_of_range_weight() -> None:
    with pytest.raises(ValueError, match="continuation_weight"):
        make_scaled_planner(
            value=_ConstantValue(1.0),
            forecaster=_ConstantForecaster(1.0),
            gamma=0.99,
            continuation_weight=1.1,
        )


def test_scaled_value_network_restores_declared_output_scale() -> None:
    normalizer = __import__(
        "dap.direct_action_planning_dataset_validation.models",
        fromlist=["FeatureNormalizer"],
    ).FeatureNormalizer.fit(np.zeros((2, 14), dtype=np.float32))
    torch.manual_seed(7)
    unit = ScaledEvidenceValueNetwork(normalizer, hidden_dim=8, output_scale=1.0)
    scaled = ScaledEvidenceValueNetwork(normalizer, hidden_dim=8, output_scale=16.0)
    scaled.load_state_dict(unit.state_dict())
    observations = torch.zeros((3, 14), dtype=torch.float32)
    np.testing.assert_allclose(
        scaled(observations).detach().numpy(),
        16.0 * unit(observations).detach().numpy(),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_scaled_value_network_can_start_from_zero_continuation() -> None:
    normalizer = __import__(
        "dap.direct_action_planning_dataset_validation.models",
        fromlist=["FeatureNormalizer"],
    ).FeatureNormalizer.fit(np.zeros((2, 14), dtype=np.float32))
    model = ScaledEvidenceValueNetwork(
        normalizer,
        hidden_dim=8,
        output_scale=32.0,
        zero_initialize_output=True,
    )
    observations = torch.randn((4, 14), generator=torch.Generator().manual_seed(9))
    np.testing.assert_allclose(
        model(observations).detach().numpy(),
        np.zeros(4, dtype=np.float32),
        atol=0.0,
    )


def test_value_target_scale_is_train_only_and_finite() -> None:
    data = _dataset()
    branch = collect_branch_dataset(
        data,
        split="train",
        horizon=8,
        budget=8.0,
        episodes_per_domain=2,
        seed=17,
    )
    scale = compute_value_target_scale(branch, horizon=8)
    assert np.isfinite(scale)
    assert scale >= 1.0


def test_domain_action_calibration_uses_training_quantile_only() -> None:
    training = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    calibration = calibrate_domain_actions(
        training,
        quantile=0.8,
        base_capacity=5.0,
    )
    assert isinstance(calibration, DomainActionCalibration)
    assert calibration.source == "training_only"
    assert calibration.quantile_load == pytest.approx(4.2)
    assert calibration.capacity_multiplier == pytest.approx(1.0)


def test_domain_action_calibration_scales_high_load_domain() -> None:
    calibration = calibrate_domain_actions(
        np.asarray([5.0, 10.0, 20.0, 40.0], dtype=np.float64),
        quantile=0.75,
        base_capacity=5.0,
    )
    assert calibration.capacity_multiplier > 1.0
    assert calibration.action_capacity_deltas[-1] == pytest.approx(
        calibration.capacity_multiplier * 4.8
    )


def test_calibrated_baseline_runtime_is_reversible() -> None:
    from dap.direct_action_planning_dataset_benchmark import rl

    original = rl._factory
    dataset = TraceDataset(
        name="adapter-test",
        domains={
            "d": {
                "train": np.asarray([0.0, 20.0, 40.0, 20.0] * 8),
                "validation_eval": np.asarray([1.0, 2.0, 3.0, 4.0] * 8),
            }
        },
        split_contract="test",
    )
    with calibrated_baseline_runtime(quantile=0.95):
        assert rl._factory is not original
        env = rl._factory(dataset, dataset.domain_names, 8, 8.0, 3)
        assert env.capacity_deltas[-1] > 4.8
    assert rl._factory is original


def test_safety_constraint_env_maps_slo_to_budget_scaled_signal() -> None:
    base = TraceDrivenQueueEnv(
        np.asarray([20.0, 0.0]),
        SyntheticQueueConfig(horizon=2, budget=8.0),
    )
    env = TrainingConstraintEnv(base, slo_constraint_rate=0.25)
    env.reset(seed=0)
    _, _, _, _, info = env.step(3)
    assert info["resource_cost"] == pytest.approx(16.0)
    assert info["actual_resource_cost"] == pytest.approx(4.0)
    assert env.cumulative_cost == pytest.approx(4.0)
    assert env.action_costs[3] == pytest.approx(4.0)


def test_unconstrained_policy_config_has_no_cost_critic() -> None:
    assert _policy_train_config("ppo", 1024, 64).cost_value_coef == pytest.approx(0.0)
    assert _policy_train_config("a2c", 1024, 64).cost_value_coef == pytest.approx(0.0)
    assert _policy_train_config("ppo_lagrangian", 1024, 64).cost_value_coef > 0.0


def test_baseline_protocol_requires_frozen_comparison_families(tmp_path) -> None:
    path = tmp_path / "baseline.yaml"
    path.write_text(
        """tier: test
manifest_schema: test
matrix_row_id: TEST
datasets: [azure2019]
horizon: 4
budgets: [8]
seeds: [1]
seed_role: fixed_repro
gamma: 0.99
training_steps: 8
methods: [ppo]
primary_methods: [ppo]
supplementary_methods: []
evaluation_split: validation_eval
evaluation_episodes_per_domain: 1
capacity_training_quantile: 0.95
slo_constraint_rate: 0.25
evaluation_seed_offset: 3
""",
        encoding="utf-8",
    )
    config = load_baseline_protocol(path)
    assert config["primary_methods"] == ["ppo"]


def test_candidate_selection_applies_guardrails_before_return() -> None:
    rows = [
        {
            "candidate_iteration": -1,
            "continuation_weight": 0.0,
            "discounted_return": 10.0,
            "completion_ratio": 0.90,
            "slo_violation_rate": 0.10,
            "total_cost": 50.0,
        },
        {
            "candidate_iteration": 4,
            "continuation_weight": 1.0,
            "discounted_return": 15.0,
            "completion_ratio": 0.80,
            "slo_violation_rate": 0.10,
            "total_cost": 50.0,
        },
        {
            "candidate_iteration": 2,
            "continuation_weight": 0.5,
            "discounted_return": 12.0,
            "completion_ratio": 0.90,
            "slo_violation_rate": 0.10,
            "total_cost": 52.0,
        },
    ]
    selected, table = select_planning_candidate(
        rows,
        budget=48.0,
        guardrails=SelectionGuardrails(
            completion_tolerance=0.005,
            slo_tolerance=0.01,
            cost_budget_fraction=0.10,
        ),
    )
    assert selected.candidate_iteration == 2
    assert selected.continuation_weight == pytest.approx(0.5)
    assert not bool(
        table.loc[
            (table.candidate_iteration == 4)
            & (table.continuation_weight == 1.0),
            "eligible",
        ].item()
    )


def test_candidate_selection_uses_deterministic_conservative_tie_break() -> None:
    shared = {
        "discounted_return": 11.0,
        "completion_ratio": 0.91,
        "slo_violation_rate": 0.09,
        "total_cost": 49.0,
    }
    rows = [
        {"candidate_iteration": -1, "continuation_weight": 0.0, **shared},
        {"candidate_iteration": 6, "continuation_weight": 0.75, **shared},
        {"candidate_iteration": 2, "continuation_weight": 0.25, **shared},
    ]
    selected, _ = select_planning_candidate(
        rows,
        budget=48.0,
        guardrails=SelectionGuardrails(),
    )
    assert selected.candidate_iteration == -1
    assert selected.continuation_weight == pytest.approx(0.0)


def test_value_training_retains_only_registered_candidate_rounds() -> None:
    dataset = _dataset()
    training = collect_branch_dataset(
        dataset,
        split="train",
        horizon=8,
        budget=8.0,
        episodes_per_domain=2,
        seed=13,
    )
    validation = collect_branch_dataset(
        dataset,
        split="validation_fit",
        horizon=8,
        budget=8.0,
        episodes_per_domain=1,
        seed=14,
    )
    candidates, history = train_value_candidates(
        training,
        validation,
        seed=15,
        gamma=0.99,
        iterations=3,
        candidate_iterations=(0, 2),
        epochs_per_iteration=1,
        learning_rate=1.0e-3,
        hidden_dim=8,
    )
    assert tuple(candidates) == (0, 2)
    assert [row["iteration"] for row in history] == [0.0, 1.0, 2.0]
    left = candidates[0].state_dict()
    right = candidates[2].state_dict()
    assert any(not np.array_equal(left[name].numpy(), right[name].numpy()) for name in left)


@pytest.mark.parametrize(
    "iterations,candidates,message",
    [
        (0, (0,), "iterations"),
        (3, (), "candidate_iterations"),
        (3, (0, 3), "outside"),
        (3, (1, 1), "unique"),
    ],
)
def test_value_candidate_grid_fails_closed(
    iterations: int,
    candidates: tuple[int, ...],
    message: str,
) -> None:
    dataset = _dataset()
    data = collect_branch_dataset(
        dataset,
        split="train",
        horizon=4,
        budget=4.0,
        episodes_per_domain=1,
        seed=3,
    )
    with pytest.raises(ValueError, match=message):
        train_value_candidates(
            data,
            data,
            seed=4,
            gamma=0.99,
            iterations=iterations,
            candidate_iterations=candidates,
            epochs_per_iteration=1,
            learning_rate=1.0e-3,
            hidden_dim=8,
        )


def test_candidate_grid_has_one_immediate_control_and_unique_names() -> None:
    grid = candidate_grid((0, 2), (0.0, 0.5, 1.0))
    assert sum(candidate.continuation_weight == 0.0 for candidate in grid.values()) == 1
    assert len(grid) == 5
    assert len(grid) == len(set(grid))
    assert grid["candidate_i002_l1000"].candidate_iteration == 2


def test_protocol_rejects_candidate_round_outside_training(tmp_path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        "\n".join(
            [
                "tier: bad",
                "manifest_schema: bad",
                "datasets: [gentd26]",
                "horizon: 16",
                "budgets: [8.0]",
                "seeds: [1]",
                "seed_role: fixed_repro",
                "gamma: 0.99",
                "collection_episodes_per_domain: 2",
                "validation_episodes_per_domain: 1",
                "selection_episodes_per_domain: 1",
                "evaluation_episodes_per_domain: 1",
                "fvi_iterations: 3",
                "candidate_iterations: [0, 3]",
                "continuation_weights: [0.0, 1.0]",
                "epochs_per_iteration: 1",
                "model_epochs: 1",
                "hidden_dim: 8",
                "learning_rate: 0.001",
                "model_validation_split: validation_fit",
                "selection_split: validation_select",
                "evaluation_split: validation_eval",
                "selection_guardrails:",
                "  completion_tolerance: 0.005",
                "  slo_tolerance: 0.01",
                "  cost_budget_fraction: 0.10",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside"):
        load_protocol(config)


def test_dataset_gate_requires_return_nonzero_value_and_guardrails() -> None:
    rows = []
    for seed in range(5):
        for budget in (48.0, 96.0, 144.0):
            for method, value in (("dap_immediate", 100.0), ("dap_calibrated", 104.0)):
                rows.append(
                    {
                        "dataset": "test",
                        "budget": budget,
                        "training_seed": seed,
                        "method": method,
                        "discounted_return": value if seed < 4 else 99.0,
                        "completion_ratio": 0.91,
                        "slo_violation_rate": 0.09,
                        "total_cost": 50.0,
                        "selected_continuation_weight": 0.5,
                        "budget_overspend": 0.0,
                    }
                )
    decision = assess_dataset_gate(np.asarray(rows, dtype=object).tolist())
    assert decision["decision"] == "continue"
    assert decision["seed_wins"] == 4
    assert decision["nonzero_cells"] == 15

    for row in rows:
        if row["method"] == "dap_calibrated" and row["budget"] == 48.0:
            row["selected_continuation_weight"] = 0.0
        if row["method"] == "dap_calibrated" and row["budget"] == 96.0:
            row["selected_continuation_weight"] = 0.0
    stopped = assess_dataset_gate(rows)
    assert stopped["decision"] == "stop"
    assert stopped["nonzero_cells"] == 5


def test_checkpoint_payload_is_weights_only_safe(tmp_path) -> None:
    value = torch.nn.Linear(2, 1)
    forecaster = torch.nn.Linear(2, 1)
    payload = build_checkpoint_payload(
        {0: value},
        forecaster,
        normalizer_mean=np.zeros(2),
        normalizer_scale=np.ones(2),
    )
    checkpoint = tmp_path / "models.pt"
    torch.save(payload, checkpoint)
    loaded = torch.load(checkpoint, weights_only=True)
    torch.testing.assert_close(loaded["normalizer_mean"], torch.zeros(2))
