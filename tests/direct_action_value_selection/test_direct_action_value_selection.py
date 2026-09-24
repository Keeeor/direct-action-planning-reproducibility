from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

from stage2_dynamic_budget.action_conditioned_budget_advantage.branching import (
    BranchableDiscreteEnv,
)
from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    solve_action_dp,
)
from stage2_dynamic_budget.direct_action_value_selection.data import (
    generate_k1_branch_data,
    validate_split_integrity,
)
from stage2_dynamic_budget.direct_action_value_selection.analysis import (
    PAIRINGS,
    paired_comparisons,
)
from stage2_dynamic_budget.direct_action_value_selection.gate import assess_continuation
from stage2_dynamic_budget.direct_action_value_selection.model import (
    DAVSAgent,
    DAVSEnsemble,
    fit_davs_model,
)
from stage2_dynamic_budget.direct_action_value_selection.continuous import (
    ContinuousDAVSAgent,
    fit_continuous_davs,
)
from stage2_dynamic_budget.direct_action_value_selection.continuous_data import (
    collect_k1_branch_episode,
    validate_continuous_branch_data,
)
from stage2_dynamic_budget.dynamic_shadow_price.synthetic_joint import (
    AbsoluteBudgetObservationWrapper,
)
from stage2_dynamic_budget.envs.synthetic_queue_env import (
    DynamicBudgetSchedulingEnv,
    SyntheticQueueConfig,
)


def _small_mdp() -> tuple[ActionConditionedBudgetMDP, object]:
    config = ACBADPConfig(
        horizon=8,
        max_budget=3,
        max_queue=2,
        scenario="early_burst",
        gamma=0.93,
    )
    mdp = ActionConditionedBudgetMDP(config)
    return mdp, solve_action_dp(mdp)


def test_k1_dataset_scores_all_feasible_actions_with_common_random_number() -> None:
    mdp, optimum = _small_mdp()
    split = {
        "train": [0, 1],
        "validation": [3],
        "test": [5],
        "embargo": [2, 4, 6, 7],
    }
    frame = generate_k1_branch_data(
        mdp, optimum, "early_burst", seed=7, replications=2, split_by_t=split
    )
    state = frame[(frame.t == 1) & (frame.load == 2) & (frame.queue == 1) &
                  (frame.remaining_budget == 2) & (frame.branch_replication == 0)]
    assert set(state.action) == {0, 1, 2}
    assert state.first_uniform.nunique() == 1
    assert state.random_tape_sha256.nunique() == 1
    assert np.allclose(
        state.sort_values("action").Q_star,
        optimum.q_values[1, 2, 1, 2, :3],
    )
    assert state.groupby("state_id").split.nunique().max() == 1


def test_time_split_has_state_isolation_and_embargo_between_active_splits() -> None:
    mdp, optimum = _small_mdp()
    split = {
        "train": [0, 1],
        "validation": [3],
        "test": [5],
        "embargo": [2, 4, 6, 7],
    }
    frame = generate_k1_branch_data(
        mdp, optimum, "early_burst", seed=3, replications=1, split_by_t=split
    )
    report = validate_split_integrity(frame, split, horizon=8)
    assert report["status"] == "PASS"
    assert report["cross_split_state_count"] == 0
    assert report["adjacent_active_split_count"] == 0


def test_davs_regression_and_rank_select_feasible_actions() -> None:
    mdp, optimum = _small_mdp()
    split = {
        "train": [0, 1, 6, 7],
        "validation": [3],
        "test": [5],
        "embargo": [2, 4],
    }
    frame = generate_k1_branch_data(
        mdp, optimum, "early_burst", seed=11, replications=16, split_by_t=split
    )
    regression = fit_davs_model(mdp, frame, degree=2, ridge=1e-3, rank_beta=0.0)
    ranked = fit_davs_model(mdp, frame, degree=2, ridge=1e-3, rank_beta=1.0)
    for budget in range(mdp.config.max_budget + 1):
        for model in (regression, ranked):
            scores = model.predict_scores(load=1, queue=1, budget=budget, remaining_horizon=3)
            action = int(np.nanargmax(scores))
            assert mdp.action_costs[action] <= budget
            assert np.isnan(scores[mdp.action_costs > budget]).all()


def test_davs_agent_decodes_budget_and_horizon_without_transition_model() -> None:
    mdp, optimum = _small_mdp()
    split = {
        "train": [0, 1, 6, 7],
        "validation": [3],
        "test": [5],
        "embargo": [2, 4],
    }
    frame = generate_k1_branch_data(
        mdp, optimum, "early_burst", seed=13, replications=8, split_by_t=split
    )
    model = fit_davs_model(mdp, frame, degree=2, ridge=1e-3, rank_beta=1.0)
    agent = DAVSAgent(mdp, model)
    env = BranchableDiscreteEnv(mdp.config, initial_budget=3, budget_scale=3)
    observation = env.set_markov_state(t=5, load=1, queue=1, budget=2)
    output = agent.act(torch.tensor(observation).unsqueeze(0))
    assert output.action.shape == (1,)
    assert int(mdp.action_costs[int(output.action.item())]) <= 2
    assert output.q_values.shape == (1, mdp.n_actions)


def test_ensemble_exposes_score_variance_and_ranking_uncertainty() -> None:
    mdp, optimum = _small_mdp()
    split = {
        "train": [0, 1, 6, 7],
        "validation": [3],
        "test": [5],
        "embargo": [2, 4],
    }
    frame = generate_k1_branch_data(
        mdp, optimum, "early_burst", seed=17, replications=8, split_by_t=split
    )
    ensemble = DAVSEnsemble.fit(
        mdp, frame, members=3, seed=17, degree=2, ridge=1e-3, rank_beta=1.0
    )
    prediction = ensemble.predict(load=2, queue=1, budget=3, remaining_horizon=3)
    assert prediction.mean.shape == (mdp.n_actions,)
    assert prediction.variance.shape == (mdp.n_actions,)
    assert np.nanmin(prediction.variance) >= 0.0
    assert 0.0 <= prediction.ranking_uncertainty <= 1.0


def test_continuation_gate_requires_fidelity_loso_and_uncertainty() -> None:
    rows = []
    methods = ["dsp_b", "learned_value_branch", "davs_r", "davs_rank", "davs_ensemble"]
    for method in methods:
        for scenario in ("early_burst", "late_burst", "periodic"):
            for budget in (4, 8, 12):
                for seed in range(5):
                    agreement, regret = 0.80, 0.30
                    if method == "learned_value_branch":
                        agreement, regret = 0.96, 0.04
                    elif method.startswith("davs"):
                        agreement, regret = 0.95, 0.06
                    rows.append(
                        {
                            "method": method,
                            "scenario": scenario,
                            "budget": budget,
                            "seed": seed,
                            "action_consistency_rate": agreement,
                            "mean_Q_star_regret": regret,
                            "completion_rate": 0.90,
                            "slo_violation_rate": 0.10,
                            "total_cost": 2.0,
                        }
                    )
    loso = pd.DataFrame(
        [
            {
                "method": method,
                "heldout_scenario": scenario,
                "action_consistency_rate": 0.82 if method.startswith("davs") else 0.80,
                "mean_Q_star_regret": 0.28 if method.startswith("davs") else 0.30,
            }
            for method in ("dsp_b", "davs_r", "davs_rank", "davs_ensemble")
            for scenario in ("early_burst", "late_burst", "periodic")
        ]
    )
    uncertainty = pd.DataFrame(
        [{"method": "davs_ensemble", "error_auroc": 0.75, "error_spearman": 0.30}]
    )
    gate = assess_continuation(pd.DataFrame(rows), loso, uncertainty, {})
    assert gate["decision"] == "CONTINUE"
    uncertainty.loc[0, "error_auroc"] = 0.50
    stopped = assess_continuation(pd.DataFrame(rows), loso, uncertainty, {})
    assert stopped["decision"] == "STOP"
    assert not stopped["checks"]["ensemble_uncertainty_identifies_errors"]


def test_continuation_gate_handles_empty_loso_smoke_table() -> None:
    rows = []
    for method in ("dsp_b", "learned_value_branch", "davs_r", "davs_rank", "davs_ensemble"):
        rows.append(
            {
                "method": method,
                "scenario": "early_burst",
                "budget": 4,
                "seed": 0,
                "action_consistency_rate": 0.9,
                "mean_Q_star_regret": 0.1,
                "completion_rate": 0.8,
                "slo_violation_rate": 0.2,
                "total_cost": 1.0,
            }
        )
    uncertainty = pd.DataFrame(
        [{"method": "davs_ensemble", "error_auroc": 0.7, "error_spearman": 0.2}]
    )
    gate = assess_continuation(pd.DataFrame(rows), pd.DataFrame(), uncertainty, {})
    assert gate["decision"] == "STOP"
    assert not gate["methods"]["davs_r"]["checks"]["leave_one_scenario_no_reversal"]


def test_paired_analysis_uses_fifteen_budget_seed_cells_and_one_bh_family() -> None:
    rows = []
    methods = sorted({method for pair in PAIRINGS for method in pair})
    metric_names = (
        "action_consistency_rate",
        "mean_Q_star_regret",
        "return_gap_to_paired_optimal",
        "high_risk_low_cost_balanced_accuracy",
        "budget_trajectory_mae",
        "completion_rate",
        "slo_violation_rate",
        "total_cost",
    )
    for method_index, method in enumerate(methods):
        for scenario in ("early_burst", "late_burst", "periodic"):
            for budget in (4, 8, 12):
                for seed in range(5):
                    row = {
                        "method": method,
                        "scenario": scenario,
                        "budget": budget,
                        "seed": seed,
                    }
                    row.update({metric: float(method_index) for metric in metric_names})
                    rows.append(row)
    result = paired_comparisons(pd.DataFrame(rows))
    assert len(result) == len(PAIRINGS) * len(metric_names)
    assert set(result.paired_cells) == {15}
    assert result.bh_q.between(0.0, 1.0).all()


class _ZeroValueReference:
    def reset_budget_controller(self) -> None:
        return None

    def act(self, observation, **kwargs):
        del kwargs
        return SimpleNamespace(
            action=torch.zeros(observation.shape[0], dtype=torch.long),
            reward_value=torch.zeros(observation.shape[0]),
        )


def test_continuous_branches_clone_one_arrival_tape_for_every_action() -> None:
    base = DynamicBudgetSchedulingEnv(
        SyntheticQueueConfig(horizon=6, budget=3.0, scenario="early_burst")
    )
    env = AbsoluteBudgetObservationWrapper(base, budget_scale=3.0)
    frame = collect_k1_branch_episode(
        env,
        _ZeroValueReference(),
        split="train",
        scenario="early_burst",
        budget=3.0,
        budget_scale=3.0,
        seed=0,
        trajectory_seed=101,
        episode=0,
        gamma=0.99,
        action_costs=base.action_costs,
        device=torch.device("cpu"),
    )
    report = validate_continuous_branch_data(frame, base.action_costs)
    assert report["status"] == "PASS"
    assert frame.groupby("state_id").random_tape_sha256.nunique().max() == 1


def test_continuous_direct_scorer_hard_masks_unaffordable_actions() -> None:
    base = DynamicBudgetSchedulingEnv(
        SyntheticQueueConfig(horizon=6, budget=3.0, scenario="stable")
    )
    env = AbsoluteBudgetObservationWrapper(base, budget_scale=3.0)
    frame = collect_k1_branch_episode(
        env,
        _ZeroValueReference(),
        split="train",
        scenario="stable",
        budget=3.0,
        budget_scale=3.0,
        seed=0,
        trajectory_seed=107,
        episode=0,
        gamma=0.99,
        action_costs=base.action_costs,
        device=torch.device("cpu"),
    )
    model = fit_continuous_davs(
        frame,
        base.action_costs,
        budget_scale=3.0,
        ridge=0.01,
        rank_beta=1.0,
        rank_margin=0.1,
    )
    observation = frame.iloc[-1][[f"obs_{index}" for index in range(14)]].to_numpy(float)
    observation[-2] = 1.0 / 3.0
    output = ContinuousDAVSAgent(model).act(
        torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    )
    assert base.action_costs[int(output.action.item())] <= 1.0
    assert torch.isnan(output.q_values[0, 2:]).all()
