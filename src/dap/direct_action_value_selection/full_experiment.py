from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import torch

from dap.data.trace_windows import select_trace_window
from dap.dynamic_shadow_price.synthetic_experiment import (
    _build_agent,
    _make_env,
)
from dap.dynamic_shadow_price.trace_experiment import (
    _load_domains,
    _trace_env,
)
from dap.evaluation.metrics import summarize_episode
from dap.experiment import load_config
from dap.utils.artifacts import (
    environment_record,
    sha256_file,
    sha256_tree,
    write_json,
)
from dap.utils.seed import set_global_seed

from .continuous import (
    ContinuousDAVSAgent,
    ContinuousDAVSEnsemble,
    ContinuousDAVSModel,
    OBS_COLUMNS,
    fit_continuous_davs,
)
from .continuous_data import collect_k1_branch_episode, validate_continuous_branch_data


METHODS = ("davs_r", "davs_rank", "davs_ensemble")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolved_config(config: dict, run_id: str, smoke: bool) -> dict:
    resolved = json.loads(json.dumps(config))
    resolved["run_id"] = run_id
    resolved["smoke"] = bool(smoke)
    resolved["minimal_gate_source"] = "minimal_v1/continuation_gate.json:CONTINUE"
    if smoke:
        resolved["budgets"] = [110.0]
        resolved["seeds"] = [0]
        resolved["train_scenarios"] = ["stable"]
        resolved["eval_scenarios"] = ["stable", "ood_burst"]
        resolved["evaluation_episodes"] = 1
        resolved["branch_data"]["synthetic_episodes_per_scenario_budget"] = {
            "train": 1,
            "validation": 1,
            "test": 1,
        }
        resolved["branch_data"]["trace_windows_per_domain_budget"] = {
            "train": 1,
            "validation": 1,
            "test": 1,
        }
        resolved["models"]["ensemble_members"] = 3
        resolved["run_sensitivity"] = False
    else:
        resolved["run_sensitivity"] = True
    return resolved


def _load_frozen_agent(
    project_root: Path,
    source_config: dict,
    domain: str,
    method: str,
    seed: int,
    device: torch.device,
) -> tuple[object, Path]:
    budget_scale = max(float(value) for value in source_config["budgets"])
    agent, _, checkpoint_agent = _build_agent(
        source_config, method, budget_scale, seed, device
    )
    prefix = "synthetic_joint" if domain == "synthetic" else "trace_joint"
    checkpoint = (
        project_root
        / f"results/dynamic_shadow_price/{domain}"
        / f"{prefix}__formal__{method}__s{seed}"
        / "model.pt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_agent.load_state_dict(payload["state_dict"])
    agent.to(device)
    agent.eval()
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    return agent, checkpoint


def _save_model(path: Path, model: ContinuousDAVSModel) -> None:
    np.savez_compressed(
        path,
        coefficients=model.coefficients,
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
        target_mean=np.asarray(model.target_mean),
        target_scale=np.asarray(model.target_scale),
        action_costs=model.action_costs,
        budget_scale=np.asarray(model.budget_scale),
        ridge=np.asarray(model.ridge),
        rank_beta=np.asarray(model.rank_beta),
        training_states=np.asarray(model.training_states),
    )


def _fit_formal_models(
    frame: pd.DataFrame, config: dict, action_costs: np.ndarray, seed: int
) -> dict[str, ContinuousDAVSModel | ContinuousDAVSEnsemble]:
    model = config["models"]
    common = {
        "action_costs": action_costs,
        "budget_scale": max(float(value) for value in config["budgets"]),
        "ridge": float(model["ridge"]),
        "rank_margin": float(model["rank_margin"]),
    }
    regression = fit_continuous_davs(
        frame, rank_beta=0.0, seed=seed + 11, **common
    )
    ranked = fit_continuous_davs(
        frame, rank_beta=float(model["rank_beta"]), seed=seed + 23, **common
    )
    ensemble = ContinuousDAVSEnsemble.fit(
        frame,
        rank_beta=float(model["rank_beta"]),
        members=int(model["ensemble_members"]),
        seed=seed + 37,
        **common,
    )
    return {"davs_r": regression, "davs_rank": ranked, "davs_ensemble": ensemble}


def _save_models(
    directory: Path,
    prefix: str,
    models: dict[str, ContinuousDAVSModel | ContinuousDAVSEnsemble],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for method, scorer in models.items():
        if isinstance(scorer, ContinuousDAVSEnsemble):
            for index, member in enumerate(scorer.models):
                _save_model(directory / f"{prefix}__{method}__m{index}.npz", member)
        else:
            _save_model(directory / f"{prefix}__{method}.npz", scorer)


def _synthetic_branch_data(
    reference: object,
    source_config: dict,
    config: dict,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    budgets = [float(value) for value in config["budgets"]]
    budget_scale = max(budgets)
    action_costs = np.asarray(source_config["environment"]["action_costs"], dtype=float)
    counts = config["branch_data"]["synthetic_episodes_per_scenario_budget"]
    rows: list[pd.DataFrame] = []
    split_band = {"train": 1_000_003, "validation": 2_000_003, "test": 3_000_003}
    for split in ("train", "validation", "test"):
        scenarios = (
            config["train_scenarios"] if split == "train" else config["eval_scenarios"]
        )
        for scenario_index, scenario in enumerate(scenarios):
            for budget_index, budget in enumerate(budgets):
                for episode in range(int(counts[split])):
                    trajectory_seed = (
                        seed * 10_000_019
                        + split_band[split]
                        + scenario_index * 100_003
                        + budget_index * 10_007
                        + episode * 101
                    )
                    torch.manual_seed(trajectory_seed)
                    env = _make_env(source_config, scenario, budget, budget_scale)
                    rows.append(
                        collect_k1_branch_episode(
                            env,
                            reference,
                            split=split,
                            scenario=scenario,
                            budget=budget,
                            budget_scale=budget_scale,
                            seed=seed,
                            trajectory_seed=trajectory_seed,
                            episode=episode,
                            gamma=float(config["gamma"]),
                            action_costs=action_costs,
                            device=device,
                            extra_metadata={"domain": "synthetic"},
                        )
                    )
    return pd.concat(rows, ignore_index=True)


def _trace_branch_data(
    reference: object,
    source_config: dict,
    config: dict,
    domains: dict[str, dict[str, np.ndarray]],
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    budgets = [float(value) for value in config["budgets"]]
    budget_scale = max(budgets)
    action_costs = np.asarray(source_config["environment"]["action_costs"], dtype=float)
    counts = config["branch_data"]["trace_windows_per_domain_budget"]
    split_band = {"train": 4_000_037, "validation": 5_000_041, "test": 6_000_043}
    horizon = int(source_config["environment"]["horizon"])
    rows: list[pd.DataFrame] = []
    for split in ("train", "validation", "test"):
        for domain_index, domain in enumerate(source_config["trace"]["domains"]):
            for budget_index, budget in enumerate(budgets):
                for episode in range(int(counts[split])):
                    trajectory_seed = (
                        seed * 10_000_019
                        + split_band[split]
                        + domain_index * 100_003
                        + budget_index * 10_007
                        + episode * 101
                    )
                    trace, start = select_trace_window(
                        domains[domain][split], horizon, trajectory_seed
                    )
                    scenario = f"azure_{domain}_{split}"
                    torch.manual_seed(trajectory_seed)
                    env = _trace_env(
                        source_config, trace, budget, budget_scale, scenario
                    )
                    rows.append(
                        collect_k1_branch_episode(
                            env,
                            reference,
                            split=split,
                            scenario=scenario,
                            budget=budget,
                            budget_scale=budget_scale,
                            seed=seed,
                            trajectory_seed=trajectory_seed,
                            episode=episode,
                            gamma=float(config["gamma"]),
                            action_costs=action_costs,
                            device=device,
                            extra_metadata={
                                "domain": domain,
                                "trace_start": int(start),
                            },
                        )
                    )
    return pd.concat(rows, ignore_index=True)


def _risk_action_metrics(step_rows: list[dict[str, object]]) -> dict[str, float]:
    risk = np.asarray([row["risk_level"] for row in step_rows], dtype=float)
    cost = np.asarray([row["resource_cost"] for row in step_rows], dtype=float)
    low, high = np.quantile(risk, [0.25, 0.75])
    if np.std(risk) > 1e-12 and np.std(cost) > 1e-12:
        correlation = float(spearmanr(risk, cost).statistic)
    else:
        correlation = np.nan
    low_cost = float(cost[risk <= low].mean())
    high_cost = float(cost[risk >= high].mean())
    return {
        "risk_action_cost_correlation": correlation,
        "low_risk_action_cost": low_cost,
        "high_risk_action_cost": high_cost,
        "action_reallocation_difference": high_cost - low_cost,
    }


def _evaluate_synthetic_davs(
    models: dict[str, ContinuousDAVSModel | ContinuousDAVSEnsemble],
    source_config: dict,
    config: dict,
    seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    budgets = [float(value) for value in config["budgets"]]
    budget_scale = max(budgets)
    episodes: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    for method, scorer in models.items():
        agent = ContinuousDAVSAgent(scorer)
        for budget in budgets:
            for scenario in config["eval_scenarios"]:
                latencies: list[float] = []
                for episode in range(int(config["evaluation_episodes"])):
                    episode_seed = seed + 500_009 + episode * 100_003
                    env = _make_env(source_config, scenario, budget, budget_scale)
                    observation, _ = env.reset(seed=episode_seed)
                    current: list[dict[str, object]] = []
                    uncertainties: list[float] = []
                    while True:
                        started = time.perf_counter_ns()
                        output = agent.act(
                            torch.as_tensor(
                                observation, dtype=torch.float32, device=device
                            ).unsqueeze(0)
                        )
                        latencies.append((time.perf_counter_ns() - started) / 1e6)
                        next_observation, reward, terminated, truncated, info = env.step(
                            int(output.action.item())
                        )
                        current.append({**info, "reward": reward, "episode": episode})
                        uncertainties.append(float(output.ranking_uncertainty.item()))
                        observation = next_observation
                        if terminated or truncated:
                            break
                    summary = summarize_episode(
                        current, budget, int(source_config["environment"]["horizon"])
                    )
                    summary.update(
                        {
                            "scenario": scenario,
                            "budget": budget,
                            "eval_episode": episode,
                            "eval_seed": episode_seed,
                            "method": method,
                            "seed": seed,
                            "variant": "formal",
                            "ranking_uncertainty_mean": float(np.mean(uncertainties)),
                            **_risk_action_metrics(current),
                        }
                    )
                    episodes.append(summary)
                runtime.append(
                    {
                        "domain": "synthetic",
                        "method": method,
                        "seed": seed,
                        "scenario": scenario,
                        "budget": budget,
                        "decision_latency_ms_mean": float(np.mean(latencies)),
                        "decision_latency_ms_p95": float(np.quantile(latencies, 0.95)),
                    }
                )
    return pd.DataFrame(episodes), pd.DataFrame(runtime)


def _evaluate_trace_davs(
    models: dict[str, ContinuousDAVSModel | ContinuousDAVSEnsemble],
    source_config: dict,
    config: dict,
    domains: dict[str, dict[str, np.ndarray]],
    seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    budgets = [float(value) for value in config["budgets"]]
    budget_scale = max(budgets)
    horizon = int(source_config["environment"]["horizon"])
    episodes: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    for method, scorer in models.items():
        agent = ContinuousDAVSAgent(scorer)
        for budget in budgets:
            for domain in source_config["trace"]["domains"]:
                latencies: list[float] = []
                scenario = f"azure_{domain}_test"
                for episode in range(int(config["evaluation_episodes"])):
                    episode_seed = seed + 700_001 + episode * 100_003
                    trace, start = select_trace_window(
                        domains[domain]["test"], horizon, episode_seed
                    )
                    env = _trace_env(
                        source_config, trace, budget, budget_scale, scenario
                    )
                    observation, _ = env.reset(seed=episode_seed)
                    current: list[dict[str, object]] = []
                    uncertainties: list[float] = []
                    while True:
                        started = time.perf_counter_ns()
                        output = agent.act(
                            torch.as_tensor(
                                observation, dtype=torch.float32, device=device
                            ).unsqueeze(0)
                        )
                        latencies.append((time.perf_counter_ns() - started) / 1e6)
                        next_observation, reward, terminated, truncated, info = env.step(
                            int(output.action.item())
                        )
                        current.append({**info, "reward": reward, "episode": episode})
                        uncertainties.append(float(output.ranking_uncertainty.item()))
                        observation = next_observation
                        if terminated or truncated:
                            break
                    summary = summarize_episode(current, budget, horizon)
                    summary.update(
                        {
                            "domain": domain,
                            "split": "test",
                            "scenario": scenario,
                            "budget": budget,
                            "eval_episode": episode,
                            "eval_seed": episode_seed,
                            "trace_start": start,
                            "method": method,
                            "seed": seed,
                            "variant": "formal",
                            "ranking_uncertainty_mean": float(np.mean(uncertainties)),
                            **_risk_action_metrics(current),
                        }
                    )
                    episodes.append(summary)
                runtime.append(
                    {
                        "domain": "public_trace",
                        "method": method,
                        "seed": seed,
                        "scenario": scenario,
                        "budget": budget,
                        "decision_latency_ms_mean": float(np.mean(latencies)),
                        "decision_latency_ms_p95": float(np.quantile(latencies, 0.95)),
                    }
                )
    return pd.DataFrame(episodes), pd.DataFrame(runtime)


def _error_auroc(error: np.ndarray, uncertainty: np.ndarray) -> float:
    error = np.asarray(error, dtype=bool)
    positives = int(error.sum())
    negatives = int((~error).sum())
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = rankdata(np.asarray(uncertainty, dtype=float), method="average")
    statistic = float(ranks[error].sum() - positives * (positives + 1) / 2.0)
    return statistic / (positives * negatives)


def _branch_diagnostics(
    frame: pd.DataFrame,
    models: dict[str, ContinuousDAVSModel | ContinuousDAVSEnsemble],
    small_gap: float,
    *,
    split: str = "test",
    variant: str = "formal",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = frame[frame.split == split]
    state_rows: list[dict[str, object]] = []
    for method, scorer in models.items():
        for state_id, group in test.groupby("state_id", sort=False):
            observation = group.iloc[0].loc[list(OBS_COLUMNS)].to_numpy(dtype=float)
            if isinstance(scorer, ContinuousDAVSEnsemble):
                prediction = scorer.predict(observation)
                scores = prediction.mean
                uncertainty = prediction.ranking_uncertainty
                score_variance = float(np.nanmean(prediction.variance))
            else:
                scores = scorer.predict_all(observation)
                uncertainty = 0.0
                score_variance = 0.0
            actions = group.action.to_numpy(dtype=int)
            truth = group.Q_branch.to_numpy(dtype=float)
            predicted = scores[actions]
            best_index = int(np.argmax(truth))
            selected_action = int(np.nanargmax(scores))
            selected_match = np.flatnonzero(actions == selected_action)
            if not len(selected_match):
                raise RuntimeError("DAVS selected an action absent from the feasible group")
            selected_value = float(truth[int(selected_match[0])])
            pair_correct: list[float] = []
            for left in range(len(actions)):
                for right in range(left + 1, len(actions)):
                    true_gap = float(truth[left] - truth[right])
                    if abs(true_gap) <= 1e-12:
                        continue
                    pair_correct.append(
                        float(
                            np.sign(predicted[left] - predicted[right])
                            == np.sign(true_gap)
                        )
                    )
            ordered = np.sort(truth)
            gap = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else np.inf
            state_rows.append(
                {
                    "variant": variant,
                    "method": method,
                    "state_id": state_id,
                    "scenario": group.scenario.iloc[0],
                    "domain": group.domain.iloc[0],
                    "budget": float(group.budget.iloc[0]),
                    "seed": int(group.seed.iloc[0]),
                    "action_value_mae": float(np.mean(np.abs(predicted - truth))),
                    "pairwise_action_ranking_accuracy": (
                        float(np.mean(pair_correct)) if pair_correct else 1.0
                    ),
                    "action_consistency_rate": float(
                        selected_action == int(actions[best_index])
                    ),
                    "Q_branch_regret": float(np.max(truth) - selected_value),
                    "top1_top2_gap": gap,
                    "small_gap": bool(gap <= small_gap),
                    "error": float(selected_action != int(actions[best_index])),
                    "ranking_uncertainty": uncertainty,
                    "ensemble_score_variance_mean": score_variance,
                }
            )
    states = pd.DataFrame(state_rows)
    summaries: list[dict[str, object]] = []
    for keys, group in states.groupby(
        ["variant", "method", "domain", "scenario", "budget", "seed"],
        sort=False,
    ):
        variant_value, method, domain, scenario, budget, seed = keys
        error = group.error.to_numpy(dtype=float)
        uncertainty = group.ranking_uncertainty.to_numpy(dtype=float)
        correlation = (
            float(spearmanr(uncertainty, error).statistic)
            if np.std(uncertainty) > 1e-12 and np.std(error) > 1e-12
            else np.nan
        )
        small = group[group.small_gap]
        summaries.append(
            {
                "variant": variant_value,
                "method": method,
                "domain": domain,
                "scenario": scenario,
                "budget": budget,
                "seed": seed,
                "test_states": int(len(group)),
                "action_value_mae": float(group.action_value_mae.mean()),
                "pairwise_action_ranking_accuracy": float(
                    group.pairwise_action_ranking_accuracy.mean()
                ),
                "action_consistency_rate": float(group.action_consistency_rate.mean()),
                "mean_Q_branch_regret": float(group.Q_branch_regret.mean()),
                "small_gap_states": int(len(small)),
                "small_gap_error_rate": float(small.error.mean()) if len(small) else np.nan,
                "all_state_error_rate": float(group.error.mean()),
                "ensemble_score_variance_mean": float(
                    group.ensemble_score_variance_mean.mean()
                ),
                "error_auroc": _error_auroc(error, uncertainty),
                "error_spearman": correlation,
            }
        )
    return pd.DataFrame(summaries), states


def _sensitivity_diagnostics(
    frame: pd.DataFrame,
    config: dict,
    action_costs: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    if not config.get("run_sensitivity", True):
        return pd.DataFrame()
    model = config["models"]
    budget_scale = max(float(value) for value in config["budgets"])
    cases: list[tuple[str, str, object]] = []
    common = {
        "action_costs": action_costs,
        "budget_scale": budget_scale,
        "ridge": float(model["ridge"]),
        "rank_margin": float(model["rank_margin"]),
    }
    for beta in config["ablations"]["rank_beta"]:
        scorer = fit_continuous_davs(
            frame,
            rank_beta=float(beta),
            seed=seed + int(float(beta) * 1000) + 101,
            **common,
        )
        cases.append(("rank_beta", f"beta={float(beta):.3g}", scorer))
    for members in config["ablations"]["ensemble_members"]:
        members = int(members)
        if members == 1:
            scorer = fit_continuous_davs(
                frame,
                rank_beta=float(model["rank_beta"]),
                seed=seed + 211,
                **common,
            )
        else:
            scorer = ContinuousDAVSEnsemble.fit(
                frame,
                rank_beta=float(model["rank_beta"]),
                members=members,
                seed=seed + 211 + members,
                **common,
            )
        cases.append(("ensemble_members", f"members={members}", scorer))
    for fraction in config["data_scale_fractions"]:
        scorer = fit_continuous_davs(
            frame,
            rank_beta=float(model["rank_beta"]),
            data_fraction=float(fraction),
            seed=seed + 307,
            **common,
        )
        cases.append(("data_scale", f"fraction={float(fraction):.3g}", scorer))
    for noise in config["label_noise_standard_deviations"]:
        scorer = fit_continuous_davs(
            frame,
            rank_beta=float(model["rank_beta"]),
            label_noise_std=float(noise),
            seed=seed + 401,
            **common,
        )
        cases.append(("label_noise", f"std={float(noise):.3g}", scorer))
    outputs: list[pd.DataFrame] = []
    for family, setting, scorer in cases:
        diagnostics, _ = _branch_diagnostics(
            frame,
            {setting: scorer},
            float(model["small_q_gap"]),
            variant=family,
        )
        diagnostics["setting"] = setting
        outputs.append(diagnostics)
    return pd.concat(outputs, ignore_index=True)


def _loso_diagnostics(
    frame: pd.DataFrame,
    config: dict,
    action_costs: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    if not config.get("run_sensitivity", True):
        return pd.DataFrame()
    model = config["models"]
    outputs: list[pd.DataFrame] = []
    for index, heldout in enumerate(config["train_scenarios"]):
        amended = frame[
            ~((frame.split == "train") & (frame.scenario == heldout))
        ].copy()
        scorer = fit_continuous_davs(
            amended,
            action_costs,
            max(float(value) for value in config["budgets"]),
            float(model["ridge"]),
            float(model["rank_beta"]),
            float(model["rank_margin"]),
            seed=seed + 10_003 + index,
        )
        heldout_frame = frame[
            (frame.split != "test") | (frame.scenario == heldout)
        ].copy()
        diagnostics, _ = _branch_diagnostics(
            heldout_frame,
            {"davs_rank_loso": scorer},
            float(model["small_q_gap"]),
            variant="leave_one_scenario_out",
        )
        diagnostics["heldout_scenario"] = heldout
        outputs.append(diagnostics)
    return pd.concat(outputs, ignore_index=True)


def _frozen_baselines(
    project_root: Path, domain: str, seeds: list[int], smoke: bool
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[pd.DataFrame] = []
    lineage: dict[str, str] = {}
    methods = ("b4_joint_hard", "dsp_b")
    prefix = "synthetic_joint" if domain == "synthetic" else "trace_joint"
    for seed in seeds:
        for method in methods:
            directory = (
                project_root
                / f"results/dynamic_shadow_price/{domain}"
                / f"{prefix}__formal__{method}__s{seed}"
            )
            metrics_path = directory / "metrics.csv"
            manifest_path = directory / "manifest.json"
            frame = pd.read_csv(metrics_path)
            if domain == "trace":
                frame = frame[frame.split == "test"].copy()
            if smoke:
                first_budget = float(frame.budget.min())
                frame = frame[
                    (frame.budget == first_budget) & (frame.eval_episode == 0)
                ].copy()
                if domain == "synthetic":
                    frame = frame[frame.scenario.isin(["stable", "ood_burst"])]
                else:
                    frame = frame[frame.domain.isin(["http", "async"])]
            rows.append(frame)
            for path in (metrics_path, manifest_path, directory / "model.pt"):
                relative = str(path.relative_to(project_root))
                lineage[relative] = sha256_file(path)
    return pd.concat(rows, ignore_index=True), lineage


def _pareto_and_stop_gate(
    synthetic: pd.DataFrame,
    public: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict[str, object]]:
    thresholds = config["full_stop_gate"]
    candidates = METHODS
    pareto_rows: list[dict[str, object]] = []
    method_checks: dict[str, object] = {}
    for method in candidates:
        domain_checks: dict[str, object] = {}
        for domain, frame, keys in (
            ("synthetic", synthetic, ["scenario", "budget", "seed"]),
            ("public_trace", public, ["domain", "budget", "seed"]),
        ):
            cells = frame.groupby(["method", *keys], as_index=False)[
                ["episode_reward", "completion_rate", "slo_violation_rate", "total_cost"]
            ].mean()
            candidate = cells[cells.method == method].set_index(keys)
            baseline = cells[cells.method == "dsp_b"].set_index(keys)
            paired = candidate.join(
                baseline, lsuffix="_candidate", rsuffix="_baseline", how="inner"
            )
            completion_loss = (
                paired.completion_rate_baseline - paired.completion_rate_candidate
            )
            slo_increase = (
                paired.slo_violation_rate_candidate - paired.slo_violation_rate_baseline
            )
            reward_difference = paired.episode_reward_candidate - paired.episode_reward_baseline
            cost_difference = paired.total_cost_candidate - paired.total_cost_baseline
            non_dominated = (
                completion_loss <= float(thresholds["maximum_mean_completion_loss"])
            ) & (slo_increase <= float(thresholds["maximum_mean_slo_increase"])) & (
                (reward_difference > 0) | (cost_difference < 0)
            )
            for index, value in enumerate(non_dominated):
                key_values = paired.index[index]
                if not isinstance(key_values, tuple):
                    key_values = (key_values,)
                pareto_rows.append(
                    {
                        "domain": domain,
                        "method": method,
                        **dict(zip(keys, key_values)),
                        "completion_difference": float(-completion_loss.iloc[index]),
                        "slo_difference": float(slo_increase.iloc[index]),
                        "reward_difference": float(reward_difference.iloc[index]),
                        "cost_difference": float(cost_difference.iloc[index]),
                        "non_dominated_vs_dsp_b": bool(value),
                    }
                )
            domain_pass = bool(
                float(non_dominated.mean())
                >= float(
                    thresholds["minimum_non_dominated_budget_scenario_seed_fraction"]
                )
                and float(completion_loss.mean())
                <= float(thresholds["maximum_mean_completion_loss"])
                and float(slo_increase.mean())
                <= float(thresholds["maximum_mean_slo_increase"])
                and (
                    not bool(thresholds["require_positive_reward_difference"])
                    or float(reward_difference.mean()) > 0
                )
            )
            domain_checks[domain] = {
                "passes": domain_pass,
                "paired_cells": int(len(paired)),
                "non_dominated_fraction": float(non_dominated.mean()),
                "mean_completion_difference": float(-completion_loss.mean()),
                "mean_slo_difference": float(slo_increase.mean()),
                "mean_reward_difference": float(reward_difference.mean()),
                "mean_cost_difference": float(cost_difference.mean()),
            }
        method_checks[method] = {
            "passes": all(value["passes"] for value in domain_checks.values()),
            "domains": domain_checks,
        }
    pass_methods = [
        method for method, result in method_checks.items() if bool(result["passes"])
    ]
    return pd.DataFrame(pareto_rows), {
        "schema": "direct_action_value_selection.full_stop_gate.v1",
        "decision": "COMPLETE_POSITIVE" if pass_methods else "STOP",
        "thresholds": thresholds,
        "passing_methods": pass_methods,
        "methods": method_checks,
        "stop_condition_triggered": not bool(pass_methods),
    }


def run_full_validation(
    project_root: str | Path,
    config_path: str | Path,
    run_id: str = "full_v1",
    smoke: bool = False,
) -> Path:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = _resolved_config(load_config(config_path), run_id, smoke)
    run_dir = project_root / "results/direct_action_value_selection" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "completed":
                return run_dir
        raise RuntimeError(f"run directory already contains an incomplete run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "branch_data").mkdir()
    (run_dir / "models").mkdir()
    started_at = _utcnow()
    started_clock = time.perf_counter()
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment_record())
    write_json(run_dir / "stage_status.json", {"stage": "starting", "at": _utcnow()})
    (run_dir / "stdout.log").write_text("", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    set_global_seed(0, torch_threads=int(config.get("torch_threads", 1)))
    device = torch.device(str(config.get("device", "cpu")))
    synthetic_source = load_config(project_root / config["synthetic_source_config"])
    trace_source = load_config(project_root / config["trace_source_config"])
    trace_domains = _load_domains(project_root, trace_source)
    seeds = [int(value) for value in config["seeds"]]
    action_costs = np.asarray(
        synthetic_source["environment"]["action_costs"], dtype=float
    )
    try:
        synth_baseline, synth_lineage = _frozen_baselines(
            project_root, "synthetic", seeds, smoke
        )
        trace_baseline, trace_lineage = _frozen_baselines(
            project_root, "trace", seeds, smoke
        )
        lineage = {**synth_lineage, **trace_lineage}
        synth_metrics: list[pd.DataFrame] = [synth_baseline]
        trace_metrics: list[pd.DataFrame] = [trace_baseline]
        runtimes: list[pd.DataFrame] = []
        diagnostics: list[pd.DataFrame] = []
        uncertainty_states: list[pd.DataFrame] = []
        sensitivities: list[pd.DataFrame] = []
        loso_rows: list[pd.DataFrame] = []
        split_reports: list[dict[str, object]] = []
        for seed in seeds:
            write_json(
                run_dir / "stage_status.json",
                {"stage": "synthetic_branch", "seed": seed, "at": _utcnow()},
            )
            b4_synth, checkpoint = _load_frozen_agent(
                project_root, synthetic_source, "synthetic", "b4_joint_hard", seed, device
            )
            lineage[str(checkpoint.relative_to(project_root))] = sha256_file(checkpoint)
            synth_frame = _synthetic_branch_data(
                b4_synth, synthetic_source, config, seed, device
            )
            report = validate_continuous_branch_data(synth_frame, action_costs)
            report.update({"domain": "synthetic", "seed": seed})
            if report["status"] != "PASS":
                raise RuntimeError(f"synthetic branch integrity failed for seed {seed}")
            split_reports.append(report)
            synth_frame.to_csv(
                run_dir / "branch_data" / f"synthetic__s{seed}.csv.gz",
                index=False,
                compression="gzip",
            )
            synth_models = _fit_formal_models(synth_frame, config, action_costs, seed)
            _save_models(run_dir / "models", f"synthetic__s{seed}", synth_models)
            diag, states = _branch_diagnostics(
                synth_frame,
                synth_models,
                float(config["models"]["small_q_gap"]),
            )
            diagnostics.append(diag)
            uncertainty_states.append(states)
            sensitivities.append(
                _sensitivity_diagnostics(synth_frame, config, action_costs, seed)
            )
            loso_rows.append(_loso_diagnostics(synth_frame, config, action_costs, seed))
            synth_eval, synth_runtime = _evaluate_synthetic_davs(
                synth_models, synthetic_source, config, seed, device
            )
            synth_metrics.append(synth_eval)
            runtimes.append(synth_runtime)

            write_json(
                run_dir / "stage_status.json",
                {"stage": "trace_branch", "seed": seed, "at": _utcnow()},
            )
            b4_trace, checkpoint = _load_frozen_agent(
                project_root, trace_source, "trace", "b4_joint_hard", seed, device
            )
            lineage[str(checkpoint.relative_to(project_root))] = sha256_file(checkpoint)
            trace_frame = _trace_branch_data(
                b4_trace, trace_source, config, trace_domains, seed, device
            )
            trace_report = validate_continuous_branch_data(trace_frame, action_costs)
            trace_report.update({"domain": "public_trace", "seed": seed})
            if trace_report["status"] != "PASS":
                raise RuntimeError(f"trace branch integrity failed for seed {seed}")
            split_reports.append(trace_report)
            trace_frame.to_csv(
                run_dir / "branch_data" / f"public_trace__s{seed}.csv.gz",
                index=False,
                compression="gzip",
            )
            trace_models = _fit_formal_models(trace_frame, config, action_costs, seed + 50_000)
            _save_models(run_dir / "models", f"public_trace__s{seed}", trace_models)
            trace_diag, trace_states = _branch_diagnostics(
                trace_frame,
                trace_models,
                float(config["models"]["small_q_gap"]),
            )
            diagnostics.append(trace_diag)
            uncertainty_states.append(trace_states)
            trace_eval, trace_runtime = _evaluate_trace_davs(
                trace_models, trace_source, config, trace_domains, seed, device
            )
            trace_metrics.append(trace_eval)
            runtimes.append(trace_runtime)

        write_json(run_dir / "stage_status.json", {"stage": "aggregation", "at": _utcnow()})
        synthetic_metrics = pd.concat(synth_metrics, ignore_index=True)
        public_metrics = pd.concat(trace_metrics, ignore_index=True)
        runtime = pd.concat(runtimes, ignore_index=True)
        diagnostic_frame = pd.concat(diagnostics, ignore_index=True)
        state_frame = pd.concat(uncertainty_states, ignore_index=True)
        sensitivity_frame = pd.concat(
            [frame for frame in sensitivities if not frame.empty], ignore_index=True
        ) if any(not frame.empty for frame in sensitivities) else pd.DataFrame()
        loso_frame = pd.concat(
            [frame for frame in loso_rows if not frame.empty], ignore_index=True
        ) if any(not frame.empty for frame in loso_rows) else pd.DataFrame()
        synthetic_metrics.to_csv(run_dir / "synthetic_metrics.csv", index=False)
        public_metrics.to_csv(run_dir / "public_trace_metrics.csv", index=False)
        runtime.to_csv(run_dir / "runtime.csv", index=False)
        diagnostic_frame.to_csv(run_dir / "branch_diagnostics.csv", index=False)
        state_frame.to_csv(
            run_dir / "uncertainty_states.csv.gz", index=False, compression="gzip"
        )
        sensitivity_frame.to_csv(run_dir / "sensitivity.csv", index=False)
        loso_frame.to_csv(run_dir / "cross_scenario_generalization.csv", index=False)
        write_json(
            run_dir / "split_integrity.json",
            {
                "schema": "direct_action_value_selection.full_split_bundle.v1",
                "status": "PASS",
                "reports": split_reports,
            },
        )
        write_json(
            run_dir / "lineage.json",
            {
                "frozen_predecessor_artifacts": lineage,
                "predecessor_runs_modified": False,
                "decision_uses_transition_model": False,
                "decision_uses_policy_network": False,
                "branch_label_bootstrap": "frozen_b4_reward_value",
            },
        )
        pareto, gate = _pareto_and_stop_gate(
            synthetic_metrics, public_metrics, config
        )
        pareto.to_csv(run_dir / "pareto_analysis.csv", index=False)
        if smoke:
            gate["decision"] = "NOT_EVALUATED_SMOKE"
            gate["stop_condition_triggered"] = False
        write_json(run_dir / "full_stop_gate.json", gate)
        write_json(run_dir / "stage_status.json", {"stage": "completed", "at": _utcnow()})
        artifacts = sorted(
            path for path in run_dir.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        )
        ended_at = _utcnow()
        write_json(
            run_dir / "manifest.json",
            {
                "schema": "light.run_manifest.v3",
                "run_id": run_id,
                "matrix_row_id": "DAVS-FULL-001",
                "status": "completed",
                "termination": "full_validation_complete",
                "completion": {
                    "formal_claim_eligible": not smoke,
                    "minimal_gate": "CONTINUE",
                    "full_decision": gate["decision"],
                },
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed_seconds": time.perf_counter() - started_clock,
                "config_sha256": sha256_file(run_dir / "config.json"),
                "code_sha256": sha256_tree(
                    project_root / "src/dap/direct_action_value_selection"
                ),
                "input_config": {
                    "path": str(config_path.relative_to(project_root)),
                    "sha256": sha256_file(config_path),
                },
                "artifacts": {
                    str(path.relative_to(run_dir)): sha256_file(path) for path in artifacts
                },
                "guardrails": [
                    "frozen_predecessors_read_only",
                    "all_feasible_actions_same_arrival_tape",
                    "trajectory_group_split",
                    "chronological_public_trace_split",
                    "hard_true_cost_budget_mask",
                    "no_transition_or_policy_at_davs_decision",
                    "frozen_full_stop_gate",
                ],
            },
        )
        return run_dir
    except Exception as exc:
        write_json(
            run_dir / "failure.json",
            {
                "run_id": run_id,
                "status": "failed",
                "at": _utcnow(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
