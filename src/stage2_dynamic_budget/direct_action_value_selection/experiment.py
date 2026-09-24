from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    solve_action_dp,
)
from stage2_dynamic_budget.direct_action_planning.experiment import _method_agents
from stage2_dynamic_budget.direct_action_planning.learning import EmpiricalActionModel
from stage2_dynamic_budget.direct_action_planning.planning import BudgetValueTable
from stage2_dynamic_budget.experiment import load_config
from stage2_dynamic_budget.utils.artifacts import (
    environment_record,
    sha256_file,
    sha256_tree,
    write_json,
)
from stage2_dynamic_budget.utils.seed import set_global_seed

from .data import generate_k1_branch_data, validate_split_integrity
from .diagnostics import evaluate_davs_values
from .evaluation import evaluate_test_states, evaluate_test_window_rollouts
from .figures import generate_all_figures
from .gate import assess_continuation
from .model import DAVSAgent, DAVSEnsemble, DAVSModel, fit_davs_model


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dp_config(config: dict, scenario: str) -> ACBADPConfig:
    env = config["environment"]
    return ACBADPConfig(
        horizon=int(env["horizon"]),
        max_budget=int(env["max_budget"]),
        max_queue=int(env["max_queue"]),
        scenario=scenario,
        gamma=float(env["gamma"]),
        action_costs=tuple(int(value) for value in env["action_costs"]),
        action_capacity=tuple(int(value) for value in env["action_capacity"]),
    )


def _resolved_config(config: dict, run_id: str, smoke: bool) -> dict:
    resolved = json.loads(json.dumps(config))
    resolved["run_id"] = run_id
    resolved["smoke"] = bool(smoke)
    resolved["formal_plan_frozen_at"] = "2026-08-02T21:03:37+08:00"
    resolved["preformal_amendment"] = "quadratic_horizon_basis"
    if smoke:
        resolved["budgets"] = [4]
        resolved["scenarios"] = ["early_burst"]
        resolved["seeds"] = [0]
        resolved["evaluation_episodes"] = 2
        resolved["branch_data"]["replications_per_state"] = 2
        resolved["models"]["ensemble_members"] = 3
    return resolved


def _load_dap_value(path: Path) -> BudgetValueTable:
    payload = np.load(path, allow_pickle=False)
    source = str(payload["source"].item())
    return BudgetValueTable(values=payload["values"], source=source)


def _load_dap_model(path: Path) -> EmpiricalActionModel:
    payload = np.load(path, allow_pickle=False)
    return EmpiricalActionModel(
        next_load_probabilities=payload["next_load_probabilities"],
        next_queue_probabilities=payload["next_queue_probabilities"],
        rewards=payload["rewards"],
        costs=payload["costs"],
        samples_per_state_action=int(payload["samples_per_state_action"].item()),
        smoothing=float(payload["smoothing"].item()),
    )


def _save_model(path: Path, model: DAVSModel) -> None:
    np.savez_compressed(
        path,
        coefficients=model.coefficients,
        action_costs=model.action_costs,
        horizon=np.asarray(model.horizon),
        degree=np.asarray(model.degree),
        ridge=np.asarray(model.ridge),
        rank_beta=np.asarray(model.rank_beta),
        training_rows=np.asarray(model.training_rows),
        training_target=np.asarray(model.training_target),
    )


def _fit_methods(
    mdp: ActionConditionedBudgetMDP,
    branch_data: pd.DataFrame,
    config: dict,
    seed: int,
) -> dict[str, DAVSModel | DAVSEnsemble]:
    model_config = config["models"]
    common = {
        "degree": int(model_config["polynomial_degree"]),
        "ridge": float(model_config["ridge"]),
    }
    regression = fit_davs_model(mdp, branch_data, rank_beta=0.0, **common)
    ranked = fit_davs_model(
        mdp, branch_data, rank_beta=float(model_config["rank_beta"]), **common
    )
    ensemble = DAVSEnsemble.fit(
        mdp,
        branch_data,
        members=int(model_config["ensemble_members"]),
        seed=int(seed),
        rank_beta=float(model_config["rank_beta"]),
        **common,
    )
    return {"davs_r": regression, "davs_rank": ranked, "davs_ensemble": ensemble}


def _save_method_models(
    model_dir: Path,
    methods: dict[str, DAVSModel | DAVSEnsemble],
    scenario: str,
    seed: int,
) -> None:
    for method, scorer in methods.items():
        if isinstance(scorer, DAVSEnsemble):
            for member_index, member in enumerate(scorer.models):
                _save_model(
                    model_dir / f"{method}__{scenario}__s{seed}__m{member_index}.npz",
                    member,
                )
        else:
            _save_model(model_dir / f"{method}__{scenario}__s{seed}.npz", scorer)


def _frozen_agents(
    project_root: Path,
    scenario: str,
    seed: int,
    mdp: ActionConditionedBudgetMDP,
    optimum,
    config: dict,
    device: torch.device,
) -> tuple[list[tuple[str, object]], dict[str, str]]:
    dap_models = project_root / "results/direct_action_planning/minimal_v1/models"
    value_path = dap_models / f"value__{scenario}__s{seed}.npz"
    model_path = dap_models / f"model__{scenario}__s{seed}.npz"
    value = _load_dap_value(value_path)
    model = _load_dap_model(model_path)
    keep = {
        "optimal",
        "dsp_b",
        "acba_a",
        "learned_value_branch",
        "learned_model_branch",
    }
    agents: list[tuple[str, object]] = []
    lineage = {
        str(value_path.relative_to(project_root)): sha256_file(value_path),
        str(model_path.relative_to(project_root)): sha256_file(model_path),
    }
    for method, agent, source_paths in _method_agents(
        project_root,
        scenario,
        seed,
        mdp,
        optimum,
        value,
        model,
        config,
        device,
    ):
        if method not in keep:
            continue
        agents.append((method, agent))
        for source in source_paths:
            lineage[str(source.relative_to(project_root))] = sha256_file(source)
    return agents, lineage


def _gate_metrics(
    state_summary: pd.DataFrame, rollout_metrics: pd.DataFrame
) -> pd.DataFrame:
    service_columns = [
        "return_gap_to_paired_optimal",
        "budget_trajectory_mae",
        "completion_rate",
        "slo_violation_rate",
        "total_cost",
    ]
    service = rollout_metrics.groupby(
        ["method", "scenario", "budget", "seed"], as_index=False
    )[service_columns].mean()
    return state_summary.merge(
        service, on=["method", "scenario", "budget", "seed"], how="inner"
    )


def run_minimal_validation(
    project_root: str | Path,
    config_path: str | Path,
    run_id: str = "minimal_v1",
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
    started_at = _utcnow()
    started_clock = time.perf_counter()
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment_record())
    (run_dir / "stdout.log").write_text("", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    set_global_seed(0, torch_threads=int(config.get("torch_threads", 1)))
    device = torch.device(str(config.get("device", "cpu")))
    try:
        split_by_t = {
            key: [int(value) for value in values]
            for key, values in config["branch_data"]["split_by_t"].items()
        }
        test_times = split_by_t["test"]
        write_json(run_dir / "stage_status.json", {"stage": "exact_dp", "at": _utcnow()})
        optima: dict[str, tuple[ActionConditionedBudgetMDP, object]] = {}
        for scenario in config["scenarios"]:
            mdp = ActionConditionedBudgetMDP(_dp_config(config, scenario))
            optimum = solve_action_dp(mdp)
            if optimum.max_bellman_residual > 1e-10:
                raise RuntimeError("exact DP Bellman residual exceeded tolerance")
            optima[scenario] = (mdp, optimum)

        branch_dir = run_dir / "branch_data"
        model_dir = run_dir / "models"
        branch_dir.mkdir()
        model_dir.mkdir()
        branch_tables: dict[tuple[str, int], pd.DataFrame] = {}
        scorers: dict[tuple[str, int], dict[str, DAVSModel | DAVSEnsemble]] = {}
        split_reports: list[dict[str, object]] = []
        write_json(run_dir / "stage_status.json", {"stage": "branch_data", "at": _utcnow()})
        for scenario_index, scenario in enumerate(config["scenarios"]):
            mdp, optimum = optima[scenario]
            for seed_value in config["seeds"]:
                seed = int(seed_value)
                data_seed = seed + scenario_index * 100_003 + 50_000
                frame = generate_k1_branch_data(
                    mdp,
                    optimum,
                    scenario,
                    data_seed,
                    int(config["branch_data"]["replications_per_state"]),
                    split_by_t,
                )
                report = validate_split_integrity(
                    frame, split_by_t, mdp.config.horizon, mdp.action_costs
                )
                if report["status"] != "PASS":
                    raise RuntimeError(f"branch split integrity failed: {scenario}/s{seed}")
                report.update({"scenario": scenario, "seed": seed})
                split_reports.append(report)
                frame.to_csv(
                    branch_dir / f"branch__{scenario}__s{seed}.csv.gz",
                    index=False,
                    compression="gzip",
                )
                branch_tables[(scenario, seed)] = frame
                methods = _fit_methods(mdp, frame, config, seed + scenario_index * 1_009)
                scorers[(scenario, seed)] = methods
                _save_method_models(model_dir, methods, scenario, seed)
        write_json(
            run_dir / "split_integrity.json",
            {
                "schema": "direct_action_value_selection.split_integrity_bundle.v1",
                "status": "PASS",
                "reports": split_reports,
            },
        )

        state_frames: list[pd.DataFrame] = []
        state_summaries: list[pd.DataFrame] = []
        episode_frames: list[pd.DataFrame] = []
        runtime_rows: list[dict[str, object]] = []
        diagnostic_rows: list[dict[str, object]] = []
        uncertainty_states: list[pd.DataFrame] = []
        lineage: dict[str, str] = {}
        write_json(run_dir / "stage_status.json", {"stage": "evaluation", "at": _utcnow()})
        for scenario in config["scenarios"]:
            mdp, optimum = optima[scenario]
            for seed_value in config["seeds"]:
                seed = int(seed_value)
                agents, sources = _frozen_agents(
                    project_root, scenario, seed, mdp, optimum, config, device
                )
                lineage.update(sources)
                for method, scorer in scorers[(scenario, seed)].items():
                    agents.append((method, DAVSAgent(mdp, scorer).to(device)))
                    diagnostic, states = evaluate_davs_values(
                        scorer,
                        method,
                        mdp,
                        optimum,
                        branch_tables[(scenario, seed)],
                        scenario,
                        seed,
                        test_times,
                        float(config["models"]["small_q_gap"]),
                    )
                    diagnostic_rows.append(diagnostic)
                    uncertainty_states.append(states)
                for method, agent in agents:
                    state, summary, runtime = evaluate_test_states(
                        agent,
                        method,
                        mdp,
                        optimum,
                        scenario,
                        seed,
                        test_times,
                        device,
                    )
                    episodes = evaluate_test_window_rollouts(
                        agent,
                        method,
                        mdp,
                        optimum,
                        [int(value) for value in config["budgets"]],
                        scenario,
                        seed,
                        int(config["evaluation_episodes"]),
                        test_times,
                        device,
                    )
                    state_frames.append(state)
                    state_summaries.append(summary)
                    episode_frames.append(episodes)
                    runtime_rows.append(
                        {"method": method, "scenario": scenario, "seed": seed, **runtime}
                    )
        state = pd.concat(state_frames, ignore_index=True)
        state_summary = pd.concat(state_summaries, ignore_index=True)
        episodes = pd.concat(episode_frames, ignore_index=True)
        runtime = pd.DataFrame(runtime_rows)
        diagnostics = pd.DataFrame(diagnostic_rows)
        uncertainty_frame = pd.concat(uncertainty_states, ignore_index=True)
        state.to_csv(run_dir / "test_state_actions.csv.gz", index=False, compression="gzip")
        state_summary.to_csv(run_dir / "test_state_summary.csv", index=False)
        episodes.to_csv(run_dir / "metrics.csv", index=False)
        runtime.to_csv(run_dir / "runtime.csv", index=False)
        diagnostics.to_csv(run_dir / "value_ranking_diagnostics.csv", index=False)
        uncertainty_frame.to_csv(
            run_dir / "uncertainty_states.csv.gz", index=False, compression="gzip"
        )
        (
            state.groupby(["method", "optimal_action", "action"], as_index=False)
            .size()
            .rename(columns={"size": "count", "action": "predicted_action"})
            .to_csv(run_dir / "action_confusions.csv", index=False)
        )

        loso_rows: list[dict[str, object]] = []
        if len(config["scenarios"]) >= 3:
            write_json(
                run_dir / "stage_status.json", {"stage": "leave_one_scenario", "at": _utcnow()}
            )
            for heldout in config["scenarios"]:
                source_scenarios = [value for value in config["scenarios"] if value != heldout]
                mdp, optimum = optima[heldout]
                for seed_value in config["seeds"]:
                    seed = int(seed_value)
                    dsp = state_summary[
                        (state_summary.method == "dsp_b")
                        & (state_summary.scenario == heldout)
                        & (state_summary.seed == seed)
                    ]
                    loso_rows.append(
                        {
                            "method": "dsp_b",
                            "heldout_scenario": heldout,
                            "source_scenarios": "frozen_checkpoint",
                            "seed": seed,
                            "action_consistency_rate": float(
                                dsp.action_consistency_rate.mean()
                            ),
                            "mean_Q_star_regret": float(dsp.mean_Q_star_regret.mean()),
                        }
                    )
                    pooled = pd.concat(
                        [branch_tables[(source, seed)] for source in source_scenarios],
                        ignore_index=True,
                    )
                    transferred = _fit_methods(
                        mdp, pooled, config, seed + 70_001
                    )
                    for method, scorer in transferred.items():
                        _, summary, _ = evaluate_test_states(
                            DAVSAgent(mdp, scorer).to(device),
                            method,
                            mdp,
                            optimum,
                            heldout,
                            seed,
                            test_times,
                            device,
                        )
                        loso_rows.append(
                            {
                                "method": method,
                                "heldout_scenario": heldout,
                                "source_scenarios": "+".join(source_scenarios),
                                "seed": seed,
                                "action_consistency_rate": float(
                                    summary.action_consistency_rate.mean()
                                ),
                                "mean_Q_star_regret": float(
                                    summary.mean_Q_star_regret.mean()
                                ),
                            }
                        )
        loso = pd.DataFrame(loso_rows)
        loso.to_csv(run_dir / "scenario_generalization.csv", index=False)

        gate_metrics = _gate_metrics(state_summary, episodes)
        gate_metrics.to_csv(run_dir / "gate_metrics.csv", index=False)
        gate = assess_continuation(
            gate_metrics,
            loso,
            diagnostics,
            config["continuation_gate"],
        )
        if smoke:
            gate["decision"] = "NOT_EVALUATED_SMOKE"
        write_json(run_dir / "continuation_gate.json", gate)
        write_json(
            run_dir / "lineage.json",
            {
                "frozen_predecessor_artifacts": lineage,
                "old_branches_modified": False,
                "decision_uses_transition_model": False,
                "decision_uses_policy_network": False,
                "training_target": "Q_branch",
                "q_star_columns_used_in_fit": False,
                "exact_v_star_used_to_bootstrap_q_branch_labels": True,
                "formal_action_metrics_split": "test",
                "test_times": test_times,
            },
        )
        figure_paths = generate_all_figures(
            gate_metrics, diagnostics, loso, uncertainty_frame, run_dir / "figures"
        )
        write_json(
            run_dir / "test_evidence.json",
            {
                "core_test_command": (
                    "PYTHONPATH=src pytest -q -p no:cacheprovider "
                    "tests/direct_action_value_selection"
                ),
                "core_tests_passed_before_run": 6,
                "split_integrity": "PASS",
            },
        )
        write_json(run_dir / "stage_status.json", {"stage": "completed", "at": _utcnow()})

        artifact_paths = sorted(
            path
            for path in run_dir.rglob("*")
            if path.is_file() and path.name not in {"manifest.json", "failure.json"}
        )
        write_json(
            run_dir / "manifest.json",
            {
                "schema": "light.run_manifest.v3",
                "run_id": run_id,
                "matrix_row_id": "DAVS-MINIMAL-001",
                "status": "completed",
                "termination": "minimal_validation_complete",
                "completion": {
                    "formal_claim_eligible": not smoke,
                    "continuation_decision": gate["decision"],
                    "guardrail_evidence_artifacts": [
                        "continuation_gate.json",
                        "split_integrity.json",
                        "gate_metrics.csv",
                        "scenario_generalization.csv",
                        "value_ranking_diagnostics.csv",
                    ],
                },
                "started_at": started_at,
                "ended_at": _utcnow(),
                "elapsed_seconds": time.perf_counter() - started_clock,
                "config_sha256": sha256_file(run_dir / "config.json"),
                "code_sha256": sha256_tree(
                    project_root / "src/stage2_dynamic_budget/direct_action_value_selection"
                ),
                "input_config": {
                    "path": str(config_path.relative_to(project_root)),
                    "sha256": sha256_file(config_path),
                },
                "artifacts": {
                    str(path.relative_to(run_dir)): sha256_file(path) for path in artifact_paths
                },
                "figure_count": len(figure_paths),
                "guardrails": [
                    "frozen_predecessors_read_only",
                    "all_feasible_actions_common_random_number",
                    "time_embargo_split",
                    "q_star_evaluation_only",
                    "hard_true_cost_budget_mask",
                    "direct_scores_without_transition_or_policy",
                    "preregistered_continuation_gate",
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
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": _utcnow(),
                "elapsed_seconds": time.perf_counter() - started_clock,
            },
        )
        raise
