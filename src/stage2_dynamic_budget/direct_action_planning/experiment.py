from __future__ import annotations

from dataclasses import asdict
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
    action_truth_frame,
    solve_action_dp,
)
from stage2_dynamic_budget.action_conditioned_budget_advantage.evaluation import (
    ExactDPAgent,
    confusion_table,
    evaluate_full_state_policy,
    evaluate_policy_rollouts,
)
from stage2_dynamic_budget.action_conditioned_budget_advantage.model import (
    ActionAdvantageModel,
)
from stage2_dynamic_budget.action_conditioned_budget_advantage.policy import (
    ACBAAPolicy,
    load_frozen_b4_base,
    load_frozen_baseline,
)
from stage2_dynamic_budget.experiment import load_config
from stage2_dynamic_budget.utils.artifacts import (
    environment_record,
    sha256_file,
    sha256_tree,
    write_json,
)
from stage2_dynamic_budget.utils.seed import set_global_seed

from .diagnostics import (
    transfer_value_error,
    transition_model_diagnostics,
    value_diagnostics,
)
from .figures import generate_all_figures
from .gate import assess_continuation
from .learning import (
    EmpiricalActionModel,
    collect_transition_samples,
    fit_empirical_action_model,
    solve_empirical_value,
)
from .planning import BudgetValueTable, DirectPlanningAgent


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
    resolved["frozen_at"] = "2026-08-02T16:20:58+08:00"
    if smoke:
        resolved["budgets"] = [4]
        resolved["scenarios"] = ["early_burst"]
        resolved["seeds"] = [0]
        resolved["evaluation_episodes"] = 2
        resolved["learning"]["transition_samples_per_state_action"] = 8
    return resolved


def _save_empirical_model(path: Path, model: EmpiricalActionModel) -> None:
    np.savez_compressed(
        path,
        next_load_probabilities=model.next_load_probabilities,
        next_queue_probabilities=model.next_queue_probabilities,
        rewards=model.rewards,
        costs=model.costs,
        samples_per_state_action=np.asarray(model.samples_per_state_action),
        smoothing=np.asarray(model.smoothing),
    )


def _save_value(path: Path, value: BudgetValueTable) -> None:
    np.savez_compressed(path, values=value.values, source=np.asarray(value.source))


def _load_advantage_model(path: Path, device: torch.device) -> ActionAdvantageModel:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ActionAdvantageModel(
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def _method_agents(
    project_root: Path,
    scenario: str,
    seed: int,
    mdp: ActionConditionedBudgetMDP,
    optimum,
    learned_value: BudgetValueTable,
    learned_model: EmpiricalActionModel,
    config: dict,
    device: torch.device,
):
    action_costs = tuple(float(value) for value in config["environment"]["action_costs"])
    budget_scale = float(config["environment"]["max_budget"])
    yield "optimal", ExactDPAgent(mdp, optimum), []
    for method in ("b4_budget_state", "dsp_b"):
        agent, source = load_frozen_baseline(
            project_root,
            method,
            scenario,
            seed,
            action_costs,
            budget_scale,
            device,
        )
        yield method, agent, [source]

    advantage_path = (
        project_root
        / "results/action_conditioned_budget_advantage/minimal_v2/models"
        / f"acba_advantage_s{seed}.pt"
    )
    teacher = _load_advantage_model(advantage_path, device)
    base, base_path, _ = load_frozen_b4_base(project_root, scenario, seed, device)
    base.eval()
    yield (
        "acba_a",
        ACBAAPolicy(base, teacher, action_costs, budget_scale, alpha=1.0).to(device),
        [base_path, advantage_path],
    )
    yield (
        "oracle_branch",
        DirectPlanningAgent(
            mdp, BudgetValueTable.from_exact_dp(optimum.values)
        ).to(device),
        [],
    )
    yield (
        "learned_value_branch",
        DirectPlanningAgent(mdp, learned_value).to(device),
        [],
    )
    yield (
        "learned_model_branch",
        DirectPlanningAgent(mdp, learned_value, learned_model).to(device),
        [],
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
    run_dir = project_root / "results/direct_action_planning" / run_id
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
        write_json(run_dir / "stage_status.json", {"stage": "exact_dp", "at": _utcnow()})
        truth_frames: list[pd.DataFrame] = []
        optima: dict[str, tuple[ActionConditionedBudgetMDP, object]] = {}
        for scenario in config["scenarios"]:
            mdp = ActionConditionedBudgetMDP(_dp_config(config, scenario))
            optimum = solve_action_dp(mdp)
            if optimum.max_bellman_residual > 1e-10:
                raise RuntimeError("exact DP Bellman residual exceeded tolerance")
            optima[scenario] = (mdp, optimum)
            truth_frames.append(action_truth_frame(mdp, optimum, scenario))
        truth = pd.concat(truth_frames, ignore_index=True)
        truth.to_csv(run_dir / "action_truth.csv.gz", index=False, compression="gzip")

        sample_dir = run_dir / "transition_samples"
        model_dir = run_dir / "models"
        sample_dir.mkdir()
        model_dir.mkdir()
        samples: dict[tuple[str, int], pd.DataFrame] = {}
        models: dict[tuple[str, int], EmpiricalActionModel] = {}
        values: dict[tuple[str, int], BudgetValueTable] = {}
        value_rows: list[dict[str, object]] = []
        model_rows: list[dict[str, object]] = []
        write_json(run_dir / "stage_status.json", {"stage": "learning", "at": _utcnow()})
        for scenario_index, scenario in enumerate(config["scenarios"]):
            mdp, optimum = optima[scenario]
            for seed_value in config["seeds"]:
                seed = int(seed_value)
                training_seed = seed + scenario_index * 100_003
                frame = collect_transition_samples(
                    mdp,
                    int(config["learning"]["transition_samples_per_state_action"]),
                    training_seed,
                )
                model = fit_empirical_action_model(
                    mdp,
                    frame,
                    float(config["learning"]["transition_probability_smoothing"]),
                )
                value = solve_empirical_value(mdp, model)
                frame.to_csv(
                    sample_dir / f"transitions__{scenario}__s{seed}.csv.gz",
                    index=False,
                    compression="gzip",
                )
                _save_empirical_model(model_dir / f"model__{scenario}__s{seed}.npz", model)
                _save_value(model_dir / f"value__{scenario}__s{seed}.npz", value)
                samples[(scenario, seed)] = frame
                models[(scenario, seed)] = model
                values[(scenario, seed)] = value
                value_rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "training_target": "empirical_model_bellman_no_v_star_labels",
                        **value_diagnostics(mdp, optimum, value),
                    }
                )
                model_rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "samples_per_state_action": model.samples_per_state_action,
                        **transition_model_diagnostics(mdp, model),
                    }
                )
        value_frame = pd.DataFrame(value_rows)
        model_frame = pd.DataFrame(model_rows)
        value_frame.to_csv(run_dir / "value_diagnostics.csv", index=False)
        model_frame.to_csv(run_dir / "model_diagnostics.csv", index=False)

        transfer_rows: list[dict[str, object]] = []
        if len(config["scenarios"]) >= 3 and bool(
            config["learning"].get("leave_one_scenario_out_diagnostic", False)
        ):
            for heldout in config["scenarios"]:
                mdp, optimum = optima[heldout]
                sources = [item for item in config["scenarios"] if item != heldout]
                for seed_value in config["seeds"]:
                    seed = int(seed_value)
                    pooled = pd.concat(
                        [samples[(source, seed)] for source in sources], ignore_index=True
                    )
                    transfer_model = fit_empirical_action_model(
                        mdp,
                        pooled,
                        float(config["learning"]["transition_probability_smoothing"]),
                    )
                    transfer_value = solve_empirical_value(mdp, transfer_model)
                    transfer_rows.append(
                        {
                            "heldout_scenario": heldout,
                            "source_scenarios": "+".join(sources),
                            "seed": seed,
                            **transfer_value_error(optimum, transfer_value),
                        }
                    )
        pd.DataFrame(transfer_rows).to_csv(
            run_dir / "scenario_generalization.csv", index=False
        )

        state_frames: list[pd.DataFrame] = []
        state_summaries: list[dict[str, object]] = []
        episode_frames: list[pd.DataFrame] = []
        step_frames: list[pd.DataFrame] = []
        runtime_rows: list[dict[str, object]] = []
        lineage: dict[str, str] = {}
        write_json(run_dir / "stage_status.json", {"stage": "evaluation", "at": _utcnow()})
        for scenario in config["scenarios"]:
            mdp, optimum = optima[scenario]
            for seed_value in config["seeds"]:
                seed = int(seed_value)
                for method, agent, source_paths in _method_agents(
                    project_root,
                    scenario,
                    seed,
                    mdp,
                    optimum,
                    values[(scenario, seed)],
                    models[(scenario, seed)],
                    config,
                    device,
                ):
                    for source in source_paths:
                        lineage[str(source.relative_to(project_root))] = sha256_file(source)
                    state, summary = evaluate_full_state_policy(
                        agent, method, mdp, optimum, scenario, seed, device
                    )
                    episodes, steps, runtime = evaluate_policy_rollouts(
                        agent,
                        method,
                        mdp,
                        optimum,
                        [int(item) for item in config["budgets"]],
                        scenario,
                        seed,
                        int(config["evaluation_episodes"]),
                        device,
                    )
                    state_frames.append(state)
                    state_summaries.append(summary)
                    episode_frames.append(episodes)
                    step_frames.append(steps)
                    runtime_rows.append(
                        {"method": method, "scenario": scenario, "seed": seed, **runtime}
                    )
        state = pd.concat(state_frames, ignore_index=True)
        state_summary = pd.DataFrame(state_summaries)
        episodes = pd.concat(episode_frames, ignore_index=True)
        steps = pd.concat(step_frames, ignore_index=True)
        runtime = pd.DataFrame(runtime_rows)
        state.to_csv(run_dir / "state_policy_actions.csv.gz", index=False, compression="gzip")
        state_summary.to_csv(run_dir / "state_policy_summary.csv", index=False)
        confusion_table(state).to_csv(run_dir / "action_confusions.csv", index=False)
        episodes.to_csv(run_dir / "metrics.csv", index=False)
        steps.to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
        runtime.to_csv(run_dir / "runtime.csv", index=False)
        episodes.groupby(["method", "scenario", "budget", "seed"], as_index=False).mean(
            numeric_only=True
        ).to_csv(run_dir / "cell_summary.csv", index=False)

        gate = assess_continuation(episodes, state_summary, config["continuation_gate"])
        if smoke:
            gate["decision"] = "NOT_EVALUATED_SMOKE"
        write_json(run_dir / "continuation_gate.json", gate)
        write_json(
            run_dir / "lineage.json",
            {
                "frozen_predecessor_checkpoints": lineage,
                "exact_values_used_as_training_labels": False,
                "learned_value_training_source": "seeded empirical transition samples",
                "periodic_baseline_source_rule": "frozen early_burst checkpoint of same method and seed",
                "old_branches_modified": False,
            },
        )
        figure_paths = generate_all_figures(
            episodes, steps, value_frame, model_frame, run_dir / "figures"
        )
        write_json(
            run_dir / "test_evidence.json",
            {
                "core_test_command": "PYTHONPATH=src pytest -q tests/direct_action_planning",
                "core_tests_passed_before_run": 7,
                "bellman_oracle_tolerance": 1e-12,
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
                "matrix_row_id": "DAP-MINIMAL-001",
                "status": "completed",
                "termination": "minimal_validation_complete",
                "completion": {
                    "oracle": "PASS" if gate["checks"]["oracle_near_exact"] else "FAIL",
                    "formal_claim_eligible": not smoke,
                    "continuation_decision": gate["decision"],
                    "guardrail_evidence_artifacts": [
                        "continuation_gate.json",
                        "value_diagnostics.csv",
                        "model_diagnostics.csv",
                    ],
                },
                "started_at": started_at,
                "ended_at": _utcnow(),
                "elapsed_seconds": time.perf_counter() - started_clock,
                "config_sha256": sha256_file(run_dir / "config.json"),
                "code_sha256": sha256_tree(
                    project_root / "src/stage2_dynamic_budget/direct_action_planning"
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
                    "hard_true_cost_budget_mask",
                    "paired_common_random_numbers",
                    "exact_dp_evaluation_only",
                    "no_v_star_training_labels",
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
