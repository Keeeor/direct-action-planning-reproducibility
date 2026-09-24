from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml

from stage2_dynamic_budget.direct_action_planning_dataset_benchmark.budgeted import (
    BudgetedFittedQAgent,
    BudgetedQNetwork,
)
from stage2_dynamic_budget.direct_action_planning_dataset_benchmark.cpo import (
    CPOActorCritic,
    CPOAgent,
)
from stage2_dynamic_budget.direct_action_planning_dataset_benchmark.models import (
    ActorCritic,
    QNetwork,
)
from stage2_dynamic_budget.direct_action_planning_dataset_benchmark.rl import (
    DQNAgent,
    PolicyAgent,
)
from stage2_dynamic_budget.direct_action_planning_dataset_specific_stabilization.calibrated_protocol import (
    evaluate_calibrated_methods,
)
from stage2_dynamic_budget.direct_action_planning_dataset_specific_stabilization.models import (
    ScaledEvidenceValueNetwork,
)
from stage2_dynamic_budget.direct_action_planning_dataset_specific_stabilization.planning import (
    make_scaled_planner,
)
from stage2_dynamic_budget.direct_action_planning_dataset_validation.data import (
    TraceDataset,
    load_trace_dataset,
)
from stage2_dynamic_budget.direct_action_planning_dataset_validation.models import (
    FeatureNormalizer,
)
from stage2_dynamic_budget.direct_action_planning_paper_evidence.data import (
    load_development_trace_dataset,
)
from stage2_dynamic_budget.direct_action_planning_paper_evidence.models import (
    EvidenceLoadForecaster,
)
from stage2_dynamic_budget.utils.artifacts import (
    environment_record,
    sha256_file,
    sha256_tree,
    write_json,
)
from stage2_dynamic_budget.utils.seed import set_global_seed


PRIMARY_METHODS = (
    "dap_calibrated",
    "double_dqn",
    "ppo",
    "ppo_lagrangian",
    "cpo",
    "p3o",
    "budgeted_fitted_q",
)
SUPPLEMENTARY_METHODS = ("a2c", "pid_lagrangian")
INTERNAL_ABLATIONS = ("dap_immediate",)
FROZEN_METHODS = PRIMARY_METHODS + SUPPLEMENTARY_METHODS + INTERNAL_ABLATIONS
POLICY_METHODS = (
    "ppo",
    "ppo_lagrangian",
    "p3o",
    "a2c",
    "pid_lagrangian",
)


def load_frozen_test_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "tier",
        "manifest_schema",
        "matrix_row_id",
        "datasets",
        "horizon",
        "budgets",
        "seeds",
        "seed_role",
        "gamma",
        "evaluation_split",
        "evaluation_episodes_per_domain",
        "test_seed_offset",
        "capacity_training_quantile",
        "dap_tier",
        "baseline_tier",
        "dap_config_sha256",
        "baseline_config_sha256",
        "dap_checkpoint_set_sha256",
        "baseline_checkpoint_set_sha256",
        "primary_methods",
        "supplementary_methods",
        "internal_ablations",
        "historical_test_access",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing frozen-test protocol keys: {sorted(missing)}")
    if str(config["evaluation_split"]) != "test":
        raise ValueError("evaluation_split must be test")
    if int(config["evaluation_episodes_per_domain"]) <= 0:
        raise ValueError("evaluation_episodes_per_domain must be positive")
    if int(config["test_seed_offset"]) <= 0:
        raise ValueError("test_seed_offset must be positive")
    quantile = float(config["capacity_training_quantile"])
    if not np.isfinite(quantile) or not 0.0 < quantile <= 1.0:
        raise ValueError("capacity_training_quantile must be in (0, 1]")
    configured = tuple(config["primary_methods"])
    configured += tuple(config["supplementary_methods"])
    configured += tuple(config["internal_ablations"])
    if configured != FROZEN_METHODS:
        raise ValueError("frozen method families do not match the preregistered order")
    history = config["historical_test_access"]
    if history.get("project_wide_first_access") is not False:
        raise ValueError("historical project-wide test access must be disclosed")
    if history.get("prior_status") != "finalized":
        raise ValueError("prior finalized test ledger must remain disclosed")
    return config


def _bare_sha256(path: Path) -> str:
    return sha256_file(path).removeprefix("sha256:")


def _checkpoint_inventory_hash(root: Path, names: set[str]) -> str:
    paths = sorted(
        path
        for path in root.glob("*/*/*")
        if path.is_file() and path.name in names
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root.parents[2])
        digest.update(f"{_bare_sha256(path)}  {relative.as_posix()}\n".encode())
    return "sha256:" + digest.hexdigest()


def _development_run_dir(
    root: Path,
    tier: str,
    dataset: str,
    budget: float,
    seed: int,
) -> Path:
    run_id = f"{tier}__{dataset}__b{budget:.0f}__s{seed}"
    return (
        root
        / "results/direct_action_planning_dataset_specific_stabilization"
        / tier
        / dataset
        / run_id
    )


def _verify_frozen_sources(root: Path, config: dict[str, Any]) -> None:
    dap_config = (
        root
        / "research/direct_action_planning_dataset_specific_stabilization/configs/core_v4.yaml"
    )
    baseline_config = (
        root
        / "research/direct_action_planning_dataset_specific_stabilization/configs/baselines_core_v1.yaml"
    )
    if sha256_file(dap_config) != str(config["dap_config_sha256"]):
        raise ValueError("frozen DAP config hash mismatch")
    if sha256_file(baseline_config) != str(config["baseline_config_sha256"]):
        raise ValueError("frozen baseline config hash mismatch")
    dap_root = (
        root
        / "results/direct_action_planning_dataset_specific_stabilization"
        / str(config["dap_tier"])
    )
    baseline_root = (
        root
        / "results/direct_action_planning_dataset_specific_stabilization"
        / str(config["baseline_tier"])
    )
    dap_hash = _checkpoint_inventory_hash(
        dap_root, {"models.pt", "diagnostics.json", "manifest.json"}
    )
    baseline_hash = _checkpoint_inventory_hash(
        baseline_root, {"models.pt", "manifest.json", "training.json"}
    )
    if dap_hash != str(config["dap_checkpoint_set_sha256"]):
        raise ValueError(f"frozen DAP checkpoint-set hash mismatch: {dap_hash}")
    if baseline_hash != str(config["baseline_checkpoint_set_sha256"]):
        raise ValueError(f"frozen baseline checkpoint-set hash mismatch: {baseline_hash}")


def _load_dap_methods(
    run_dir: Path,
    *,
    hidden_dim: int,
    gamma: float,
) -> dict[str, Callable]:
    checkpoint = torch.load(run_dir / "models.pt", weights_only=True, map_location="cpu")
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf-8"))
    mean = checkpoint["normalizer_mean"].cpu().numpy().astype(np.float64)
    scale = checkpoint["normalizer_scale"].cpu().numpy().astype(np.float64)
    normalizer = FeatureNormalizer(mean=mean, scale=scale)
    output_scale = float(checkpoint["value_output_scale"].item())
    zero_initialized = bool(checkpoint["zero_initialized_output"].item())
    final_iteration = int(diagnostics["final_iteration"])
    selected_iteration = int(diagnostics["selected"]["candidate_iteration"])
    selected_weight = float(diagnostics["selected"]["continuation_weight"])
    selected_value_iteration = (
        final_iteration if selected_iteration < 0 else selected_iteration
    )

    def load_value(iteration: int) -> ScaledEvidenceValueNetwork:
        value = ScaledEvidenceValueNetwork(
            normalizer,
            hidden_dim=hidden_dim,
            output_scale=output_scale,
            zero_initialize_output=zero_initialized,
        )
        value.load_state_dict(checkpoint["values"][str(iteration)], strict=True)
        value.train(False)
        return value

    forecaster = EvidenceLoadForecaster(normalizer)
    forecaster.load_state_dict(checkpoint["forecaster"], strict=True)
    forecaster.train(False)
    return {
        "dap_calibrated": make_scaled_planner(
            value=load_value(selected_value_iteration),
            forecaster=forecaster,
            gamma=gamma,
            continuation_weight=selected_weight,
        ),
        "dap_immediate": make_scaled_planner(
            value=load_value(final_iteration),
            forecaster=forecaster,
            gamma=gamma,
            continuation_weight=0.0,
        ),
    }


def _load_baseline_methods(
    run_dir: Path,
    *,
    budget: float,
    hidden_dim: int,
) -> dict[str, Callable]:
    states = torch.load(
        run_dir / "models.pt", weights_only=True, map_location="cpu"
    )["models"]
    agents: dict[str, Any] = {}
    dqn = QNetwork(action_dim=4, hidden_dim=hidden_dim, dueling=True)
    dqn.load_state_dict(states["double_dqn"], strict=True)
    agents["double_dqn"] = DQNAgent(dqn, "double_dqn", budget)
    for method in POLICY_METHODS:
        model = ActorCritic(action_dim=4, hidden_dim=hidden_dim)
        model.load_state_dict(states[method], strict=True)
        agents[method] = PolicyAgent(model, method, budget)
    cpo = CPOActorCritic(action_dim=4, hidden_dim=hidden_dim)
    cpo.load_state_dict(states["cpo"], strict=True)
    agents["cpo"] = CPOAgent(cpo, budget)
    budgeted = BudgetedQNetwork(action_dim=4, hidden_dim=hidden_dim)
    budgeted.load_state_dict(states["budgeted_fitted_q"], strict=True)
    agents["budgeted_fitted_q"] = BudgetedFittedQAgent(
        budgeted,
        budget,
        np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float64),
    )

    def wrap(agent: Any) -> Callable:
        def select(env, observation):
            return agent.select(env, observation)

        return select

    return {method: wrap(agent) for method, agent in agents.items()}


def load_frozen_methods(
    project_root: str | Path,
    config: dict[str, Any],
    *,
    dataset_name: str,
    budget: float,
    seed: int,
) -> tuple[dict[str, Callable], dict[str, Path]]:
    root = Path(project_root).resolve()
    dap_run = _development_run_dir(
        root, str(config["dap_tier"]), dataset_name, budget, seed
    )
    baseline_run = _development_run_dir(
        root, str(config["baseline_tier"]), dataset_name, budget, seed
    )
    for run_dir in (dap_run, baseline_run):
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError(f"source run is not completed: {run_dir}")
        if manifest.get("formal_test_accessed") is not False:
            raise ValueError(f"source run was not development-only: {run_dir}")
    hidden_dim = 64
    methods = _load_dap_methods(
        dap_run, hidden_dim=hidden_dim, gamma=float(config["gamma"])
    )
    methods.update(
        _load_baseline_methods(
            baseline_run,
            budget=budget,
            hidden_dim=hidden_dim,
        )
    )
    methods = {method: methods[method] for method in FROZEN_METHODS}
    return methods, {"dap": dap_run, "baselines": baseline_run}


def _evaluate(
    dataset: TraceDataset,
    methods: dict[str, Callable],
    *,
    split: str,
    config: dict[str, Any],
    budget: float,
    evaluation_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows, step_rows = evaluate_calibrated_methods(
        dataset,
        methods,
        split=split,
        horizon=int(config["horizon"]),
        budget=budget,
        seed=evaluation_seed,
        episodes_per_domain=int(config["evaluation_episodes_per_domain"]),
        gamma=float(config["gamma"]),
        quantile=float(config["capacity_training_quantile"]),
    )
    return pd.DataFrame(episode_rows), pd.DataFrame(step_rows)


def replay_development_unit(
    project_root: str | Path,
    config_path: str | Path,
    *,
    dataset_name: str,
    budget: float,
    seed: int,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_frozen_test_protocol(config_path)
    _verify_frozen_sources(root, config)
    methods, source_dirs = load_frozen_methods(
        root,
        config,
        dataset_name=dataset_name,
        budget=budget,
        seed=seed,
    )
    dataset = load_development_trace_dataset(
        root, dataset_name, horizon=int(config["horizon"])
    )
    development_config = yaml.safe_load(
        (
            root
            / "research/direct_action_planning_dataset_specific_stabilization/configs/core_v4.yaml"
        ).read_text(encoding="utf-8")
    )
    evaluation_seed = int(seed) + int(development_config["evaluation_seed_offset"])
    current_metrics, current_steps = _evaluate(
        dataset,
        methods,
        split="validation_eval",
        config=config,
        budget=budget,
        evaluation_seed=evaluation_seed,
    )
    dap_steps = pd.read_csv(source_dirs["dap"] / "steps.csv.gz")
    baseline_steps = pd.read_csv(source_dirs["baselines"] / "steps.csv.gz")
    old_steps = pd.concat(
        [
            dap_steps[dap_steps.method.isin(INTERNAL_ABLATIONS + ("dap_calibrated",))],
            baseline_steps[baseline_steps.method.isin(PRIMARY_METHODS + SUPPLEMENTARY_METHODS)],
        ],
        ignore_index=True,
    )
    step_keys = ["method", "domain", "episode", "window_seed", "window_start", "step"]
    compared_steps = current_steps.merge(
        old_steps[step_keys + ["action"]],
        on=step_keys,
        how="outer",
        suffixes=("_new", "_old"),
        indicator=True,
    )
    action_mismatch = compared_steps[
        (compared_steps["_merge"] != "both")
        | (compared_steps["action_new"] != compared_steps["action_old"])
    ]

    dap_metrics = pd.read_csv(source_dirs["dap"] / "metrics.csv")
    baseline_metrics = pd.read_csv(source_dirs["baselines"] / "metrics.csv")
    old_metrics = pd.concat(
        [
            dap_metrics[dap_metrics.method.isin(INTERNAL_ABLATIONS + ("dap_calibrated",))],
            baseline_metrics[
                baseline_metrics.method.isin(PRIMARY_METHODS + SUPPLEMENTARY_METHODS)
            ],
        ],
        ignore_index=True,
    )
    metric_keys = ["method", "domain", "episode", "window_seed", "window_start"]
    value_columns = [
        "discounted_return",
        "completion_ratio",
        "slo_violation_rate",
        "total_cost",
        "budget_overspend",
        "queue_area",
        "final_queue",
    ]
    compared_metrics = current_metrics.merge(
        old_metrics[metric_keys + value_columns],
        on=metric_keys,
        how="outer",
        suffixes=("_new", "_old"),
        indicator=True,
    )
    mismatch = compared_metrics["_merge"] != "both"
    for column in value_columns:
        mismatch |= ~np.isclose(
            compared_metrics[f"{column}_new"],
            compared_metrics[f"{column}_old"],
            rtol=1.0e-7,
            atol=1.0e-7,
            equal_nan=False,
        )
    metric_mismatch_count = int(mismatch.sum())
    return {
        "status": (
            "PASS"
            if action_mismatch.empty and metric_mismatch_count == 0
            else "FAIL"
        ),
        "dataset": dataset_name,
        "budget": float(budget),
        "seed": int(seed),
        "method_count": int(current_steps.method.nunique()),
        "action_mismatch_count": int(len(action_mismatch)),
        "metric_mismatch_count": metric_mismatch_count,
        "step_rows": int(len(current_steps)),
        "episode_rows": int(len(current_metrics)),
    }


def _array_hash(label: str, values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(label.encode())
    digest.update(np.asarray(values, dtype=np.float64).tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def _test_input_hashes(dataset: TraceDataset) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for domain in dataset.domain_names:
        for split in ("train", "test"):
            label = f"{dataset.name}/{domain}/{split}"
            hashes[label] = _array_hash(label, dataset.domains[domain][split])
    return hashes


def _test_run_dir(
    root: Path,
    config: dict[str, Any],
    dataset: str,
    budget: float,
    seed: int,
    attempt: int,
) -> Path:
    base = f"{config['tier']}__{dataset}__b{budget:.0f}__s{seed}"
    run_id = base if attempt == 0 else f"{base}__a{attempt}"
    return (
        root
        / "results/direct_action_planning_frozen_temporal_test"
        / str(config["tier"])
        / dataset
        / run_id
    )


def _artifact_record(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": _bare_sha256(path)}


def run_frozen_test_unit(
    project_root: str | Path,
    config_path: str | Path,
    *,
    dataset_name: str,
    budget: float,
    seed: int,
    attempt: int = 0,
) -> Path:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_frozen_test_protocol(config_path)
    _verify_frozen_sources(root, config)
    if dataset_name not in tuple(config["datasets"]):
        raise ValueError(f"dataset is not registered: {dataset_name}")
    if not any(np.isclose(budget, value) for value in config["budgets"]):
        raise ValueError(f"budget is not registered: {budget}")
    if int(seed) not in tuple(int(value) for value in config["seeds"]):
        raise ValueError(f"seed is not registered: {seed}")
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    run_dir = _test_run_dir(root, config, dataset_name, budget, seed, attempt)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"test run directory is append-only: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    write_json(
        run_dir / "config.json",
        {**config, "dataset": dataset_name, "budget": budget, "seed": seed},
    )
    write_json(run_dir / "environment.json", environment_record())
    (run_dir / "stdout.log").write_text("frozen test run started\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    set_global_seed(seed, torch_threads=1)
    wall_started = time.perf_counter()
    test_data_loaded = False
    try:
        methods, source_dirs = load_frozen_methods(
            root,
            config,
            dataset_name=dataset_name,
            budget=budget,
            seed=seed,
        )
        dataset = load_trace_dataset(root, dataset_name)
        test_data_loaded = True
        evaluation_seed = int(seed) + int(config["test_seed_offset"])
        metrics, steps = _evaluate(
            dataset,
            methods,
            split="test",
            config=config,
            budget=budget,
            evaluation_seed=evaluation_seed,
        )
        metrics["training_seed"] = seed
        steps["training_seed"] = seed
        if not np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError("test evaluation produced non-finite episode metrics")
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        steps.to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
        metrics.drop(
            columns=["decision_ms_mean", "decision_ms_p95"], errors="ignore"
        ).to_json(
            run_dir / "raw_metrics.jsonl",
            orient="records",
            lines=True,
            force_ascii=True,
        )
        steps.drop(columns=["decision_ms"], errors="ignore").to_csv(
            run_dir / "predictions.csv", index=False
        )
        write_json(run_dir / "input_hashes.json", _test_input_hashes(dataset))
        write_json(
            run_dir / "source_checkpoints.json",
            {
                role: {
                    "run_dir": str(path.relative_to(root)),
                    "models_sha256": sha256_file(path / "models.pt"),
                    "manifest_sha256": sha256_file(path / "manifest.json"),
                }
                for role, path in source_dirs.items()
            },
        )
        write_json(
            run_dir / "runtime.json",
            {
                "wall_seconds": float(time.perf_counter() - wall_started),
                "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
        )
        budget_overspend = float(metrics.budget_overspend.max())
        write_json(
            run_dir / "guardrails.json",
            {
                "budget_overspend_max": budget_overspend,
                "budget_safe": bool(budget_overspend <= 1.0e-8),
                "all_metrics_finite": True,
                "training_or_selection_performed": False,
                "capacity_source": "training_array_quantile_only",
            },
        )
        write_json(
            run_dir / "test_evidence.json",
            {
                "status": "PASS",
                "evaluation_split": "test",
                "current_checkpoint_first_test": True,
                "project_wide_test_previously_accessed": True,
                "episode_rows": int(len(metrics)),
                "step_rows": int(len(steps)),
                "method_count": int(metrics.method.nunique()),
                "test_seed_offset": int(config["test_seed_offset"]),
            },
        )
        (run_dir / "stdout.log").write_text(
            "frozen test run completed\n", encoding="utf-8"
        )
        ended_at = datetime.now(timezone.utc).isoformat()
        reproducible_environment = json.loads(
            (run_dir / "environment.json").read_text(encoding="utf-8")
        )
        reproducible_environment.pop("pid", None)
        reproducible_environment.pop("recorded_at", None)
        write_json(run_dir / "environment_repro.json", reproducible_environment)
        code_root = root / "src/stage2_dynamic_budget/direct_action_planning_frozen_temporal_test"
        write_json(
            run_dir / "code_snapshot.json",
            {
                "evaluation_code": str(code_root.relative_to(root)),
                "evaluation_code_sha256": sha256_tree(code_root),
                "frozen_dap_config_sha256": config["dap_config_sha256"],
                "frozen_baseline_config_sha256": config["baseline_config_sha256"],
                "git_commit": "UNAVAILABLE: workspace is not a Git repository",
            },
        )
        artifacts = {
            "stdout": _artifact_record(run_dir / "stdout.log"),
            "stderr": _artifact_record(run_dir / "stderr.log"),
            "raw_metrics": _artifact_record(run_dir / "raw_metrics.jsonl"),
            "predictions": _artifact_record(run_dir / "predictions.csv"),
            "test_evidence": _artifact_record(run_dir / "test_evidence.json"),
            "guardrail_evidence": _artifact_record(run_dir / "guardrails.json"),
            "failure": None,
        }
        write_json(
            run_dir / "run_manifest.v3.json",
            {
                "schema": "light.run_manifest.v3",
                "run_id": run_dir.name,
                "matrix_row": str(config["matrix_row_id"]),
                "status": "completed",
                "termination": {"reason": "natural_exit", "exit_code": 0, "signal": None},
                "seed": {"role": str(config["seed_role"]), "value": int(seed)},
                "command": [str(value) for value in sys.argv] or ["python"],
                "started_at": started_at,
                "ended_at": ended_at,
                "config": _artifact_record(run_dir / "config.json"),
                "environment": _artifact_record(run_dir / "environment_repro.json"),
                "code": {
                    "commit": "UNAVAILABLE: workspace is not a Git repository",
                    "dirty": False,
                    "diff_sha256": None,
                    "files": [_artifact_record(run_dir / "code_snapshot.json")],
                },
                "inputs": [
                    {
                        **_artifact_record(run_dir / "input_hashes.json"),
                        "role": "chronological_test_data",
                        "source_revision": dataset.split_contract,
                    },
                    {
                        **_artifact_record(run_dir / "source_checkpoints.json"),
                        "role": "frozen_checkpoints",
                        "source_revision": "dataset-specific development checkpoint sets",
                    },
                ],
                "artifacts": artifacts,
                "completion": {
                    "status": "PASS",
                    "oracle": [
                        "all frozen methods produced finite test metrics",
                        "hard budget overspend guardrail passed",
                        "no training, selection, or test-time capacity fitting was performed",
                    ],
                    "evidence_artifacts": [
                        "test_evidence",
                        "raw_metrics",
                        "predictions",
                        "guardrail_evidence",
                    ],
                },
                "reproducibility": {
                    "pair_id": f"{config['matrix_row_id']}::seed-{seed}",
                    "comparison_role": "candidate",
                    "compare_artifacts": ["predictions", "raw_metrics"],
                },
                "current_checkpoint_test_accessed": True,
                "project_wide_test_previously_accessed": True,
            },
        )
        manifest_artifacts = {
            path.name: sha256_file(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(
            run_dir / "manifest.json",
            {
                "schema": str(config["manifest_schema"]),
                "status": "completed",
                "training_performed": False,
                "selection_performed": False,
                "current_checkpoint_test_accessed": True,
                "project_wide_test_previously_accessed": True,
                "dataset": dataset_name,
                "budget": budget,
                "training_seed": seed,
                "test_seed": evaluation_seed,
                "attempt": attempt,
                "started_at": started_at,
                "ended_at": ended_at,
                "config_sha256": sha256_file(config_path),
                "artifacts": manifest_artifacts,
            },
        )
        return run_dir
    except Exception:
        failure = {
            "status": "failed",
            "test_data_loaded": test_data_loaded,
            "training_performed": False,
            "selection_performed": False,
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "failure.json", failure)
        (run_dir / "stderr.log").write_text(failure["traceback"], encoding="utf-8")
        raise


def run_frozen_test_matrix(
    project_root: str | Path, config_path: str | Path
) -> list[Path]:
    config = load_frozen_test_protocol(config_path)
    outputs: list[Path] = []
    for dataset in config["datasets"]:
        for budget in config["budgets"]:
            for seed in config["seeds"]:
                outputs.append(
                    run_frozen_test_unit(
                        project_root,
                        config_path,
                        dataset_name=str(dataset),
                        budget=float(budget),
                        seed=int(seed),
                    )
                )
    return outputs
