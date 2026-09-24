"""Development-only, same-information controls for DAP attribution."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import pandas as pd
import torch
import yaml

from stage2_dynamic_budget.direct_action_planning_paper_evidence.data import (
    load_development_trace_dataset,
)
from stage2_dynamic_budget.direct_action_planning_dataset_validation.models import (
    FeatureNormalizer,
)
from stage2_dynamic_budget.direct_action_planning_dataset_validation.training import (
    train_full_transition,
)
from stage2_dynamic_budget.direct_action_planning_paper_evidence.models import (
    EvidenceLoadForecaster,
)
from stage2_dynamic_budget.direct_action_planning_paper_evidence.planning import (
    make_evidence_planner,
)
from stage2_dynamic_budget.direct_action_planning_paper_evidence.training import (
    train_distilled_policy,
)
from stage2_dynamic_budget.utils.artifacts import (
    environment_record,
    sha256_file,
    sha256_tree,
    write_json,
)
from stage2_dynamic_budget.utils.seed import set_global_seed

from .calibrated_protocol import (
    collect_calibrated_branch_dataset,
    evaluate_calibrated_methods,
)
from .execution_lock import verify_code_lock
from .controls import feasible_actions, make_mpc_planner
from .models import ScaledEvidenceValueNetwork
from .planning import make_scaled_planner
from .training import compute_value_target_scale, train_value_candidates


BASE_CONTROL_METHODS = (
    "dap_full",
    "dap_immediate_structured",
    "mpc_2",
    "mpc_4",
    "mpc_8",
    "dap_black_box_transition",
    "dap_actor_distilled",
)
CONTROL_METHODS = BASE_CONTROL_METHODS + ("dap_no_budget_horizon",)


def load_control_protocol(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "tier", "manifest_schema", "matrix_row_id", "source_tier", "datasets",
        "budgets", "seeds", "horizon", "gamma", "capacity_training_quantile",
        "training_split", "validation_split", "evaluation_split", "collection_episodes_per_domain",
        "validation_episodes_per_domain", "evaluation_episodes_per_domain", "model_epochs",
        "hidden_dim", "evaluation_seed_offset", "dap_source_config",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"control protocol missing keys: {sorted(missing)}")
    if str(config["training_split"]) != "train" or str(config["validation_split"]) != "validation_fit":
        raise ValueError("controls may only train on train and validation_fit")
    if str(config["evaluation_split"]) != "validation_eval":
        raise ValueError("controls must be development-only")
    if tuple(config.get("methods", BASE_CONTROL_METHODS)) not in {BASE_CONTROL_METHODS, CONTROL_METHODS}:
        raise ValueError("control method order is not a registered family")
    if tuple(config.get("methods", BASE_CONTROL_METHODS)) == CONTROL_METHODS:
        for key in ("masked_value_iterations", "masked_value_candidate_iterations"):
            if key not in config:
                raise ValueError(f"extended controls require {key}")
    return config


def _source_dir(root: Path, tier: str, dataset: str, budget: float, seed: int) -> Path:
    return root / "results/direct_action_planning_paper_closure" / tier / dataset / f"{tier}__{dataset}__b{budget:.0f}__s{seed}"


def _run_dir(root: Path, config: dict, dataset: str, budget: float, seed: int, attempt: int) -> Path:
    name = f"{config['tier']}__{dataset}__b{budget:.0f}__s{seed}"
    if attempt:
        name += f"__a{attempt}"
    return root / "results/direct_action_planning_paper_closure" / str(config["tier"]) / dataset / name


def _load_source_components(source: Path, *, hidden_dim: int):
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("formal_test_accessed") is not False:
        raise ValueError(f"invalid development source: {source}")
    checkpoint = torch.load(source / "models.pt", weights_only=True, map_location="cpu")
    diagnostics = json.loads((source / "diagnostics.json").read_text(encoding="utf-8"))
    normalizer = FeatureNormalizer(
        mean=checkpoint["normalizer_mean"].cpu().numpy().astype(np.float64),
        scale=checkpoint["normalizer_scale"].cpu().numpy().astype(np.float64),
    )
    final_iteration = int(diagnostics["final_iteration"])
    selected = diagnostics["selected"]
    selected_iteration = int(selected["candidate_iteration"])
    selected_iteration = final_iteration if selected_iteration < 0 else selected_iteration
    scale = float(checkpoint["value_output_scale"].item())
    zero_init = bool(checkpoint["zero_initialized_output"].item())

    def load_value(iteration: int) -> ScaledEvidenceValueNetwork:
        value = ScaledEvidenceValueNetwork(
            normalizer, hidden_dim=hidden_dim, output_scale=scale,
            zero_initialize_output=zero_init,
        )
        value.load_state_dict(checkpoint["values"][str(iteration)], strict=True)
        return value.train(False)

    forecaster = EvidenceLoadForecaster(normalizer)
    forecaster.load_state_dict(checkpoint["forecaster"], strict=True)
    return load_value(selected_iteration), load_value(final_iteration), forecaster.train(False), selected


def _distilled_planner(policy):
    def plan(env, observation: np.ndarray):
        with torch.no_grad():
            logits = policy(torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1))[0].numpy()
        scores = np.asarray(logits, dtype=np.float64)
        feasible = feasible_actions(env)
        masked = np.full_like(scores, -np.inf, dtype=np.float64)
        masked[feasible] = scores[feasible]
        return int(np.argmax(masked)), masked
    return plan


def run_control_unit(project_root: str | Path, config_path: str | Path, *, dataset_name: str, budget: float, seed: int, attempt: int = 0) -> Path:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_control_protocol(config_path)
    verify_code_lock(root, config)
    if dataset_name not in tuple(config["datasets"]) or float(budget) not in tuple(float(v) for v in config["budgets"]) or int(seed) not in tuple(int(v) for v in config["seeds"]):
        raise ValueError("unregistered control cell")
    run_dir = _run_dir(root, config, dataset_name, float(budget), int(seed), attempt)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"control run directory is append-only: {run_dir}")
    source = _source_dir(root, str(config["source_tier"]), dataset_name, float(budget), int(seed))
    if not source.exists():
        raise FileNotFoundError(f"DAP source is unavailable: {source}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "config.json", {**config, "dataset": dataset_name, "budget": budget, "seed": seed})
    write_json(run_dir / "environment.json", environment_record())
    set_global_seed(int(seed), torch_threads=1)
    wall_started = time.perf_counter()
    try:
        dataset = load_development_trace_dataset(
            root, dataset_name, horizon=int(config["horizon"])
        )
        full_value, final_value, forecaster, selected = _load_source_components(source, hidden_dim=int(config["hidden_dim"]))
        train = collect_calibrated_branch_dataset(
            dataset, split=str(config["training_split"]), horizon=int(config["horizon"]), budget=float(budget),
            episodes_per_domain=int(config["collection_episodes_per_domain"]), seed=int(seed),
            quantile=float(config["capacity_training_quantile"]),
        )
        validation = collect_calibrated_branch_dataset(
            dataset, split=str(config["validation_split"]), horizon=int(config["horizon"]), budget=float(budget),
            episodes_per_domain=int(config["validation_episodes_per_domain"]), seed=int(seed) + 10_000_019,
            quantile=float(config["capacity_training_quantile"]),
        )
        transition, transition_history = train_full_transition(
            train, validation, full_value.normalizer, seed=int(seed) + 2,
            epochs=int(config["model_epochs"]),
        )
        distilled, distillation_history = train_distilled_policy(
            train, validation, full_value, forecaster,
            action_costs=np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float32),
            episode_budget=float(budget), gamma=float(config["gamma"]), seed=int(seed) + 4,
            hidden_dim=int(config["hidden_dim"]), epochs=int(config["model_epochs"]),
        )
        masked_candidates = {}
        masked_history = []
        masked_iteration = None
        if tuple(config.get("methods", BASE_CONTROL_METHODS)) == CONTROL_METHODS:
            feature_mask = np.ones(14, dtype=np.float32)
            feature_mask[-2:] = 0.0
            masked_candidates, masked_history = train_value_candidates(
                train, validation, seed=int(seed) + 7, gamma=float(config["gamma"]),
                iterations=int(config["masked_value_iterations"]),
                candidate_iterations=tuple(int(value) for value in config["masked_value_candidate_iterations"]),
                epochs_per_iteration=2, learning_rate=1.0e-3, hidden_dim=int(config["hidden_dim"]),
                target_scale=compute_value_target_scale(train, horizon=int(config["horizon"])),
                zero_initialize_output=True, feature_mask=feature_mask,
            )
            masked_iteration = int(selected["candidate_iteration"])
            masked_iteration = max(masked_candidates) if masked_iteration < 0 else masked_iteration
            if masked_iteration not in masked_candidates:
                raise ValueError("source-selected FVI round is absent from masked candidate grid")
        gamma = float(config["gamma"])
        all_methods = {
            "dap_full": make_scaled_planner(value=full_value, forecaster=forecaster, gamma=gamma, continuation_weight=float(selected["continuation_weight"])),
            "dap_immediate_structured": make_scaled_planner(value=final_value, forecaster=forecaster, gamma=gamma, continuation_weight=0.0),
            "mpc_2": make_mpc_planner(forecaster=forecaster, gamma=gamma, horizon=2),
            "mpc_4": make_mpc_planner(forecaster=forecaster, gamma=gamma, horizon=4),
            "mpc_8": make_mpc_planner(forecaster=forecaster, gamma=gamma, horizon=8),
            "dap_black_box_transition": make_evidence_planner(value=full_value, forecaster=forecaster, gamma=gamma, mode="full_transition", full_transition=transition),
            "dap_actor_distilled": _distilled_planner(distilled.train(False)),
        }
        if masked_candidates:
            all_methods["dap_no_budget_horizon"] = make_scaled_planner(value=masked_candidates[masked_iteration], forecaster=forecaster, gamma=gamma, continuation_weight=float(selected["continuation_weight"]))
        methods = {name: all_methods[name] for name in tuple(config.get("methods", BASE_CONTROL_METHODS))}
        episodes, steps = evaluate_calibrated_methods(
            dataset, methods, split=str(config["evaluation_split"]), horizon=int(config["horizon"]), budget=float(budget),
            seed=int(seed) + int(config["evaluation_seed_offset"]), episodes_per_domain=int(config["evaluation_episodes_per_domain"]),
            gamma=gamma, quantile=float(config["capacity_training_quantile"]),
        )
        metrics = pd.DataFrame(episodes)
        step_frame = pd.DataFrame(steps)
        metrics["training_seed"] = int(seed)
        step_frame["training_seed"] = int(seed)
        if tuple(sorted(metrics.method.unique())) != tuple(sorted(methods)):
            raise AssertionError("control method set is incomplete")
        if not np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError("non-finite control metrics")
        if float(metrics.budget_overspend.max()) > 1.0e-8:
            raise AssertionError("control violated hard budget")
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        step_frame.to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
        write_json(run_dir / "training.json", {"transition": transition_history, "distillation": distillation_history, "masked_value": masked_history, "masked_feature_indices": [12, 13], "masked_iteration": masked_iteration, "train_states": train.n_states, "validation_states": validation.n_states})
        write_json(run_dir / "source.json", {"source_run": str(source.relative_to(root)), "source_models_sha256": sha256_file(source / "models.pt"), "source_manifest_sha256": sha256_file(source / "manifest.json")})
        write_json(run_dir / "guardrails.json", {"budget_overspend_max": float(metrics.budget_overspend.max()), "budget_safe": True, "all_metrics_finite": True, "formal_test_accessed": False})
        ended_at = datetime.now(timezone.utc).isoformat()
        write_json(run_dir / "runtime.json", {"wall_seconds": time.perf_counter() - wall_started, "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)})
        code_root = root / "src/stage2_dynamic_budget/direct_action_planning_paper_closure"
        write_json(run_dir / "code_snapshot.json", {"source": str(code_root.relative_to(root)), "sha256": sha256_tree(code_root)})
        artifacts = {path.name: sha256_file(path) for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "manifest.json"}
        write_json(run_dir / "manifest.json", {"schema": str(config["manifest_schema"]), "status": "completed", "development_only": True, "formal_test_accessed": False, "dataset": dataset_name, "budget": float(budget), "training_seed": int(seed), "attempt": attempt, "started_at": started_at, "ended_at": ended_at, "config_sha256": sha256_file(config_path), "artifacts": artifacts})
        return run_dir
    except Exception:
        write_json(run_dir / "failure.json", {"status": "failed", "formal_test_accessed": False, "traceback": traceback.format_exc()})
        raise


def run_control_matrix(project_root: str | Path, config_path: str | Path) -> list[Path]:
    config = load_control_protocol(config_path)
    outputs = []
    for dataset in config["datasets"]:
        for budget in config["budgets"]:
            for seed in config["seeds"]:
                outputs.append(run_control_unit(project_root, config_path, dataset_name=str(dataset), budget=float(budget), seed=int(seed)))
    return outputs
