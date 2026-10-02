from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import yaml

from dap.direct_action_planning_dataset_validation.evaluation import (
    evaluate_methods,
)
from dap.direct_action_planning_dataset_validation.training import (
    collect_branch_dataset,
)
from dap.direct_action_planning_paper_evidence.data import (
    development_input_hashes,
    load_development_trace_dataset,
)
from dap.direct_action_planning_paper_evidence.training import (
    ForecasterWeights,
    train_forecaster,
)
from dap.utils.artifacts import (
    environment_record,
    sha256_file,
    sha256_tree,
    write_json,
)
from dap.utils.seed import set_global_seed

from .planning import make_scaled_planner
from .calibrated_protocol import (
    collect_calibrated_branch_dataset,
    evaluate_calibrated_methods,
)
from .environment import calibrate_domain_actions
from .selection import (
    PlanningCandidate,
    SelectionGuardrails,
    select_planning_candidate,
)
from .training import compute_value_target_scale, train_value_candidates


def candidate_grid(
    candidate_iterations: tuple[int, ...],
    continuation_weights: tuple[float, ...],
) -> dict[str, PlanningCandidate]:
    iterations = tuple(int(value) for value in candidate_iterations)
    weights = tuple(float(value) for value in continuation_weights)
    if len(set(iterations)) != len(iterations) or not iterations:
        raise ValueError("candidate_iterations must be non-empty and unique")
    if len(set(weights)) != len(weights) or not weights:
        raise ValueError("continuation_weights must be non-empty and unique")
    if not any(np.isclose(weight, 0.0) for weight in weights):
        raise ValueError("continuation_weights must include the immediate control 0")
    if any(not np.isfinite(weight) or not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError("continuation weights must be finite and in [0, 1]")
    grid = {"candidate_immediate": PlanningCandidate(-1, 0.0)}
    for iteration in sorted(iterations):
        for weight in sorted(weight for weight in weights if not np.isclose(weight, 0.0)):
            name = f"candidate_i{iteration:03d}_l{int(round(1000 * weight)):04d}"
            grid[name] = PlanningCandidate(iteration, weight)
    return grid


def load_protocol(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "tier",
        "manifest_schema",
        "datasets",
        "horizon",
        "budgets",
        "seeds",
        "seed_role",
        "gamma",
        "collection_episodes_per_domain",
        "validation_episodes_per_domain",
        "selection_episodes_per_domain",
        "evaluation_episodes_per_domain",
        "fvi_iterations",
        "candidate_iterations",
        "continuation_weights",
        "epochs_per_iteration",
        "model_epochs",
        "hidden_dim",
        "learning_rate",
        "model_validation_split",
        "selection_split",
        "evaluation_split",
        "selection_guardrails",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing protocol keys: {sorted(missing)}")
    iterations = int(config["fvi_iterations"])
    candidates = tuple(int(value) for value in config["candidate_iterations"])
    if iterations <= 0:
        raise ValueError("fvi_iterations must be positive")
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("candidate_iterations must be non-empty and unique")
    if any(value < 0 or value >= iterations for value in candidates):
        raise ValueError("candidate iteration is outside the trained FVI range")
    candidate_grid(candidates, tuple(float(value) for value in config["continuation_weights"]))
    allowed_splits = {
        "model_validation_split": "validation_fit",
        "selection_split": "validation_select",
        "evaluation_split": "validation_eval",
    }
    for field, expected in allowed_splits.items():
        if str(config[field]) != expected:
            raise ValueError(f"{field} must be {expected}; formal test access is forbidden")
    SelectionGuardrails(**config["selection_guardrails"])
    if str(config["seed_role"]) not in {"fixed_repro", "randomness_estimation"}:
        raise ValueError("seed_role must be fixed_repro or randomness_estimation")
    scale_mode = str(config.get("value_target_scale_mode", "unit"))
    if scale_mode not in {"unit", "train_reward_rms_sqrt_horizon"}:
        raise ValueError("value_target_scale_mode is not supported")
    if not isinstance(config.get("zero_initialize_value_output", False), bool):
        raise ValueError("zero_initialize_value_output must be boolean")
    capacity_mode = str(config.get("capacity_calibration_mode", "fixed_default"))
    if capacity_mode not in {"fixed_default", "train_quantile_domain"}:
        raise ValueError("capacity_calibration_mode is not supported")
    capacity_quantile = float(config.get("capacity_training_quantile", 0.95))
    if not np.isfinite(capacity_quantile) or not 0.0 < capacity_quantile <= 1.0:
        raise ValueError("capacity_training_quantile must be in (0, 1]")
    return config


def _run_dir(
    project_root: Path,
    config: dict,
    dataset: str,
    budget: float,
    seed: int,
    attempt: int,
) -> Path:
    base = f"{config['tier']}__{dataset}__b{budget:.0f}__s{seed}"
    run_id = base if attempt == 0 else f"{base}__a{attempt}"
    return (
        project_root
        / "results/direct_action_planning_dataset_specific_stabilization"
        / str(config["tier"])
        / dataset
        / run_id
    )


def _model_bytes(models: dict[int, torch.nn.Module], forecaster: torch.nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(
        {
            "values": {str(key): model.state_dict() for key, model in models.items()},
            "forecaster": forecaster.state_dict(),
        },
        buffer,
    )
    return int(buffer.tell())


def build_checkpoint_payload(
    models: dict[int, torch.nn.Module],
    forecaster: torch.nn.Module,
    *,
    normalizer_mean: np.ndarray,
    normalizer_scale: np.ndarray,
    value_output_scale: float = 1.0,
    zero_initialized_output: bool = False,
) -> dict[str, object]:
    """Return a checkpoint containing only weights-only-safe objects."""

    return {
        "values": {str(key): model.state_dict() for key, model in models.items()},
        "forecaster": forecaster.state_dict(),
        "normalizer_mean": torch.as_tensor(normalizer_mean, dtype=torch.float32),
        "normalizer_scale": torch.as_tensor(normalizer_scale, dtype=torch.float32),
        "value_output_scale": torch.as_tensor(
            float(value_output_scale), dtype=torch.float32
        ),
        "zero_initialized_output": torch.as_tensor(bool(zero_initialized_output)),
    }


def _bare_sha256(path: Path) -> str:
    return sha256_file(path).removeprefix("sha256:")


def _hashed_file(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": _bare_sha256(path)}


def _write_standard_bundle(
    run_dir: Path,
    *,
    root: Path,
    config: dict,
    dataset_name: str,
    seed: int,
    started_at: str,
    ended_at: str,
    metrics: pd.DataFrame,
    steps: pd.DataFrame,
    code_paths: tuple[Path, ...] | None = None,
) -> None:
    reproducible_environment = json.loads(
        (run_dir / "environment.json").read_text(encoding="utf-8")
    )
    reproducible_environment.pop("pid", None)
    reproducible_environment.pop("recorded_at", None)
    write_json(run_dir / "environment_repro.json", reproducible_environment)
    default_code_root = (
        root
        / "src/dap/direct_action_planning_dataset_specific_stabilization"
    )
    resolved_code_paths = code_paths or (default_code_root,)
    components = []
    for path in resolved_code_paths:
        resolved = path.resolve()
        components.append(
            {
                "path": str(resolved.relative_to(root)),
                "sha256": (
                    sha256_tree(resolved)
                    if resolved.is_dir()
                    else sha256_file(resolved)
                ),
            }
        )
    write_json(
        run_dir / "code_snapshot.json",
        {
            "source_root": str(default_code_root.relative_to(root)),
            "code_sha256": sha256_tree(default_code_root),
            "components": components,
            "git_commit": "UNAVAILABLE: workspace is not a Git repository",
        },
    )
    write_json(
        run_dir / "input_hashes.json",
        {
            "dataset": dataset_name,
            "development_arrays": development_input_hashes(root, dataset_name),
            "formal_test": "not_loaded_or_hashed",
        },
    )
    deterministic_metrics = metrics.drop(
        columns=["decision_ms_mean", "decision_ms_p95"], errors="ignore"
    )
    deterministic_metrics.to_json(
        run_dir / "raw_metrics.jsonl",
        orient="records",
        lines=True,
        force_ascii=True,
    )
    deterministic_steps = steps.drop(columns=["decision_ms"], errors="ignore")
    deterministic_steps.to_csv(run_dir / "predictions.csv", index=False)
    write_json(
        run_dir / "test_evidence.json",
        {
            "status": "PASS",
            "development_only": True,
            "formal_test_accessed": False,
            "evaluation_split": str(config["evaluation_split"]),
            "episode_rows": int(len(metrics)),
            "step_rows": int(len(steps)),
            "all_numeric_metrics_finite": bool(
                np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all()
            ),
        },
    )
    artifacts = {
        "stdout": _hashed_file(run_dir / "stdout.log"),
        "stderr": _hashed_file(run_dir / "stderr.log"),
        "raw_metrics": _hashed_file(run_dir / "raw_metrics.jsonl"),
        "predictions": _hashed_file(run_dir / "predictions.csv"),
        "test_evidence": _hashed_file(run_dir / "test_evidence.json"),
        "guardrail_evidence": _hashed_file(run_dir / "guardrails.json"),
        "failure": None,
    }
    write_json(
        run_dir / "run_manifest.v3.json",
        {
            "schema": "light.run_manifest.v3",
            "run_id": run_dir.name,
            "matrix_row": str(config.get("matrix_row_id", "STABILIZATION-CORE-V1")),
            "status": "completed",
            "termination": {
                "reason": "natural_exit",
                "exit_code": 0,
                "signal": None,
            },
            "seed": {"role": str(config["seed_role"]), "value": int(seed)},
            "command": [str(value) for value in sys.argv] or ["python"],
            "started_at": started_at,
            "ended_at": ended_at,
            "config": _hashed_file(run_dir / "config.json"),
            "environment": _hashed_file(run_dir / "environment_repro.json"),
            "code": {
                "commit": "UNAVAILABLE: workspace is not a Git repository",
                "dirty": False,
                "diff_sha256": None,
                "files": [_hashed_file(run_dir / "code_snapshot.json")],
            },
            "inputs": [
                {
                    **_hashed_file(run_dir / "input_hashes.json"),
                    "role": "data",
                    "source_revision": "chronological train and development validation roles",
                }
            ],
            "artifacts": artifacts,
            "completion": {
                "status": "PASS",
                "oracle": [
                    "all registered methods produced finite development metrics",
                    "hard budget overspend guardrail passed",
                    "formal test arrays were not loaded or evaluated",
                ],
                "evidence_artifacts": [
                    "test_evidence",
                    "raw_metrics",
                    "predictions",
                    "guardrail_evidence",
                ],
            },
            "reproducibility": {
                "pair_id": (
                    f"{config.get('matrix_row_id', 'STABILIZATION-CORE-V1')}::seed-{seed}"
                ),
                "comparison_role": "candidate",
                "compare_artifacts": ["predictions", "raw_metrics"],
            },
            "formal_test_accessed": False,
            "notes": ["Timing metrics are retained separately and excluded from exact replay artifacts."],
        },
    )


def _best_unshrunk(selection: pd.DataFrame) -> PlanningCandidate:
    candidates = selection[
        (selection.continuation_weight > 0.0)
        & np.isclose(selection.continuation_weight, 1.0)
    ]
    eligible = candidates[candidates.eligible]
    if eligible.empty:
        eligible = candidates
    best = eligible.sort_values(
        [
            "discounted_return",
            "completion_ratio",
            "slo_violation_rate",
            "total_cost",
            "candidate_iteration",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).iloc[0]
    return PlanningCandidate(int(best.candidate_iteration), 1.0)


def run_unit(
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
    config = load_protocol(config_path)
    if dataset_name not in tuple(str(value) for value in config["datasets"]):
        raise ValueError(f"dataset is not registered: {dataset_name}")
    if not any(np.isclose(float(budget), float(value)) for value in config["budgets"]):
        raise ValueError(f"budget is not registered: {budget}")
    if int(seed) not in tuple(int(value) for value in config["seeds"]):
        raise ValueError(f"seed is not registered: {seed}")
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    run_dir = _run_dir(root, config, dataset_name, float(budget), int(seed), attempt)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is append-only: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    resolved = {
        **config,
        "dataset": dataset_name,
        "budget": float(budget),
        "seed": int(seed),
    }
    write_json(run_dir / "config.json", resolved)
    write_json(run_dir / "environment.json", environment_record())
    (run_dir / "stdout.log").write_text("run started\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    set_global_seed(int(seed), torch_threads=1)
    wall_started = time.perf_counter()
    try:
        dataset = load_development_trace_dataset(
            root, dataset_name, horizon=int(config["horizon"])
        )
        capacity_mode = str(config.get("capacity_calibration_mode", "fixed_default"))
        capacity_quantile = float(config.get("capacity_training_quantile", 0.95))
        if capacity_mode == "train_quantile_domain":
            training = collect_calibrated_branch_dataset(
                dataset,
                split="train",
                horizon=int(config["horizon"]),
                budget=float(budget),
                episodes_per_domain=int(config["collection_episodes_per_domain"]),
                seed=int(seed),
                quantile=capacity_quantile,
            )
            validation = collect_calibrated_branch_dataset(
                dataset,
                split=str(config["model_validation_split"]),
                horizon=int(config["horizon"]),
                budget=float(budget),
                episodes_per_domain=int(config["validation_episodes_per_domain"]),
                seed=int(seed) + 10_000_019,
                quantile=capacity_quantile,
            )
            evaluate = evaluate_calibrated_methods
        else:
            training = collect_branch_dataset(
                dataset,
                split="train",
                horizon=int(config["horizon"]),
                budget=float(budget),
                episodes_per_domain=int(config["collection_episodes_per_domain"]),
                seed=int(seed),
            )
            validation = collect_branch_dataset(
                dataset,
                split=str(config["model_validation_split"]),
                horizon=int(config["horizon"]),
                budget=float(budget),
                episodes_per_domain=int(config["validation_episodes_per_domain"]),
                seed=int(seed) + 10_000_019,
            )
            evaluate = evaluate_methods
        scale_mode = str(config.get("value_target_scale_mode", "unit"))
        value_target_scale = (
            compute_value_target_scale(training, horizon=int(config["horizon"]))
            if scale_mode == "train_reward_rms_sqrt_horizon"
            else 1.0
        )
        candidates, value_history = train_value_candidates(
            training,
            validation,
            seed=int(seed),
            gamma=float(config["gamma"]),
            iterations=int(config["fvi_iterations"]),
            candidate_iterations=tuple(
                int(value) for value in config["candidate_iterations"]
            ),
            epochs_per_iteration=int(config["epochs_per_iteration"]),
            learning_rate=float(config["learning_rate"]),
            hidden_dim=int(config["hidden_dim"]),
            target_scale=value_target_scale,
            zero_initialize_output=bool(
                config.get("zero_initialize_value_output", False)
            ),
        )
        final_iteration = max(candidates)
        forecaster, forecast_history = train_forecaster(
            training,
            validation,
            candidates[final_iteration],
            seed=int(seed) + 1,
            gamma=float(config["gamma"]),
            epochs=int(config["model_epochs"]),
            weights=ForecasterWeights(load=1.0, planning_q=0.0, rank=0.0),
        )
        grid = candidate_grid(
            tuple(int(value) for value in config["candidate_iterations"]),
            tuple(float(value) for value in config["continuation_weights"]),
        )
        selection_methods = {}
        for name, candidate in grid.items():
            value_iteration = (
                final_iteration
                if candidate.candidate_iteration < 0
                else candidate.candidate_iteration
            )
            selection_methods[name] = make_scaled_planner(
                value=candidates[value_iteration],
                forecaster=forecaster,
                gamma=float(config["gamma"]),
                continuation_weight=candidate.continuation_weight,
            )
        selection_rows, _ = evaluate(
            dataset,
            selection_methods,
            split=str(config["selection_split"]),
            horizon=int(config["horizon"]),
            budget=float(budget),
            seed=int(seed) + int(config["selection_seed_offset"]),
            episodes_per_domain=int(config["selection_episodes_per_domain"]),
            gamma=float(config["gamma"]),
            **(
                {"quantile": capacity_quantile}
                if capacity_mode == "train_quantile_domain"
                else {}
            ),
        )
        for row in selection_rows:
            candidate = grid[str(row["method"])]
            row["candidate_iteration"] = candidate.candidate_iteration
            row["continuation_weight"] = candidate.continuation_weight
        guardrails = SelectionGuardrails(**config["selection_guardrails"])
        selected, selection_summary = select_planning_candidate(
            selection_rows, budget=float(budget), guardrails=guardrails
        )
        unshrunk = _best_unshrunk(selection_summary)
        selected_value_iteration = (
            final_iteration
            if selected.candidate_iteration < 0
            else selected.candidate_iteration
        )
        methods = {
            "dap_calibrated": make_scaled_planner(
                value=candidates[selected_value_iteration],
                forecaster=forecaster,
                gamma=float(config["gamma"]),
                continuation_weight=selected.continuation_weight,
            ),
            "dap_immediate": make_scaled_planner(
                value=candidates[final_iteration],
                forecaster=forecaster,
                gamma=float(config["gamma"]),
                continuation_weight=0.0,
            ),
            "dap_selected_unshrunk": make_scaled_planner(
                value=candidates[unshrunk.candidate_iteration],
                forecaster=forecaster,
                gamma=float(config["gamma"]),
                continuation_weight=1.0,
            ),
            "dap_forced_final": make_scaled_planner(
                value=candidates[final_iteration],
                forecaster=forecaster,
                gamma=float(config["gamma"]),
                continuation_weight=1.0,
            ),
        }
        episode_rows, step_rows = evaluate(
            dataset,
            methods,
            split=str(config["evaluation_split"]),
            horizon=int(config["horizon"]),
            budget=float(budget),
            seed=int(seed) + int(config["evaluation_seed_offset"]),
            episodes_per_domain=int(config["evaluation_episodes_per_domain"]),
            gamma=float(config["gamma"]),
            **(
                {"quantile": capacity_quantile}
                if capacity_mode == "train_quantile_domain"
                else {}
            ),
        )
        for row in episode_rows:
            row["training_seed"] = int(seed)
            row["selected_iteration"] = int(selected.candidate_iteration)
            row["selected_continuation_weight"] = float(
                selected.continuation_weight
            )
            row["selected_unshrunk_iteration"] = int(
                unshrunk.candidate_iteration
            )
        for row in step_rows:
            row["training_seed"] = int(seed)
            row["selected_iteration"] = int(selected.candidate_iteration)
            row["selected_continuation_weight"] = float(
                selected.continuation_weight
            )
        metrics = pd.DataFrame(episode_rows)
        steps = pd.DataFrame(step_rows)
        selection_metrics = pd.DataFrame(selection_rows)
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        steps.to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
        selection_metrics.to_csv(run_dir / "selection_metrics.csv", index=False)
        selection_summary.to_csv(run_dir / "selection_summary.csv", index=False)
        write_json(
            run_dir / "training.json",
            {"value_candidates": value_history, "state_forecaster": forecast_history},
        )
        write_json(
            run_dir / "diagnostics.json",
            {
                "selected": asdict(selected),
                "selected_unshrunk": asdict(unshrunk),
                "final_iteration": int(final_iteration),
                "training_states": int(training.n_states),
                "validation_states": int(validation.n_states),
                "candidate_count": int(len(grid)),
                "value_target_scale_mode": scale_mode,
                "value_target_scale": value_target_scale,
                "zero_initialize_value_output": bool(
                    config.get("zero_initialize_value_output", False)
                ),
                "capacity_calibration_mode": capacity_mode,
                "capacity_training_quantile": capacity_quantile,
                "domain_capacity_calibration": {
                    domain: asdict(
                        calibrate_domain_actions(
                            dataset.domains[domain]["train"],
                            quantile=capacity_quantile,
                            base_capacity=5.0,
                        )
                    )
                    for domain in dataset.domain_names
                },
            },
        )
        torch.save(
            build_checkpoint_payload(
                candidates,
                forecaster,
                normalizer_mean=candidates[final_iteration].normalizer.mean,
                normalizer_scale=candidates[final_iteration].normalizer.scale,
                value_output_scale=value_target_scale,
                zero_initialized_output=bool(
                    config.get("zero_initialize_value_output", False)
                ),
            ),
            run_dir / "models.pt",
        )
        runtime = {
            "wall_seconds": float(time.perf_counter() - wall_started),
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "value_parameter_count": int(
                sum(parameter.numel() for parameter in candidates[final_iteration].parameters())
            ),
            "forecaster_parameter_count": int(
                sum(parameter.numel() for parameter in forecaster.parameters())
            ),
            "checkpoint_bytes": _model_bytes(candidates, forecaster),
        }
        write_json(run_dir / "runtime.json", runtime)
        write_json(
            run_dir / "guardrails.json",
            {
                "budget_overspend_max": float(metrics.budget_overspend.max()),
                "budget_safe": bool(metrics.budget_overspend.max() <= 1.0e-8),
                "all_metrics_finite": bool(
                    np.isfinite(
                        metrics.select_dtypes(include=[np.number]).to_numpy()
                    ).all()
                ),
                "formal_test_accessed": False,
            },
        )
        (run_dir / "stdout.log").write_text("run completed\n", encoding="utf-8")
        ended_at = datetime.now(timezone.utc).isoformat()
        _write_standard_bundle(
            run_dir,
            root=root,
            config=config,
            dataset_name=dataset_name,
            seed=int(seed),
            started_at=started_at,
            ended_at=ended_at,
            metrics=metrics,
            steps=steps,
        )
        artifacts = {
            path.name: sha256_file(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(
            run_dir / "manifest.json",
            {
                "schema": str(config["manifest_schema"]),
                "status": "completed",
                "development_only": True,
                "formal_test_accessed": False,
                "dataset": dataset_name,
                "budget": float(budget),
                "training_seed": int(seed),
                "attempt": int(attempt),
                "started_at": started_at,
                "ended_at": ended_at,
                "split_roles": {
                    "model_fit": "train+validation_fit",
                    "selection": "validation_select",
                    "evaluation": "validation_eval",
                    "formal_test": "not_loaded",
                },
                "config_sha256": sha256_file(config_path),
                "code_sha256": sha256_tree(
                    root
                    / "src/dap/direct_action_planning_dataset_specific_stabilization"
                ),
                "input_hashes": development_input_hashes(root, dataset_name),
                "artifacts": artifacts,
            },
        )
        return run_dir
    except Exception:
        failure = {
            "status": "failed",
            "formal_test_accessed": False,
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "failure.json", failure)
        (run_dir / "stderr.log").write_text(failure["traceback"], encoding="utf-8")
        raise


def run_matrix(project_root: str | Path, config_path: str | Path) -> list[Path]:
    config = load_protocol(config_path)
    outputs: list[Path] = []
    for dataset in config["datasets"]:
        for budget in config["budgets"]:
            for seed in config["seeds"]:
                outputs.append(
                    run_unit(
                        project_root,
                        config_path,
                        dataset_name=str(dataset),
                        budget=float(budget),
                        seed=int(seed),
                    )
                )
    return outputs
