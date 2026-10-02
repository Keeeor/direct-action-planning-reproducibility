from __future__ import annotations

import io
from datetime import datetime, timezone
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

from dap.direct_action_planning_dataset_validation.data import (
    DATASET_SPECS,
    make_trace_env,
)
from dap.direct_action_planning_dataset_validation.evaluation import (
    evaluate_methods,
)
from dap.direct_action_planning_dataset_validation.experiment import (
    select_refresh_candidate,
)
from dap.utils.artifacts import (
    environment_record,
    sha256_file,
    sha256_tree,
    write_json,
)
from dap.utils.seed import set_global_seed

from .data import development_input_hashes, load_development_trace_dataset
from .models import parameter_count
from .planning import make_evidence_planner
from .training import (
    forecaster_metrics,
    training_anchor_states,
    train_distilled_policy,
    train_paper_components,
)


CORE_METHODS = (
    "dap_full",
    "dap_direct_base",
    "dap_refresh_anchor_raw",
    "dap_refresh_no_anchor",
    "dap_no_budget_horizon",
    "dap_state_only_forecast",
    "dap_persistence_forecast",
    "dap_full_transition",
    "dap_immediate_reward",
    "dap_policy_distilled",
    "oracle_next_load",
)

FULL_TRAINING_COMPONENTS = (
    "d0_collection",
    "model_validation_collection",
    "base_value",
    "decision_forecaster",
    "on_policy_collection",
    "value_refresh_anchor",
)

STANDARD_MANIFEST_NAME = "run_manifest.v3.json"


def _registered_float(value: float, registered: object) -> bool:
    return any(np.isclose(float(value), float(candidate)) for candidate in registered)


def validate_core_request(
    config: dict,
    *,
    dataset_name: str,
    budget: float,
    seed: int,
) -> None:
    """Fail closed when a CLI request is outside the frozen core grid."""

    datasets = tuple(str(name) for name in config.get("datasets", ()))
    budgets = tuple(float(value) for value in config.get("budgets", ()))
    seeds = tuple(int(value) for value in config.get("seeds", ()))
    if dataset_name not in datasets:
        raise ValueError(f"dataset is not registered: {dataset_name}")
    if not _registered_float(budget, budgets):
        raise ValueError(f"budget is not registered: {budget}")
    if int(seed) not in seeds:
        raise ValueError(f"seed is not registered: {seed}")


def validate_sensitivity_request(
    config: dict,
    *,
    factor: str,
    level: float,
    seed: int,
) -> None:
    """Fail closed when a sensitivity request is outside the frozen levels."""

    sensitivity = config.get("sensitivity", {})
    if factor not in sensitivity:
        raise ValueError(f"sensitivity factor is not registered: {factor}")
    if not _registered_float(level, sensitivity[factor]):
        raise ValueError(f"sensitivity level is not registered: {factor}={level}")
    validate_core_request(
        config,
        dataset_name=str(config.get("sensitivity_dataset", "gentd26")),
        budget=float(config.get("sensitivity_budget", 96.0)),
        seed=seed,
    )


def _core_run_dir(
    project_root: Path,
    config: dict,
    *,
    dataset_name: str,
    budget: float,
    seed: int,
    attempt: int,
) -> Path:
    base_run_id = f"{config['tier']}__{dataset_name}__b{budget:.0f}__s{seed}"
    run_id = base_run_id if attempt == 0 else f"{base_run_id}__a{attempt}"
    return (
        project_root
        / "results/direct_action_planning_paper_evidence"
        / str(config["tier"])
        / dataset_name
        / run_id
    )


def _sensitivity_run_dir(
    project_root: Path,
    config: dict,
    *,
    factor: str,
    level: float,
    dataset_name: str,
    seed: int,
    attempt: int,
) -> Path:
    level_label = str(level).replace(".", "p")
    base_run_id = (
        f"{config['sensitivity_tier']}__{factor}-{level_label}__"
        f"{dataset_name}__s{seed}"
    )
    run_id = base_run_id if attempt == 0 else f"{base_run_id}__a{attempt}"
    return (
        project_root
        / "results/direct_action_planning_paper_evidence"
        / str(config["sensitivity_tier"])
        / factor
        / run_id
    )


def load_protocol(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "tier",
        "datasets",
        "horizon",
        "budgets",
        "seeds",
        "gamma",
        "collection_episodes_per_domain",
        "validation_episodes_per_domain",
        "evaluation_episodes_per_domain",
        "fvi_iterations",
        "refresh_iterations",
        "model_epochs",
        "hidden_dim",
        "anchor_weight",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing protocol keys: {sorted(missing)}")
    if config.get("evaluation_split", "validation_eval") not in {
        "validation",
        "validation_eval",
    }:
        raise ValueError("paper-evidence experiments are validation-only")
    seed_role = str(config.get("seed_role", "randomness_estimation"))
    if seed_role not in {"fixed_repro", "randomness_estimation"}:
        raise ValueError(f"invalid seed_role: {seed_role}")
    return config


def _model_bytes(model: torch.nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return int(buffer.tell())


def full_method_training_seconds(component_seconds: dict[str, float]) -> float:
    missing = set(FULL_TRAINING_COMPONENTS) - set(component_seconds)
    if missing:
        raise ValueError(f"missing full-method timing components: {sorted(missing)}")
    return float(sum(component_seconds[name] for name in FULL_TRAINING_COMPONENTS))


def attach_seed_roles(rows: list[dict], *, training_seed: int) -> None:
    """Preserve the evaluation RNG seed while naming the independent training unit."""

    for row in rows:
        row["evaluation_seed"] = int(row["seed"])
        row["training_seed"] = int(training_seed)


def core_value_variants(artifacts) -> dict[str, object]:
    """Bind each value ablation to exactly one registered component change."""

    if artifacts.no_anchor_value is None or artifacts.no_budget_horizon_value is None:
        raise ValueError("core ablations were not trained")
    return {
        "dap_full": artifacts.refreshed_value,
        "dap_direct_base": artifacts.base_value,
        "dap_refresh_anchor_raw": artifacts.refreshed_value,
        "dap_refresh_no_anchor": artifacts.no_anchor_value,
        "dap_no_budget_horizon": artifacts.no_budget_horizon_value,
    }


def build_checkpoint_payload(
    artifacts,
    distilled: torch.nn.Module,
    *,
    action_costs: np.ndarray,
    budget: float,
    horizon: int,
    hidden_dim: int,
    diagnostic_selected_value_variant: str = "not_run",
) -> dict:
    if (
        artifacts.no_anchor_value is None
        or artifacts.no_budget_horizon_base_value is None
        or artifacts.no_budget_horizon_value is None
        or artifacts.state_forecaster is None
        or artifacts.full_transition is None
    ):
        raise ValueError("complete core artifacts are required for the core checkpoint")
    full_value = artifacts.refreshed_value
    return {
        "metadata": {
            "observation_dim": 14,
            "action_costs": torch.as_tensor(action_costs),
            "budget": float(budget),
            "horizon": int(horizon),
            "hidden_dim": int(hidden_dim),
            "value_normalizer_mean": torch.as_tensor(
                full_value.normalizer.mean, dtype=torch.float64
            ),
            "value_normalizer_scale": torch.as_tensor(
                full_value.normalizer.scale, dtype=torch.float64
            ),
            "full_value_variant": "refreshed_value",
            "diagnostic_selected_value_variant": diagnostic_selected_value_variant,
        },
        "base_value": artifacts.base_value.state_dict(),
        "refreshed_value": artifacts.refreshed_value.state_dict(),
        "no_anchor_value": artifacts.no_anchor_value.state_dict(),
        "no_budget_horizon_base_value": (
            artifacts.no_budget_horizon_base_value.state_dict()
        ),
        "no_budget_horizon_value": artifacts.no_budget_horizon_value.state_dict(),
        "decision_forecaster": artifacts.decision_forecaster.state_dict(),
        "state_forecaster": artifacts.state_forecaster.state_dict(),
        "full_transition": artifacts.full_transition.state_dict(),
        "distilled_policy": distilled.state_dict(),
    }


def _selection(
    dataset,
    artifacts,
    *,
    config: dict,
    budget: float,
    seed: int,
) -> tuple[str, object, list[dict]]:
    gamma = float(config["gamma"])
    evaluation_gamma = float(config.get("evaluation_gamma", 0.99))
    base = make_evidence_planner(
        value=artifacts.base_value,
        forecaster=artifacts.decision_forecaster,
        gamma=gamma,
    )
    refreshed = make_evidence_planner(
        value=artifacts.refreshed_value,
        forecaster=artifacts.decision_forecaster,
        gamma=gamma,
    )
    rows, _ = evaluate_methods(
        dataset,
        {"structured_dap": base, "structured_dap_refresh": refreshed},
        split=str(config.get("selection_split", "validation_select")),
        horizon=int(config["horizon"]),
        budget=budget,
        seed=seed + int(config.get("selection_seed_offset", 60_000_011)),
        episodes_per_domain=int(config.get("selection_episodes_per_domain", 2)),
        gamma=evaluation_gamma,
    )
    selected = select_refresh_candidate(rows)
    value = (
        artifacts.refreshed_value if selected == "refreshed_value" else artifacts.base_value
    )
    return selected, value, rows


def _train(
    dataset,
    config: dict,
    *,
    budget: float,
    seed: int,
    include_ablations: bool,
):
    return train_paper_components(
        dataset,
        horizon=int(config["horizon"]),
        budget=budget,
        seed=seed,
        gamma=float(config["gamma"]),
        collection_episodes=int(config["collection_episodes_per_domain"]),
        validation_episodes=int(config["validation_episodes_per_domain"]),
        fvi_iterations=int(config["fvi_iterations"]),
        refresh_iterations=int(config["refresh_iterations"]),
        model_epochs=int(config["model_epochs"]),
        hidden_dim=int(config["hidden_dim"]),
        anchor_weight=float(config["anchor_weight"]),
        checkpoint_selection=str(
            config.get("fvi_checkpoint_selection", "validation_bellman_residual")
        ),
        validation_split=str(config.get("model_validation_split", "validation_fit")),
        include_ablations=include_ablations,
    )


def _write_manifest(
    run_dir: Path,
    config_path: Path,
    schema: str,
    *,
    project_root: Path,
    dataset_name: str,
    budget: float,
    seed: int,
    started_at: str,
    matrix_row_id: str,
    run_id: str,
    attempt: int,
) -> None:
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    write_json(
        run_dir / "manifest.json",
        {
            "schema": schema,
            "status": "completed",
            "development_only": True,
            "formal_test_accessed": False,
            "matrix_row_id": matrix_row_id,
            "run_id": run_id,
            "attempt": int(attempt),
            "seed_role": "randomness_estimation",
            "training_seed": int(seed),
            "budget": float(budget),
            "dataset": dataset_name,
            "split_roles": {
                "model_validation": "validation_fit",
                "selection": "validation_select",
                "development_evaluation": "validation_eval",
                "formal_test": "not_loaded",
            },
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "argv": list(sys.argv),
            "config_sha256": sha256_file(config_path),
            "code_sha256": sha256_tree(project_root / "src"),
            "input_hashes": development_input_hashes(project_root, dataset_name),
            "artifacts": artifacts,
        },
    )


def _bare_sha256(path: Path) -> str:
    return sha256_file(path).removeprefix("sha256:")


def _hashed_record(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": _bare_sha256(path)}


def _write_raw_jsonl(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for row in frame.to_dict(orient="records")
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mark_legacy_manifest_failed(run_dir: Path) -> None:
    """Keep the legacy analysis reader from accepting a failed bundle."""

    legacy_path = run_dir / "manifest.json"
    if not legacy_path.exists():
        return
    try:
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A malformed legacy manifest is already rejected by the analysis reader;
        # preserve the original failure and let the standard manifest carry it.
        return
    if legacy.get("status") == "completed":
        legacy["status"] = "failed"
        legacy["completion"] = {
            "status": "NOT_EVALUATED",
            "oracle": [],
            "evidence_artifacts": [],
        }
        legacy["failure"] = {
            "path": "failure.json",
            "reason": "standard run bundle failed",
        }
        write_json(legacy_path, legacy)


def _write_standard_bundle(
    run_dir: Path,
    *,
    project_root: Path,
    config: dict,
    dataset_name: str,
    budget: float,
    seed: int,
    matrix_row: str,
    started_at: str,
    status: str,
    required_methods: tuple[str, ...],
    error_traceback: str | None = None,
) -> None:
    """Emit a standard evaluation bundle alongside the legacy analysis manifest."""

    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = run_dir / "config.json"
    if not resolved_config.exists():
        write_json(
            resolved_config,
            {
                **config,
                "dataset": dataset_name,
                "budget": float(budget),
                "seed": int(seed),
            },
        )
    environment_path = run_dir / "environment.json"
    if not environment_path.exists():
        write_json(environment_path, environment_record())

    reproducible_environment = environment_record()
    reproducible_environment.pop("pid", None)
    reproducible_environment.pop("recorded_at", None)
    environment_repro_path = run_dir / "environment_repro.json"
    write_json(environment_repro_path, reproducible_environment)

    code_snapshot_path = run_dir / "code_snapshot.json"
    write_json(
        code_snapshot_path,
        {
            "source_root": "src",
            "code_sha256": sha256_tree(project_root / "src"),
            "git_commit": "UNAVAILABLE: workspace is not a Git repository",
        },
    )
    input_hash_path = run_dir / "input_hashes.json"
    write_json(
        input_hash_path,
        {
            "dataset": dataset_name,
            "source_revision": str(DATASET_SPECS[dataset_name]["split_contract"]),
            "development_arrays": development_input_hashes(project_root, dataset_name),
            "formal_test_array": "not_loaded_or_hashed",
        },
    )

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    test_evidence_path = run_dir / "test_evidence.json"
    failure_path = run_dir / "failure.json"
    stdout_path.write_text(
        "run completed\n" if status == "completed" else "run failed\n",
        encoding="utf-8",
    )
    stderr_path.write_text(error_traceback or "", encoding="utf-8")

    artifacts: dict[str, dict[str, str] | None] = {
        "stdout": _hashed_record(stdout_path),
        "stderr": _hashed_record(stderr_path),
        "raw_metrics": None,
        "predictions": None,
        "test_evidence": None,
        "guardrail_evidence": None,
        "failure": None,
    }
    completion = {
        "status": "NOT_EVALUATED",
        "oracle": [],
        "evidence_artifacts": [],
    }
    termination = {
        "reason": "error",
        "exit_code": 1,
        "signal": None,
    }

    if status == "completed":
        metrics = pd.read_csv(run_dir / "metrics_deterministic.csv")
        numeric = metrics.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        all_finite = bool(np.isfinite(numeric).all())
        observed_methods = tuple(sorted(str(value) for value in metrics.method.unique()))
        methods_complete = set(required_methods) == set(observed_methods)
        max_overspend = float(metrics.budget_overspend.max())
        budget_safe = max_overspend <= 1.0e-8
        if not all_finite or not methods_complete or not budget_safe:
            raise AssertionError(
                "completed run failed raw-bundle oracle: "
                f"finite={all_finite}, methods={methods_complete}, budget={budget_safe}"
            )

        raw_metrics_path = run_dir / "raw_metrics.jsonl"
        _write_raw_jsonl(metrics, raw_metrics_path)
        steps = pd.read_csv(run_dir / "steps_deterministic.csv.gz")
        prediction_columns = [
            column
            for column in (
                "dataset",
                "domain",
                "method",
                "budget",
                "training_seed",
                "evaluation_seed",
                "episode",
                "window_seed",
                "window_start",
                "step",
                "action",
            )
            if column in steps.columns
        ]
        predictions_path = run_dir / "predictions.csv"
        steps[prediction_columns].to_csv(predictions_path, index=False)
        write_json(
            test_evidence_path,
            {
                "status": "PASS",
                "development_only": True,
                "formal_test_accessed": False,
                "evaluation_split": str(config.get("evaluation_split", "validation_eval")),
                "episode_rows": int(len(metrics)),
                "step_rows": int(len(steps)),
                "methods": list(observed_methods),
                "all_numeric_metrics_finite": all_finite,
            },
        )
        guardrails_path = run_dir / "guardrails.json"
        write_json(
            guardrails_path,
            {
                "status": "PASS",
                "formal_test_accessed": False,
                "budget_overspend_max": max_overspend,
                "budget_safety_pass": budget_safe,
                "all_numeric_metrics_finite": all_finite,
                "registered_methods_complete": methods_complete,
                "analysis_level_service_cost_guardrails": "pending_result_analysis",
            },
        )
        artifacts.update(
            {
                "raw_metrics": _hashed_record(raw_metrics_path),
                "predictions": _hashed_record(predictions_path),
                "test_evidence": _hashed_record(test_evidence_path),
                "guardrail_evidence": _hashed_record(guardrails_path),
            }
        )
        completion = {
            "status": "PASS",
            "oracle": [
                "all registered methods produced finite development metrics",
                "hard budget overspend guardrail passed",
                "formal test split was not loaded or evaluated",
            ],
            "evidence_artifacts": [
                "test_evidence",
                "raw_metrics",
                "predictions",
                "guardrail_evidence",
            ],
        }
        termination = {
            "reason": "natural_exit",
            "exit_code": 0,
            "signal": None,
        }
    else:
        write_json(
            test_evidence_path,
            {
                "status": "NOT_EVALUATED",
                "development_only": True,
                "formal_test_accessed": False,
            },
        )
        write_json(
            failure_path,
            {
                "status": "failed",
                "stage": "experiment_unit",
                "exception_traceback": error_traceback or "unknown error",
                "exit_code": 1,
            },
        )
        _mark_legacy_manifest_failed(run_dir)
        artifacts["test_evidence"] = _hashed_record(test_evidence_path)
        artifacts["failure"] = _hashed_record(failure_path)

    seed_role = str(config.get("seed_role", "randomness_estimation"))
    if seed_role not in {"fixed_repro", "randomness_estimation"}:
        raise ValueError(f"invalid seed_role: {seed_role}")
    write_json(
        run_dir / STANDARD_MANIFEST_NAME,
        {
            "schema": "light.run_manifest.v3",
            "run_id": run_dir.name,
            "matrix_row": matrix_row,
            "status": status,
            "termination": termination,
            "seed": {"role": seed_role, "value": int(seed)},
            "command": [str(value) for value in sys.argv] or ["python"],
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "config": _hashed_record(resolved_config),
            "environment": _hashed_record(environment_repro_path),
            "code": {
                "commit": "UNAVAILABLE: workspace is not a Git repository",
                "dirty": False,
                "diff_sha256": None,
                "files": [_hashed_record(code_snapshot_path)],
            },
            "inputs": [
                {
                    **_hashed_record(input_hash_path),
                    "role": "data",
                    "source_revision": str(
                        DATASET_SPECS[dataset_name]["split_contract"]
                    ),
                }
            ],
            "artifacts": artifacts,
            "completion": completion,
            "reproducibility": {
                "pair_id": f"{matrix_row}::seed-{seed}",
                "comparison_role": (
                    "candidate" if seed_role == "fixed_repro" else "not_applicable"
                ),
                "compare_artifacts": ["predictions", "raw_metrics"],
            },
            "formal_test_accessed": False,
            "notes": [
                "Legacy manifest.json is retained for the registered analysis reader."
            ],
        },
    )


def _run_core_unit_impl(
    project_root: str | Path,
    config_path: str | Path,
    *,
    dataset_name: str,
    budget: float,
    seed: int,
    attempt: int = 0,
) -> Path:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_protocol(config_path)
    validate_core_request(
        config, dataset_name=dataset_name, budget=budget, seed=seed
    )
    dataset = load_development_trace_dataset(
        project_root, dataset_name, horizon=int(config["horizon"])
    )
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    run_dir = _core_run_dir(
        project_root,
        config,
        dataset_name=dataset_name,
        budget=budget,
        seed=seed,
        attempt=attempt,
    )
    run_id = run_dir.name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is append-only: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        **config,
        "dataset": dataset_name,
        "budget": budget,
        "seed": seed,
    }
    write_json(run_dir / "config.json", resolved)
    write_json(run_dir / "environment.json", environment_record())
    set_global_seed(seed, torch_threads=1)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    artifacts = _train(
        dataset, config, budget=budget, seed=seed, include_ablations=True
    )
    selection_started = time.perf_counter()
    selected, _, selection_rows = _selection(
        dataset, artifacts, config=config, budget=budget, seed=seed
    )
    selection_seconds = time.perf_counter() - selection_started
    probe_env, _ = make_trace_env(
        dataset,
        dataset.domain_names[0],
        "train",
        horizon=int(config["horizon"]),
        budget=budget,
        window_seed=seed,
    )
    action_costs = probe_env.action_costs.astype(np.float32, copy=True)
    distillation_started = time.perf_counter()
    distilled, distillation_history = train_distilled_policy(
        artifacts.combined_data,
        artifacts.validation_data,
        artifacts.refreshed_value,
        artifacts.decision_forecaster,
        action_costs=action_costs,
        episode_budget=budget,
        gamma=float(config["gamma"]),
        seed=seed + 4,
        hidden_dim=int(config["hidden_dim"]),
        epochs=int(config["model_epochs"]),
    )
    distillation_seconds = time.perf_counter() - distillation_started
    gamma = float(config["gamma"])
    value_variants = core_value_variants(artifacts)
    methods = {
        "dap_full": make_evidence_planner(
            value=value_variants["dap_full"],
            forecaster=artifacts.decision_forecaster,
            gamma=gamma,
        ),
        "dap_direct_base": make_evidence_planner(
            value=value_variants["dap_direct_base"],
            forecaster=artifacts.decision_forecaster,
            gamma=gamma,
        ),
        "dap_refresh_anchor_raw": make_evidence_planner(
            value=value_variants["dap_refresh_anchor_raw"],
            forecaster=artifacts.decision_forecaster,
            gamma=gamma,
        ),
        "dap_refresh_no_anchor": make_evidence_planner(
            value=value_variants["dap_refresh_no_anchor"],
            forecaster=artifacts.decision_forecaster,
            gamma=gamma,
        ),
        "dap_no_budget_horizon": make_evidence_planner(
            value=value_variants["dap_no_budget_horizon"],
            forecaster=artifacts.decision_forecaster,
            gamma=gamma,
        ),
        "dap_state_only_forecast": make_evidence_planner(
            value=value_variants["dap_full"],
            forecaster=artifacts.state_forecaster,
            gamma=gamma,
        ),
        "dap_persistence_forecast": make_evidence_planner(
            value=value_variants["dap_full"],
            forecaster=artifacts.decision_forecaster,
            gamma=gamma,
            mode="persistence",
        ),
        "dap_full_transition": make_evidence_planner(
            value=value_variants["dap_full"],
            forecaster=artifacts.decision_forecaster,
            gamma=gamma,
            mode="full_transition",
            full_transition=artifacts.full_transition,
        ),
        "dap_immediate_reward": make_evidence_planner(
            value=value_variants["dap_full"],
            forecaster=artifacts.decision_forecaster,
            gamma=gamma,
            use_continuation=False,
        ),
        "dap_policy_distilled": distilled,
        "oracle_next_load": make_evidence_planner(
            value=value_variants["dap_full"],
            forecaster=artifacts.decision_forecaster,
            gamma=gamma,
            mode="oracle_next_load",
        ),
    }
    episode_rows, step_rows = evaluate_methods(
        dataset,
        methods,
        split=str(config.get("evaluation_split", "validation_eval")),
        horizon=int(config["horizon"]),
        budget=budget,
        seed=seed + int(config.get("evaluation_seed_offset", 70_000_013)),
        episodes_per_domain=int(config["evaluation_episodes_per_domain"]),
        gamma=float(config.get("evaluation_gamma", 0.99)),
    )
    attach_seed_roles(episode_rows, training_seed=seed)
    attach_seed_roles(step_rows, training_seed=seed)
    episode_frame = pd.DataFrame(episode_rows)
    step_frame = pd.DataFrame(step_rows)
    episode_frame.to_csv(run_dir / "metrics.csv", index=False)
    step_frame.to_csv(
        run_dir / "steps.csv.gz", index=False, compression="gzip"
    )
    episode_frame.drop(
        columns=["decision_ms_mean", "decision_ms_p95"], errors="ignore"
    ).to_csv(run_dir / "metrics_deterministic.csv", index=False)
    step_frame.drop(columns=["decision_ms"], errors="ignore").to_csv(
        run_dir / "steps_deterministic.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    selection_frame = pd.DataFrame(selection_rows)
    selection_frame.to_csv(run_dir / "selection_metrics.csv", index=False)
    selection_frame.drop(
        columns=["decision_ms_mean", "decision_ms_p95"], errors="ignore"
    ).to_csv(run_dir / "selection_metrics_deterministic.csv", index=False)

    anchors = training_anchor_states(artifacts.training_data)
    base_anchor = artifacts.base_value.predict(anchors)
    anchor_diagnostics = {
        "selected_value": selected,
        "anchored_refresh_mae_from_base": float(
            np.mean(np.abs(artifacts.refreshed_value.predict(anchors) - base_anchor))
        ),
        "no_anchor_refresh_mae_from_base": float(
            np.mean(np.abs(artifacts.no_anchor_value.predict(anchors) - base_anchor))
        ),
        "decision_forecaster": forecaster_metrics(
            artifacts.decision_forecaster,
            artifacts.base_value,
            artifacts.validation_data,
            gamma,
        ),
        "state_forecaster": forecaster_metrics(
            artifacts.state_forecaster,
            artifacts.base_value,
            artifacts.validation_data,
            gamma,
        ),
    }
    write_json(run_dir / "diagnostics.json", anchor_diagnostics)
    write_json(
        run_dir / "training.json",
        {
            "training_seconds": artifacts.training_seconds,
            "full_method_training_seconds": full_method_training_seconds(
                artifacts.component_seconds
            ),
            "component_seconds": {
                **artifacts.component_seconds,
                "diagnostic_selection": selection_seconds,
                "distilled_policy": distillation_seconds,
            },
            "histories": {**artifacts.histories, "distillation": distillation_history},
            "collection": artifacts.collection,
            "parameters": {
                "full_value": parameter_count(artifacts.refreshed_value),
                "decision_forecaster": parameter_count(artifacts.decision_forecaster),
                "full_transition": parameter_count(artifacts.full_transition),
                "distilled_policy": parameter_count(distilled),
            },
            "checkpoint_bytes": {
                "full_value": _model_bytes(artifacts.refreshed_value),
                "decision_forecaster": _model_bytes(artifacts.decision_forecaster),
                "full_transition": _model_bytes(artifacts.full_transition),
                "distilled_policy": _model_bytes(distilled),
            },
        },
    )
    torch.save(
        build_checkpoint_payload(
            artifacts,
            distilled,
            action_costs=action_costs,
            budget=budget,
            horizon=int(config["horizon"]),
            hidden_dim=int(config["hidden_dim"]),
            diagnostic_selected_value_variant=selected,
        ),
        run_dir / "models.pt",
    )
    write_json(
        run_dir / "runtime.json",
        {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    )
    _write_manifest(
        run_dir,
        config_path,
        str(config.get("core_manifest_schema", "dap.dap_paper_evidence.core.v1")),
        project_root=project_root,
        dataset_name=dataset_name,
        budget=budget,
        seed=seed,
        started_at=started_at,
        matrix_row_id=f"CORE::{dataset_name}::b{budget:.0f}::s{seed}",
        run_id=run_id,
        attempt=attempt,
    )
    return run_dir


def run_core_unit(
    project_root: str | Path,
    config_path: str | Path,
    *,
    dataset_name: str,
    budget: float,
    seed: int,
    attempt: int = 0,
) -> Path:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_protocol(config_path)
    validate_core_request(
        config, dataset_name=dataset_name, budget=budget, seed=seed
    )
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    run_dir = _core_run_dir(
        project_root,
        config,
        dataset_name=dataset_name,
        budget=budget,
        seed=seed,
        attempt=attempt,
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is append-only: {run_dir}")
    started_at = datetime.now(timezone.utc).isoformat()
    matrix_row = f"CORE::{dataset_name}::b{budget:.0f}::s{seed}"
    try:
        completed = _run_core_unit_impl(
            project_root,
            config_path,
            dataset_name=dataset_name,
            budget=budget,
            seed=seed,
            attempt=attempt,
        )
        _write_standard_bundle(
            completed,
            project_root=project_root,
            config=config,
            dataset_name=dataset_name,
            budget=budget,
            seed=seed,
            matrix_row=matrix_row,
            started_at=started_at,
            status="completed",
            required_methods=CORE_METHODS,
        )
        return completed
    except Exception:
        error_traceback = traceback.format_exc()
        _write_standard_bundle(
            run_dir,
            project_root=project_root,
            config=config,
            dataset_name=dataset_name,
            budget=budget,
            seed=seed,
            matrix_row=matrix_row,
            started_at=started_at,
            status="failed",
            required_methods=CORE_METHODS,
            error_traceback=error_traceback,
        )
        raise


def _run_sensitivity_unit_impl(
    project_root: str | Path,
    config_path: str | Path,
    *,
    factor: str,
    level: float,
    seed: int,
    attempt: int = 0,
) -> Path:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    base = load_protocol(config_path)
    validate_sensitivity_request(base, factor=factor, level=level, seed=seed)
    config = dict(base)
    if factor in {"hidden_dim", "collection_episodes_per_domain"}:
        config[factor] = int(level)
    elif factor != "forecast_multiplier":
        config[factor] = float(level)
    dataset_name = str(config.get("sensitivity_dataset", "gentd26"))
    budget = float(config.get("sensitivity_budget", 96.0))
    dataset = load_development_trace_dataset(
        project_root, dataset_name, horizon=int(config["horizon"])
    )
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    run_dir = _sensitivity_run_dir(
        project_root,
        config,
        factor=factor,
        level=level,
        dataset_name=dataset_name,
        seed=seed,
        attempt=attempt,
    )
    run_id = run_dir.name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is append-only: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "config.json",
        {**config, "factor": factor, "level": level, "dataset": dataset_name, "budget": budget, "seed": seed},
    )
    write_json(run_dir / "environment.json", environment_record())
    set_global_seed(seed, torch_threads=1)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    artifacts = _train(
        dataset, config, budget=budget, seed=seed, include_ablations=False
    )
    selection_started = time.perf_counter()
    selected, _, selection_rows = _selection(
        dataset, artifacts, config=config, budget=budget, seed=seed
    )
    selection_seconds = time.perf_counter() - selection_started
    multiplier = float(level) if factor == "forecast_multiplier" else 1.0
    planner = make_evidence_planner(
        value=artifacts.refreshed_value,
        forecaster=artifacts.decision_forecaster,
        gamma=float(config["gamma"]),
        forecast_multiplier=multiplier,
    )
    episodes, steps = evaluate_methods(
        dataset,
        {"dap_full": planner},
        split=str(config.get("evaluation_split", "validation_eval")),
        horizon=int(config["horizon"]),
        budget=budget,
        seed=seed + int(config.get("evaluation_seed_offset", 70_000_013)),
        episodes_per_domain=int(config["evaluation_episodes_per_domain"]),
        gamma=float(config.get("evaluation_gamma", 0.99)),
    )
    attach_seed_roles(episodes, training_seed=seed)
    attach_seed_roles(steps, training_seed=seed)
    episode_frame = pd.DataFrame(episodes)
    step_frame = pd.DataFrame(steps)
    episode_frame.to_csv(run_dir / "metrics.csv", index=False)
    step_frame.to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
    episode_frame.drop(
        columns=["decision_ms_mean", "decision_ms_p95"], errors="ignore"
    ).to_csv(run_dir / "metrics_deterministic.csv", index=False)
    step_frame.drop(columns=["decision_ms"], errors="ignore").to_csv(
        run_dir / "steps_deterministic.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    selection_frame = pd.DataFrame(selection_rows)
    selection_frame.to_csv(run_dir / "selection_metrics.csv", index=False)
    selection_frame.drop(
        columns=["decision_ms_mean", "decision_ms_p95"], errors="ignore"
    ).to_csv(run_dir / "selection_metrics_deterministic.csv", index=False)
    write_json(
        run_dir / "training.json",
        {
            "selected_value": selected,
            "training_seconds": artifacts.training_seconds,
            "full_method_training_seconds": full_method_training_seconds(
                artifacts.component_seconds
            ),
            "component_seconds": {
                **artifacts.component_seconds,
                "diagnostic_selection": selection_seconds,
            },
            "collection": artifacts.collection,
            "parameters": {
                "full_value": parameter_count(artifacts.refreshed_value),
                "decision_forecaster": parameter_count(artifacts.decision_forecaster),
            },
        },
    )
    torch.save(
        {
            "metadata": {
                "factor": factor,
                "level": float(level),
                "observation_dim": 14,
                "budget": float(budget),
                "horizon": int(config["horizon"]),
                "hidden_dim": int(config["hidden_dim"]),
                "planning_gamma": float(config["gamma"]),
                "evaluation_gamma": float(config.get("evaluation_gamma", 0.99)),
                "value_normalizer_mean": torch.as_tensor(
                    artifacts.refreshed_value.normalizer.mean, dtype=torch.float64
                ),
                "value_normalizer_scale": torch.as_tensor(
                    artifacts.refreshed_value.normalizer.scale, dtype=torch.float64
                ),
                "full_value_variant": "refreshed_value",
                "diagnostic_selected_value_variant": selected,
            },
            "refreshed_value": artifacts.refreshed_value.state_dict(),
            "decision_forecaster": artifacts.decision_forecaster.state_dict(),
        },
        run_dir / "models.pt",
    )
    write_json(
        run_dir / "runtime.json",
        {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    )
    _write_manifest(
        run_dir,
        config_path,
        str(
            config.get(
                "sensitivity_manifest_schema",
                "dap.dap_paper_evidence.sensitivity.v1",
            )
        ),
        project_root=project_root,
        dataset_name=dataset_name,
        budget=budget,
        seed=seed,
        started_at=started_at,
        matrix_row_id=f"SENS::{factor}::{level}::{dataset_name}::s{seed}",
        run_id=run_id,
        attempt=attempt,
    )
    return run_dir


def run_sensitivity_unit(
    project_root: str | Path,
    config_path: str | Path,
    *,
    factor: str,
    level: float,
    seed: int,
    attempt: int = 0,
) -> Path:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_protocol(config_path)
    validate_sensitivity_request(config, factor=factor, level=level, seed=seed)
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    dataset_name = str(config.get("sensitivity_dataset", "gentd26"))
    budget = float(config.get("sensitivity_budget", 96.0))
    run_dir = _sensitivity_run_dir(
        project_root,
        config,
        factor=factor,
        level=level,
        dataset_name=dataset_name,
        seed=seed,
        attempt=attempt,
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is append-only: {run_dir}")
    started_at = datetime.now(timezone.utc).isoformat()
    matrix_row = f"SENS::{factor}::{level}::{dataset_name}::s{seed}"
    try:
        completed = _run_sensitivity_unit_impl(
            project_root,
            config_path,
            factor=factor,
            level=level,
            seed=seed,
            attempt=attempt,
        )
        _write_standard_bundle(
            completed,
            project_root=project_root,
            config=config,
            dataset_name=dataset_name,
            budget=budget,
            seed=seed,
            matrix_row=matrix_row,
            started_at=started_at,
            status="completed",
            required_methods=("dap_full",),
        )
        return completed
    except Exception:
        error_traceback = traceback.format_exc()
        _write_standard_bundle(
            run_dir,
            project_root=project_root,
            config=config,
            dataset_name=dataset_name,
            budget=budget,
            seed=seed,
            matrix_row=matrix_row,
            started_at=started_at,
            status="failed",
            required_methods=("dap_full",),
            error_traceback=error_traceback,
        )
        raise
