from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from stage2_dynamic_budget.direct_action_planning_dataset_validation.models import (
    FeatureNormalizer,
)
from stage2_dynamic_budget.direct_action_planning_paper_closure.calibrated_protocol import (
    collect_calibrated_branch_dataset,
    evaluate_calibrated_methods,
)
from stage2_dynamic_budget.direct_action_planning_paper_closure.experiment import (
    candidate_grid,
)
from stage2_dynamic_budget.direct_action_planning_paper_closure.models import (
    ScaledEvidenceValueNetwork,
)
from stage2_dynamic_budget.direct_action_planning_paper_closure.selection import (
    PlanningCandidate,
    SelectionGuardrails,
    select_planning_candidate,
)
from stage2_dynamic_budget.direct_action_planning_paper_evidence.data import (
    development_input_hashes,
    load_development_trace_dataset,
)
from stage2_dynamic_budget.utils.artifacts import (
    environment_record,
    sha256_file,
    sha256_tree,
    write_json,
)
from stage2_dynamic_budget.utils.seed import set_global_seed

from .planning import make_postdecision_planner
from .training import train_postdecision_candidates


def load_protocol(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
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
        "source_dap_tier",
        "source_dap_config",
        "collection_episodes_per_domain",
        "validation_episodes_per_domain",
        "selection_episodes_per_domain",
        "evaluation_episodes_per_domain",
        "candidate_iterations",
        "continuation_weights",
        "postdecision_epochs",
        "hidden_dim",
        "learning_rate",
        "batch_size",
        "capacity_training_quantile",
        "selection_seed_offset",
        "evaluation_seed_offset",
        "model_validation_split",
        "selection_split",
        "evaluation_split",
        "postdecision_exogenous_indices",
        "selection_guardrails",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing PDS protocol keys: {sorted(missing)}")
    if not config["datasets"] or not config["budgets"] or not config["seeds"]:
        raise ValueError("datasets, budgets, and seeds must not be empty")
    if config["model_validation_split"] != "validation_fit":
        raise ValueError("model_validation_split must be validation_fit")
    if config["selection_split"] != "validation_select":
        raise ValueError("selection_split must be validation_select")
    if config["evaluation_split"] != "validation_eval":
        raise ValueError("evaluation_split must be validation_eval")
    if tuple(int(value) for value in config["postdecision_exogenous_indices"]) != (0, 10, 11):
        raise ValueError("the preregistered PDS exogenous indices are (0, 10, 11)")
    candidate_grid(
        tuple(int(value) for value in config["candidate_iterations"]),
        tuple(float(value) for value in config["continuation_weights"]),
    )
    SelectionGuardrails(**config["selection_guardrails"])
    return config


def _source_run_dir(
    root: Path, config: dict[str, Any], dataset: str, budget: float, seed: int
) -> Path:
    tier = str(config["source_dap_tier"])
    name = f"{tier}__{dataset}__b{budget:.0f}__s{seed}"
    return root / "results/direct_action_planning_paper_closure" / tier / dataset / name


def _run_dir(
    root: Path,
    config: dict[str, Any],
    dataset: str,
    budget: float,
    seed: int,
    attempt: int = 0,
) -> Path:
    tier = str(config["tier"])
    suffix = "" if attempt == 0 else f"__a{attempt}"
    name = f"{tier}__{dataset}__b{budget:.0f}__s{seed}{suffix}"
    return root / "results/direct_action_planning_pds_adp" / tier / dataset / name


def load_source_value_candidates(
    run_dir: str | Path,
    *,
    candidate_iterations: tuple[int, ...],
    hidden_dim: int,
) -> tuple[dict[int, ScaledEvidenceValueNetwork], dict[str, Any]]:
    directory = Path(run_dir)
    checkpoint_path = directory / "models.pt"
    diagnostics_path = directory / "diagnostics.json"
    manifest_path = directory / "manifest.json"
    if not checkpoint_path.exists() or not diagnostics_path.exists() or not manifest_path.exists():
        raise ValueError(f"incomplete frozen DAP source: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("formal_test_accessed") is not False:
        raise ValueError("source DAP bundle is not a completed development-only run")
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    normalizer = FeatureNormalizer(
        mean=checkpoint["normalizer_mean"].cpu().numpy().astype(np.float64),
        scale=checkpoint["normalizer_scale"].cpu().numpy().astype(np.float64),
    )
    scale = float(checkpoint["value_output_scale"].item())
    zero_init = bool(checkpoint["zero_initialized_output"].item())
    values: dict[int, ScaledEvidenceValueNetwork] = {}
    for iteration in tuple(int(value) for value in candidate_iterations):
        key = str(iteration)
        if key not in checkpoint["values"]:
            raise ValueError(f"source DAP checkpoint lacks candidate {iteration}")
        model = ScaledEvidenceValueNetwork(
            normalizer,
            hidden_dim=int(hidden_dim),
            output_scale=scale,
            zero_initialize_output=zero_init,
        )
        model.load_state_dict(checkpoint["values"][key], strict=True)
        values[iteration] = model.train(False)
    return values, {
        "value_output_scale": scale,
        "zero_initialized_output": zero_init,
        "models_sha256": sha256_file(checkpoint_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _source_inventory(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, str]] = []
    for dataset in config["datasets"]:
        for budget in config["budgets"]:
            for seed in config["seeds"]:
                directory = _source_run_dir(root, config, str(dataset), float(budget), int(seed))
                for name in ("manifest.json", "models.pt", "diagnostics.json"):
                    path = directory / name
                    if not path.exists():
                        raise ValueError(f"missing frozen DAP source: {path}")
                    records.append(
                        {
                            "path": str(path.relative_to(root)),
                            "sha256": sha256_file(path),
                        }
                    )
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: row["path"]):
        digest.update(f"{record['sha256']}  {record['path']}\n".encode())
    return {"sha256": "sha256:" + digest.hexdigest(), "files": records}


def freeze_development_contract(
    project_root: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> Path:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"development contract is append-only: {output_path}")
    config = load_protocol(config_path)
    source_config = root / str(config["source_dap_config"])
    preregistration = root / "research/direct_action_planning_pds_adp/PREREGISTRATION.md"
    code_root = root / "src/stage2_dynamic_budget/direct_action_planning_pds_adp"
    contract = {
        "schema": "stage2.dap_pds_adp.development_contract.v1",
        "status": "frozen_before_formal_execution",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.relative_to(root)),
        "config_sha256": sha256_file(config_path),
        "source_dap_config_sha256": sha256_file(source_config),
        "preregistration_sha256": sha256_file(preregistration),
        "code_tree_sha256": sha256_tree(code_root),
        "source_inventory": _source_inventory(root, config),
        "test_arrays_loaded": False,
        "results_observed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, contract)
    return output_path


def _verify_development_contract(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_formal_execution":
        raise ValueError("development contract is not frozen")
    if contract.get("test_arrays_loaded") or contract.get("results_observed"):
        raise ValueError("development contract has an invalid time boundary")
    if sha256_file(config_path) != contract.get("config_sha256"):
        raise ValueError("formal PDS config changed after freeze")
    code_root = root / "src/stage2_dynamic_budget/direct_action_planning_pds_adp"
    if sha256_tree(code_root) != contract.get("code_tree_sha256"):
        raise ValueError("PDS code changed after formal freeze")
    if _source_inventory(root, config)["sha256"] != contract["source_inventory"]["sha256"]:
        raise ValueError("frozen DAP source inventory changed")
    return contract


def _checkpoint_payload(
    models: dict[int, ScaledEvidenceValueNetwork],
    selected: PlanningCandidate,
    *,
    target_scale: float,
    exogenous_indices: tuple[int, ...],
) -> dict[str, Any]:
    first = models[min(models)]
    return {
        "post_values": {str(key): model.state_dict() for key, model in models.items()},
        "normalizer_mean": torch.as_tensor(first.normalizer.mean, dtype=torch.float32),
        "normalizer_scale": torch.as_tensor(first.normalizer.scale, dtype=torch.float32),
        "target_scale": torch.as_tensor(float(target_scale), dtype=torch.float32),
        "candidate_iterations": torch.as_tensor(sorted(models), dtype=torch.int64),
        "selected_iteration": torch.as_tensor(int(selected.candidate_iteration), dtype=torch.int64),
        "selected_weight": torch.as_tensor(float(selected.continuation_weight), dtype=torch.float32),
        "exogenous_indices": torch.as_tensor(exogenous_indices, dtype=torch.int64),
    }


def _model_bytes(payload: dict[str, Any]) -> int:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return int(buffer.tell())


def run_unit(
    project_root: str | Path,
    config_path: str | Path,
    *,
    dataset_name: str,
    budget: float,
    seed: int,
    contract_path: str | Path | None = None,
    attempt: int = 0,
) -> Path:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_protocol(config_path)
    if dataset_name not in config["datasets"] or float(budget) not in config["budgets"] or int(seed) not in config["seeds"]:
        raise ValueError("unregistered PDS development cell")
    contract = None
    if contract_path is not None:
        contract = _verify_development_contract(
            root, config_path, config, Path(contract_path).resolve()
        )
    run_dir = _run_dir(root, config, dataset_name, float(budget), int(seed), int(attempt))
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"PDS run directory is append-only: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    resolved = {**config, "dataset": dataset_name, "budget": float(budget), "seed": int(seed)}
    write_json(run_dir / "config.json", resolved)
    write_json(run_dir / "environment.json", environment_record())
    (run_dir / "stdout.log").write_text("run started\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    set_global_seed(int(seed), torch_threads=1)
    wall_started = time.perf_counter()
    try:
        source_dir = _source_run_dir(root, config, dataset_name, float(budget), int(seed))
        source_values, source_metadata = load_source_value_candidates(
            source_dir,
            candidate_iterations=tuple(int(value) for value in config["candidate_iterations"]),
            hidden_dim=int(config["hidden_dim"]),
        )
        dataset = load_development_trace_dataset(root, dataset_name, horizon=int(config["horizon"]))
        training = collect_calibrated_branch_dataset(
            dataset,
            split="train",
            horizon=int(config["horizon"]),
            budget=float(budget),
            episodes_per_domain=int(config["collection_episodes_per_domain"]),
            seed=int(seed),
            quantile=float(config["capacity_training_quantile"]),
        )
        validation = collect_calibrated_branch_dataset(
            dataset,
            split=str(config["model_validation_split"]),
            horizon=int(config["horizon"]),
            budget=float(budget),
            episodes_per_domain=int(config["validation_episodes_per_domain"]),
            seed=int(seed) + 10_000_019,
            quantile=float(config["capacity_training_quantile"]),
        )
        pds_models, training_history = train_postdecision_candidates(
            training,
            validation,
            source_values=source_values,
            seed=int(seed) + 2,
            epochs=int(config["postdecision_epochs"]),
            learning_rate=float(config["learning_rate"]),
            hidden_dim=int(config["hidden_dim"]),
            batch_size=int(config["batch_size"]),
            target_scale=float(source_metadata["value_output_scale"]),
            exogenous_indices=tuple(int(value) for value in config["postdecision_exogenous_indices"]),
        )
        grid = candidate_grid(
            tuple(int(value) for value in config["candidate_iterations"]),
            tuple(float(value) for value in config["continuation_weights"]),
        )
        final_iteration = max(pds_models)
        selection_methods = {}
        for name, candidate in grid.items():
            iteration = final_iteration if candidate.candidate_iteration < 0 else candidate.candidate_iteration
            selection_methods[name] = make_postdecision_planner(
                value=pds_models[iteration],
                gamma=float(config["gamma"]),
                continuation_weight=float(candidate.continuation_weight),
            )
        selection_rows, _ = evaluate_calibrated_methods(
            dataset,
            selection_methods,
            split=str(config["selection_split"]),
            horizon=int(config["horizon"]),
            budget=float(budget),
            seed=int(seed) + int(config["selection_seed_offset"]),
            episodes_per_domain=int(config["selection_episodes_per_domain"]),
            gamma=float(config["gamma"]),
            quantile=float(config["capacity_training_quantile"]),
        )
        for row in selection_rows:
            candidate = grid[str(row["method"])]
            row["candidate_iteration"] = candidate.candidate_iteration
            row["continuation_weight"] = candidate.continuation_weight
        selected, selection_summary = select_planning_candidate(
            selection_rows,
            budget=float(budget),
            guardrails=SelectionGuardrails(**config["selection_guardrails"]),
        )
        selected_iteration = final_iteration if selected.candidate_iteration < 0 else selected.candidate_iteration
        methods = {
            "pds_adp": make_postdecision_planner(
                value=pds_models[selected_iteration],
                gamma=float(config["gamma"]),
                continuation_weight=float(selected.continuation_weight),
            ),
            "pds_adp_immediate": make_postdecision_planner(
                value=pds_models[final_iteration],
                gamma=float(config["gamma"]),
                continuation_weight=0.0,
            ),
        }
        episode_rows, step_rows = evaluate_calibrated_methods(
            dataset,
            methods,
            split=str(config["evaluation_split"]),
            horizon=int(config["horizon"]),
            budget=float(budget),
            seed=int(seed) + int(config["evaluation_seed_offset"]),
            episodes_per_domain=int(config["evaluation_episodes_per_domain"]),
            gamma=float(config["gamma"]),
            quantile=float(config["capacity_training_quantile"]),
        )
        for row in episode_rows:
            row.update(
                training_seed=int(seed),
                selected_iteration=int(selected.candidate_iteration),
                selected_continuation_weight=float(selected.continuation_weight),
            )
        for row in step_rows:
            row.update(
                training_seed=int(seed),
                selected_iteration=int(selected.candidate_iteration),
                selected_continuation_weight=float(selected.continuation_weight),
            )
        metrics = pd.DataFrame(episode_rows)
        steps = pd.DataFrame(step_rows)
        selection_frame = pd.DataFrame(selection_rows)
        if not np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError("non-finite PDS development metrics")
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        steps.to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
        selection_frame.to_csv(run_dir / "selection_metrics.csv", index=False)
        selection_summary.to_csv(run_dir / "selection_summary.csv", index=False)
        write_json(run_dir / "training.json", {"postdecision_candidates": training_history})
        write_json(
            run_dir / "diagnostics.json",
            {
                "selected": asdict(selected),
                "selected_runtime_iteration": int(selected_iteration),
                "final_iteration": int(final_iteration),
                "training_states": int(training.n_states),
                "validation_states": int(validation.n_states),
                "postdecision_pairs": int(np.sum(training.feasible)),
                "target_definition": "sampled_next_predecision_value",
                "exogenous_indices": list(config["postdecision_exogenous_indices"]),
                "source": source_metadata,
            },
        )
        payload = _checkpoint_payload(
            pds_models,
            selected,
            target_scale=float(source_metadata["value_output_scale"]),
            exogenous_indices=tuple(int(value) for value in config["postdecision_exogenous_indices"]),
        )
        torch.save(payload, run_dir / "models.pt")
        write_json(
            run_dir / "source_checkpoints.json",
            {"directory": str(source_dir.relative_to(root)), **source_metadata},
        )
        write_json(
            run_dir / "runtime.json",
            {
                "wall_seconds": float(time.perf_counter() - wall_started),
                "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "runtime_parameter_count": int(sum(parameter.numel() for parameter in pds_models[selected_iteration].parameters())),
                "trained_candidate_count": int(len(pds_models)),
                "checkpoint_bytes": _model_bytes(payload),
            },
        )
        write_json(
            run_dir / "guardrails.json",
            {
                "budget_overspend_max": float(metrics.budget_overspend.max()),
                "budget_safe": bool(metrics.budget_overspend.max() <= 1.0e-8),
                "all_metrics_finite": True,
                "formal_test_accessed": False,
            },
        )
        metrics.drop(columns=["decision_ms_mean", "decision_ms_p95"], errors="ignore").to_json(
            run_dir / "raw_metrics.jsonl", orient="records", lines=True
        )
        steps.drop(columns=["decision_ms"], errors="ignore").to_csv(
            run_dir / "predictions.csv", index=False
        )
        write_json(
            run_dir / "input_hashes.json",
            {"development_arrays": development_input_hashes(root, dataset_name), "formal_test": "not_loaded"},
        )
        (run_dir / "stdout.log").write_text("run completed\n", encoding="utf-8")
        ended_at = datetime.now(timezone.utc).isoformat()
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
                "seed_role": str(config["seed_role"]),
                "attempt": int(attempt),
                "started_at": started_at,
                "ended_at": ended_at,
                "config_sha256": sha256_file(config_path),
                "code_sha256": sha256_tree(root / "src/stage2_dynamic_budget/direct_action_planning_pds_adp"),
                "contract_sha256": sha256_file(Path(contract_path).resolve()) if contract_path else None,
                "source_dap_models_sha256": source_metadata["models_sha256"],
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


def run_matrix(
    project_root: str | Path,
    config_path: str | Path,
    *,
    contract_path: str | Path | None = None,
) -> list[Path]:
    root = Path(project_root).resolve()
    config = load_protocol(config_path)
    outputs: list[Path] = []
    for dataset in config["datasets"]:
        for budget in config["budgets"]:
            for seed in config["seeds"]:
                existing = _run_dir(root, config, str(dataset), float(budget), int(seed))
                if (existing / "manifest.json").exists():
                    manifest = json.loads((existing / "manifest.json").read_text(encoding="utf-8"))
                    if manifest.get("status") == "completed" and manifest.get("formal_test_accessed") is False:
                        outputs.append(existing)
                        continue
                outputs.append(
                    run_unit(
                        root,
                        config_path,
                        dataset_name=str(dataset),
                        budget=float(budget),
                        seed=int(seed),
                        contract_path=contract_path,
                    )
                )
    return outputs
