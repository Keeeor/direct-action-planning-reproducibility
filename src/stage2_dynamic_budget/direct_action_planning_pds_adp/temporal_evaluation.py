"""Locked chronological comparison of DAP and a standard PDS/ADP comparator.

The project-level test arrays were accessed by an earlier gate.  This module
therefore records a no-retuning temporal re-evaluation, not a new holdout.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import traceback
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml

from stage2_dynamic_budget.direct_action_planning_dataset_validation.data import (
    TraceDataset,
    load_trace_dataset,
)
from stage2_dynamic_budget.direct_action_planning_dataset_validation.models import (
    FeatureNormalizer,
)
from stage2_dynamic_budget.direct_action_planning_paper_closure.calibrated_protocol import (
    evaluate_calibrated_methods,
)
from stage2_dynamic_budget.direct_action_planning_paper_closure.models import (
    ScaledEvidenceValueNetwork,
)
from stage2_dynamic_budget.direct_action_planning_paper_closure.temporal_evaluation import (
    _load_dap,
)
from stage2_dynamic_budget.utils.artifacts import (
    environment_record,
    sha256_file,
    write_json,
)
from stage2_dynamic_budget.utils.seed import set_global_seed

from .planning import make_postdecision_planner


METHODS = ("dap_calibrated", "pds_adp")


def _expected_cells(config: dict[str, Any]) -> set[tuple[str, float, int]]:
    return {
        (str(dataset), float(budget), int(seed))
        for dataset in config["datasets"]
        for budget in config["budgets"]
        for seed in config["seeds"]
    }


def load_temporal_protocol(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "tier", "manifest_schema", "matrix_row_id", "datasets", "horizon",
        "budgets", "seeds", "seed_role", "gamma",
        "evaluation_episodes_per_domain", "test_seed_offset",
        "capacity_training_quantile", "methods", "dap_tier", "pds_tier",
        "historical_test_access",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing temporal protocol keys: {sorted(missing)}")
    if tuple(config["methods"]) != METHODS:
        raise ValueError(f"methods must be exactly {METHODS}")
    if config["historical_test_access"].get("project_wide_first_access") is not False:
        raise ValueError("prior project-wide test access must be disclosed")
    if not _expected_cells(config):
        raise ValueError("temporal test matrix must not be empty")
    return config


def _development_dir(
    root: Path, family: str, tier: str, dataset: str, budget: float, seed: int
) -> Path:
    name = f"{tier}__{dataset}__b{budget:.0f}__s{seed}"
    return root / "results" / family / tier / dataset / name


def _source_paths(root: Path, config: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for dataset, budget, seed in sorted(_expected_cells(config)):
        specifications = (
            ("direct_action_planning_paper_closure", str(config["dap_tier"])),
            ("direct_action_planning_pds_adp", str(config["pds_tier"])),
        )
        for family, tier in specifications:
            directory = _development_dir(root, family, tier, dataset, budget, seed)
            manifest_path = directory / "manifest.json"
            if not manifest_path.exists():
                raise ValueError(f"missing source manifest: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed" or manifest.get("formal_test_accessed") is not False:
                raise ValueError(f"source is not a completed development-only bundle: {directory}")
            for name in ("manifest.json", "models.pt", "diagnostics.json"):
                path = directory / name
                if not path.exists():
                    raise ValueError(f"missing source checkpoint artifact: {path}")
                paths.append(path)
    return paths


def _runtime_code_paths(root: Path) -> list[Path]:
    relative = (
        "src/stage2_dynamic_budget/direct_action_planning_pds_adp/planning.py",
        "src/stage2_dynamic_budget/direct_action_planning_pds_adp/temporal_evaluation.py",
        "src/stage2_dynamic_budget/direct_action_planning_paper_closure/planning.py",
        "src/stage2_dynamic_budget/direct_action_planning_paper_closure/models.py",
        "src/stage2_dynamic_budget/direct_action_planning_paper_closure/calibrated_protocol.py",
        "src/stage2_dynamic_budget/direct_action_planning_paper_closure/temporal_evaluation.py",
    )
    paths = [root / item for item in relative]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"missing runtime code paths: {missing}")
    return paths


def _inventory(paths: list[Path], root: Path) -> dict[str, Any]:
    records = [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(paths)
    ]
    digest = hashlib.sha256()
    for row in records:
        digest.update(f"{row['sha256']}  {row['path']}\n".encode())
    return {"sha256": "sha256:" + digest.hexdigest(), "files": records}


def freeze_temporal_contract(
    project_root: str | Path, template_path: str | Path, output_path: str | Path
) -> Path:
    """Freeze all selected checkpoints before this extension loads test arrays."""

    root = Path(project_root).resolve()
    template = Path(template_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"temporal contract is append-only: {output}")
    config = load_temporal_protocol(template)
    sources = _inventory(_source_paths(root, config), root)
    runtime = _inventory(_runtime_code_paths(root), root)
    contract = {
        "schema": "stage2.dap_pds_adp.temporal_contract.v1",
        "status": "frozen_before_extension_test_access",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "template_path": template.relative_to(root).as_posix(),
        "template_sha256": sha256_file(template),
        "config": config,
        "source_inventory": sources,
        "runtime_code_inventory": runtime,
        "test_arrays_loaded": False,
        "selection_performed": False,
        "project_wide_test_previously_accessed": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, contract)
    return output


def _verify_contract(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_extension_test_access":
        raise ValueError("temporal contract has invalid status")
    if contract.get("test_arrays_loaded") or contract.get("selection_performed"):
        raise ValueError("temporal contract crossed its time boundary")
    config = contract["config"]
    if _inventory(_source_paths(root, config), root)["sha256"] != contract["source_inventory"]["sha256"]:
        raise ValueError("source checkpoints changed after contract freeze")
    if _inventory(_runtime_code_paths(root), root)["sha256"] != contract["runtime_code_inventory"]["sha256"]:
        raise ValueError("runtime code changed after contract freeze")
    return contract


def load_pds_planner(
    run_dir: str | Path, *, gamma: float, hidden_dim: int
) -> tuple[Callable, dict[str, Any]]:
    directory = Path(run_dir)
    checkpoint = torch.load(directory / "models.pt", weights_only=True, map_location="cpu")
    diagnostics = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
    normalizer = FeatureNormalizer(
        mean=checkpoint["normalizer_mean"].cpu().numpy().astype(np.float64),
        scale=checkpoint["normalizer_scale"].cpu().numpy().astype(np.float64),
    )
    selected = int(checkpoint["selected_iteration"].item())
    runtime_iteration = max(int(key) for key in checkpoint["post_values"]) if selected < 0 else selected
    target_scale = float(checkpoint["target_scale"].item())
    value = ScaledEvidenceValueNetwork(
        normalizer,
        hidden_dim=int(hidden_dim),
        output_scale=target_scale,
        zero_initialize_output=True,
    )
    value.load_state_dict(checkpoint["post_values"][str(runtime_iteration)], strict=True)
    weight = float(checkpoint["selected_weight"].item())
    exogenous = tuple(int(value) for value in checkpoint["exogenous_indices"].tolist())
    planner = make_postdecision_planner(
        value=value.train(False), gamma=float(gamma), continuation_weight=weight,
        exogenous_indices=exogenous,
    )
    metadata = {
        "selected_candidate_iteration": selected,
        "selected_runtime_iteration": runtime_iteration,
        "continuation_weight": weight,
        "exogenous_indices": list(exogenous),
        "diagnostics_selected_runtime_iteration": int(diagnostics["selected_runtime_iteration"]),
    }
    if metadata["selected_runtime_iteration"] != metadata["diagnostics_selected_runtime_iteration"]:
        raise ValueError("PDS checkpoint and diagnostics disagree on selected iteration")
    return planner, metadata


def _test_hashes(dataset: TraceDataset) -> dict[str, str]:
    output: dict[str, str] = {}
    for domain in dataset.domain_names:
        values = dataset.domains[domain]["test"]
        digest = hashlib.sha256()
        digest.update(f"{dataset.name}/{domain}/test".encode())
        digest.update(values.astype(np.float64).tobytes(order="C"))
        output[domain] = "sha256:" + digest.hexdigest()
    return output


def _test_run_dir(
    root: Path, config: dict[str, Any], dataset: str, budget: float, seed: int, attempt: int
) -> Path:
    tier = str(config["tier"])
    suffix = "" if attempt == 0 else f"__a{attempt}"
    name = f"{tier}__{dataset}__b{budget:.0f}__s{seed}{suffix}"
    return root / "results/direct_action_planning_pds_adp" / tier / dataset / name


def run_temporal_test_unit(
    project_root: str | Path,
    contract_path: str | Path,
    *,
    dataset_name: str,
    budget: float,
    seed: int,
    attempt: int = 0,
) -> Path:
    root = Path(project_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = _verify_contract(root, contract_path)
    config = contract["config"]
    cell = (str(dataset_name), float(budget), int(seed))
    if cell not in _expected_cells(config):
        raise ValueError("unregistered temporal test cell")
    run_dir = _test_run_dir(root, config, *cell, int(attempt))
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"temporal run is append-only: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "config.json", {**config, "dataset": dataset_name, "budget": budget, "seed": seed})
    write_json(run_dir / "environment.json", environment_record())
    test_loaded = False
    try:
        dap_dir = _development_dir(
            root, "direct_action_planning_paper_closure", str(config["dap_tier"]),
            dataset_name, budget, seed,
        )
        pds_dir = _development_dir(
            root, "direct_action_planning_pds_adp", str(config["pds_tier"]),
            dataset_name, budget, seed,
        )
        methods = {"dap_calibrated": _load_dap(dap_dir, hidden_dim=64, gamma=float(config["gamma"]))["dap_calibrated"]}
        pds, pds_metadata = load_pds_planner(pds_dir, gamma=float(config["gamma"]), hidden_dim=64)
        methods["pds_adp"] = pds
        set_global_seed(int(seed), torch_threads=1)
        started = time.perf_counter()
        dataset = load_trace_dataset(root, dataset_name)
        test_loaded = True
        rows, steps = evaluate_calibrated_methods(
            dataset, methods, split="test", horizon=int(config["horizon"]),
            budget=float(budget), seed=int(seed) + int(config["test_seed_offset"]),
            episodes_per_domain=int(config["evaluation_episodes_per_domain"]),
            gamma=float(config["gamma"]), quantile=float(config["capacity_training_quantile"]),
        )
        metrics = pd.DataFrame(rows)
        step_frame = pd.DataFrame(steps)
        metrics["training_seed"] = int(seed)
        step_frame["training_seed"] = int(seed)
        if set(metrics.method.unique()) != set(METHODS):
            raise ValueError("temporal evaluator omitted a registered method")
        if not np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError("non-finite temporal metrics")
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        step_frame.to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
        metrics.drop(columns=["decision_ms_mean", "decision_ms_p95"], errors="ignore").to_json(
            run_dir / "raw_metrics.jsonl", orient="records", lines=True
        )
        step_frame.drop(columns=["decision_ms"], errors="ignore").to_csv(run_dir / "predictions.csv", index=False)
        write_json(run_dir / "input_hashes.json", _test_hashes(dataset))
        write_json(run_dir / "source_checkpoints.json", {
            "dap": {"models_sha256": sha256_file(dap_dir / "models.pt"), "manifest_sha256": sha256_file(dap_dir / "manifest.json")},
            "pds_adp": {"models_sha256": sha256_file(pds_dir / "models.pt"), "manifest_sha256": sha256_file(pds_dir / "manifest.json"), **pds_metadata},
        })
        overspend = float(metrics.budget_overspend.max())
        write_json(run_dir / "guardrails.json", {
            "budget_overspend_max": overspend, "budget_safe": overspend <= 1.0e-8,
            "all_metrics_finite": True, "training_or_selection_performed": False,
        })
        write_json(run_dir / "test_evidence.json", {
            "status": "PASS", "evaluation_split": "test",
            "extension_checkpoint_first_test": True,
            "project_wide_test_previously_accessed": True,
            "episode_rows": len(metrics), "step_rows": len(step_frame),
            "method_count": int(metrics.method.nunique()),
        })
        write_json(run_dir / "runtime.json", {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        })
        artifacts = {
            path.name: sha256_file(path) for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(run_dir / "manifest.json", {
            "schema": str(config["manifest_schema"]), "status": "completed",
            "dataset": dataset_name, "budget": float(budget), "training_seed": int(seed),
            "seed_role": str(config["seed_role"]), "attempt": int(attempt),
            "started_at": started_at, "ended_at": datetime.now(timezone.utc).isoformat(),
            "training_performed": False, "selection_performed": False,
            "extension_checkpoint_test_accessed": True,
            "project_wide_test_previously_accessed": True,
            "contract_sha256": sha256_file(contract_path), "artifacts": artifacts,
        })
        return run_dir
    except Exception:
        write_json(run_dir / "failure.json", {
            "status": "failed", "test_data_loaded": test_loaded,
            "traceback": traceback.format_exc(),
        })
        raise


def _worker(arguments: tuple[str, str, str, float, int]) -> str:
    root, contract, dataset, budget, seed = arguments
    return str(run_temporal_test_unit(root, contract, dataset_name=dataset, budget=budget, seed=seed))


def run_temporal_test_matrix(
    project_root: str | Path, contract_path: str | Path, *, workers: int = 1
) -> list[Path]:
    root = Path(project_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = _verify_contract(root, contract_path)
    cells = sorted(_expected_cells(contract["config"]))
    arguments = [(str(root), str(contract_path), dataset, budget, seed) for dataset, budget, seed in cells]
    if int(workers) <= 1:
        return [Path(_worker(item)) for item in arguments]
    outputs: list[Path] = []
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = {executor.submit(_worker, item): item for item in arguments}
        for completed, future in enumerate(as_completed(futures), start=1):
            output = Path(future.result())
            print(f"{completed}/{len(futures)} {output}", flush=True)
            outputs.append(output)
    return sorted(outputs)
