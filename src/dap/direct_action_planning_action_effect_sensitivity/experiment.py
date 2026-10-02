"""Append-only development-split action-effect sensitivity experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import pandas as pd
import yaml

from dap.direct_action_planning_paper_closure.calibrated_protocol import (
    evaluate_calibrated_methods,
)
from dap.direct_action_planning_paper_closure.control_experiment import (
    _load_source_components,
    _source_dir,
)
from dap.direct_action_planning_paper_evidence.data import (
    load_development_trace_dataset,
)
from dap.utils.artifacts import (
    environment_record,
    sha256_file,
    sha256_tree,
    write_json,
)
from dap.utils.seed import set_global_seed

from .planner import make_capacity_biased_planner


PACKAGE = "src/dap/direct_action_planning_action_effect_sensitivity"


def load_protocol(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "tier", "source_tier", "datasets", "budgets", "seeds", "horizon",
        "gamma", "capacity_training_quantile", "evaluation_split",
        "evaluation_episodes_per_domain", "evaluation_seed_offset",
        "capacity_factors", "hidden_dim",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"sensitivity protocol missing keys: {sorted(missing)}")
    factors = tuple(float(value) for value in config["capacity_factors"])
    if factors != (0.9, 0.95, 1.0, 1.05, 1.1):
        raise ValueError("capacity factors must remain the registered +/-5/10% family")
    if str(config["evaluation_split"]) != "validation_eval":
        raise ValueError("sensitivity experiment is development-split only")
    if len(config["budgets"]) != 3 or len(config["seeds"]) != 5:
        raise ValueError("sensitivity matrix must retain three budgets and five seeds")
    return config


def factor_name(value: float) -> str:
    return f"capacity_{float(value):.2f}".replace(".", "p")


def _run_dir(root: Path, config: dict, dataset: str, budget: float, seed: int) -> Path:
    cell = f"{config['tier']}__{dataset}__b{budget:.0f}__s{seed}"
    return root / "results/direct_action_planning_action_effect_sensitivity" / str(config["tier"]) / dataset / cell


def verify_contract(root: Path, config_path: Path, contract_path: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "PASS":
        raise ValueError("pre-run implementation audit did not pass")
    if contract.get("config_sha256") != sha256_file(config_path):
        raise ValueError("sensitivity config drift after audit")
    if contract.get("code_tree_sha256") != sha256_tree(root / PACKAGE):
        raise ValueError("sensitivity code drift after audit")
    return contract


def run_unit(
    project_root: str | Path,
    config_path: str | Path,
    contract_path: str | Path,
    *,
    dataset_name: str,
    budget: float,
    seed: int,
) -> Path:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    contract_path = Path(contract_path).resolve()
    config = load_protocol(config_path)
    verify_contract(root, config_path, contract_path)
    registered = {
        (str(dataset), float(current_budget), int(current_seed))
        for dataset in config["datasets"]
        for current_budget in config["budgets"]
        for current_seed in config["seeds"]
    }
    if (dataset_name, float(budget), int(seed)) not in registered:
        raise ValueError("unregistered sensitivity cell")
    run_dir = _run_dir(root, config, dataset_name, float(budget), int(seed))
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"sensitivity run is append-only: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "config.json", {**config, "dataset": dataset_name, "budget": budget, "seed": seed})
    write_json(run_dir / "environment.json", environment_record())
    source = _source_dir(root, str(config["source_tier"]), dataset_name, float(budget), int(seed))
    if not source.exists():
        raise FileNotFoundError(f"frozen DAP source unavailable: {source}")
    wall_started = time.perf_counter()
    try:
        set_global_seed(int(seed), torch_threads=1)
        dataset = load_development_trace_dataset(root, dataset_name, horizon=int(config["horizon"]))
        full_value, _final_value, forecaster, selected = _load_source_components(
            source, hidden_dim=int(config["hidden_dim"])
        )
        methods = {
            factor_name(factor): make_capacity_biased_planner(
                value=full_value,
                forecaster=forecaster,
                gamma=float(config["gamma"]),
                continuation_weight=float(selected["continuation_weight"]),
                capacity_factor=float(factor),
            )
            for factor in config["capacity_factors"]
        }
        episodes, steps = evaluate_calibrated_methods(
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
        metrics = pd.DataFrame(episodes)
        step_frame = pd.DataFrame(steps)
        metrics["training_seed"] = int(seed)
        step_frame["training_seed"] = int(seed)
        if set(metrics.method.unique()) != set(methods):
            raise AssertionError("sensitivity method family is incomplete")
        if not np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError("non-finite sensitivity metrics")
        if float(metrics.budget_overspend.max()) > 1.0e-8:
            raise AssertionError("capacity-effect perturbation changed hard-budget safety")
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        step_frame.to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
        write_json(run_dir / "guardrails.json", {
            "budget_overspend_max": float(metrics.budget_overspend.max()),
            "budget_safe": True,
            "training_performed": False,
            "selection_performed": False,
            "evaluation_split": str(config["evaluation_split"]),
        })
        write_json(run_dir / "source.json", {
            "source_run": str(source.relative_to(root)),
            "source_models_sha256": sha256_file(source / "models.pt"),
            "source_manifest_sha256": sha256_file(source / "manifest.json"),
            "audit_contract_sha256": sha256_file(contract_path),
        })
        write_json(run_dir / "runtime.json", {
            "wall_seconds": time.perf_counter() - wall_started,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        })
        artifacts = {
            path.name: sha256_file(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(run_dir / "manifest.json", {
            "schema": "dap.dap.action_effect_sensitivity.v1",
            "status": "completed",
            "development_only": True,
            "dataset": dataset_name,
            "budget": float(budget),
            "training_seed": int(seed),
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256_file(config_path),
            "code_tree_sha256": sha256_tree(root / PACKAGE),
            "artifacts": artifacts,
        })
        return run_dir
    except Exception:
        write_json(run_dir / "failure.json", {
            "status": "failed",
            "traceback": traceback.format_exc(),
        })
        raise


def run_matrix(project_root: str | Path, config_path: str | Path, contract_path: str | Path) -> list[Path]:
    config = load_protocol(config_path)
    outputs = []
    for dataset in config["datasets"]:
        for budget in config["budgets"]:
            for seed in config["seeds"]:
                outputs.append(run_unit(
                    project_root, config_path, contract_path,
                    dataset_name=str(dataset), budget=float(budget), seed=int(seed),
                ))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    outputs = run_matrix(args.project_root, args.config, args.contract)
    print(json.dumps({"status": "completed", "cells": len(outputs)}))


if __name__ == "__main__":
    main()
