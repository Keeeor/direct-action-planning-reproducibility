"""Append-only chronological evaluation for paper-closure checkpoints.

The available test arrays are historically accessed. This module therefore
records a locked temporal re-evaluation, never a newly untouched holdout.
"""

from __future__ import annotations

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

from dap.direct_action_planning_dataset_benchmark.budgeted import (
    BudgetedFittedQAgent,
    BudgetedQNetwork,
)
from dap.direct_action_planning_dataset_benchmark.cpo import (
    CPOActorCritic,
    CPOAgent,
)
from dap.direct_action_planning_dataset_benchmark.models import (
    ActorCritic,
    QNetwork,
)
from dap.direct_action_planning_dataset_benchmark.rl import (
    DQNAgent,
    PolicyAgent,
)
from dap.direct_action_planning_dataset_validation.data import (
    TraceDataset,
    load_trace_dataset,
)
from dap.direct_action_planning_dataset_validation.models import (
    FeatureNormalizer,
)
from dap.direct_action_planning_paper_evidence.models import (
    EvidenceLoadForecaster,
)
from dap.utils.artifacts import (
    environment_record,
    sha256_file,
    sha256_tree,
    write_json,
)
from dap.utils.seed import set_global_seed

from .calibrated_protocol import evaluate_calibrated_methods
from .models import ScaledEvidenceValueNetwork
from .planning import make_scaled_planner


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
METHODS = PRIMARY_METHODS + SUPPLEMENTARY_METHODS + INTERNAL_ABLATIONS
POLICY_METHODS = ("ppo", "ppo_lagrangian", "p3o", "a2c", "pid_lagrangian")


def _bare_hash(path: Path) -> str:
    return sha256_file(path).removeprefix("sha256:")


def _run_dir(root: Path, tier: str, dataset: str, budget: float, seed: int) -> Path:
    return root / "results/direct_action_planning_paper_closure" / tier / dataset / f"{tier}__{dataset}__b{budget:.0f}__s{seed}"


def _inventory_hash(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(f"{_bare_hash(path)}  {path.relative_to(root).as_posix()}\n".encode())
    return "sha256:" + digest.hexdigest()


def _expected_cells(config: dict[str, Any]) -> set[tuple[str, float, int]]:
    return {
        (str(dataset), float(budget), int(seed))
        for dataset in config["datasets"]
        for budget in config["budgets"]
        for seed in config["seeds"]
    }


def _source_paths(root: Path, config: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for tier in (str(config["dap_tier"]), str(config["baseline_tier"])):
        for dataset, budget, seed in sorted(_expected_cells(config)):
            directory = _run_dir(root, tier, dataset, budget, seed)
            manifest = directory / "manifest.json"
            if not manifest.exists():
                raise ValueError(f"missing source manifest: {manifest}")
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if data.get("status") != "completed" or data.get("formal_test_accessed") is not False:
                raise ValueError(f"source is not a completed development bundle: {directory}")
            paths.extend([manifest, directory / "models.pt"])
    return paths


def load_test_template(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "tier", "manifest_schema", "matrix_row_id", "datasets", "horizon",
        "budgets", "seeds", "seed_role", "gamma", "evaluation_episodes_per_domain",
        "test_seed_offset", "capacity_training_quantile", "dap_tier", "baseline_tier",
        "dap_config", "baseline_config", "historical_test_access",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing temporal test keys: {sorted(missing)}")
    if config["historical_test_access"].get("project_wide_first_access") is not False:
        raise ValueError("historical project-wide test access must be disclosed")
    if not tuple(config["datasets"]) or not tuple(config["budgets"]) or not tuple(config["seeds"]):
        raise ValueError("datasets, budgets, and seeds must be non-empty")
    return config


def freeze_test_contract(project_root: str | Path, template_path: str | Path, output_path: str | Path) -> Path:
    """Freeze source inventories without loading any test array."""

    root = Path(project_root).resolve()
    template_path = Path(template_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"test contract is append-only: {output_path}")
    config = load_test_template(template_path)
    source_paths = _source_paths(root, config)
    dap_config = root / str(config["dap_config"])
    baseline_config = root / str(config["baseline_config"])
    code_root = root / "src/dap/direct_action_planning_paper_closure"
    if not dap_config.exists() or not baseline_config.exists():
        raise ValueError("registered source config is absent")
    if not code_root.exists():
        raise ValueError("registered closure code root is absent")
    contract = {
        "schema": "dap.dap_paper_closure.temporal_contract.v1",
        "status": "frozen_before_current_test_access",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "template_path": str(template_path.relative_to(root)),
        "template_sha256": sha256_file(template_path),
        "config": config,
        "dap_config_sha256": sha256_file(dap_config),
        "baseline_config_sha256": sha256_file(baseline_config),
        "code_tree_sha256": sha256_tree(code_root),
        "source_inventory_sha256": _inventory_hash(source_paths, root),
        "source_artifact_count": len(source_paths),
        "test_arrays_loaded": False,
        "selection_performed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, contract)
    return output_path


def _load_dap(run_dir: Path, *, hidden_dim: int, gamma: float) -> dict[str, Callable]:
    checkpoint = torch.load(run_dir / "models.pt", weights_only=True, map_location="cpu")
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf-8"))
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

    def value(iteration: int) -> ScaledEvidenceValueNetwork:
        model = ScaledEvidenceValueNetwork(
            normalizer, hidden_dim=hidden_dim, output_scale=scale,
            zero_initialize_output=zero_init,
        )
        model.load_state_dict(checkpoint["values"][str(iteration)], strict=True)
        return model.train(False)

    forecaster = EvidenceLoadForecaster(normalizer)
    forecaster.load_state_dict(checkpoint["forecaster"], strict=True)
    return {
        "dap_calibrated": make_scaled_planner(
            value=value(selected_iteration), forecaster=forecaster.train(False), gamma=gamma,
            continuation_weight=float(selected["continuation_weight"]),
        ),
        "dap_immediate": make_scaled_planner(
            value=value(final_iteration), forecaster=forecaster.train(False), gamma=gamma,
            continuation_weight=0.0,
        ),
    }


def _load_baselines(run_dir: Path, *, budget: float, hidden_dim: int) -> dict[str, Callable]:
    states = torch.load(run_dir / "models.pt", weights_only=True, map_location="cpu")["models"]
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
        budgeted, budget, np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    )

    def wrap(agent: Any) -> Callable:
        return lambda env, observation: agent.select(env, observation)

    return {name: wrap(agent) for name, agent in agents.items()}


def _test_hashes(dataset: TraceDataset) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for domain in dataset.domain_names:
        values = dataset.domains[domain]["test"]
        digest = hashlib.sha256()
        digest.update(f"{dataset.name}/{domain}/test".encode())
        digest.update(values.astype(np.float64).tobytes(order="C"))
        hashes[domain] = "sha256:" + digest.hexdigest()
    return hashes


def run_test_unit(project_root: str | Path, contract_path: str | Path, *, dataset_name: str, budget: float, seed: int, attempt: int = 0) -> Path:
    root = Path(project_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_current_test_access" or contract.get("test_arrays_loaded"):
        raise ValueError("temporal contract is not valid")
    config = contract["config"]
    if (dataset_name, float(budget), int(seed)) not in _expected_cells(config):
        raise ValueError("unregistered temporal test cell")
    if _inventory_hash(_source_paths(root, config), root) != contract["source_inventory_sha256"]:
        raise ValueError("source inventory changed after test contract freeze")
    code_root = root / "src/dap/direct_action_planning_paper_closure"
    if sha256_tree(code_root) != contract.get("code_tree_sha256"):
        raise ValueError("closure code changed after test contract freeze")
    tier = str(config["tier"])
    name = f"{tier}__{dataset_name}__b{budget:.0f}__s{seed}" + ("" if attempt == 0 else f"__a{attempt}")
    run_dir = root / "results/direct_action_planning_paper_closure" / tier / dataset_name / name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"test run directory is append-only: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "config.json", {**config, "dataset": dataset_name, "budget": budget, "seed": seed})
    write_json(run_dir / "environment.json", environment_record())
    test_loaded = False
    try:
        dap_dir = _run_dir(root, str(config["dap_tier"]), dataset_name, budget, seed)
        baseline_dir = _run_dir(root, str(config["baseline_tier"]), dataset_name, budget, seed)
        methods = _load_dap(dap_dir, hidden_dim=64, gamma=float(config["gamma"]))
        methods.update(_load_baselines(baseline_dir, budget=budget, hidden_dim=64))
        methods = {name: methods[name] for name in METHODS}
        set_global_seed(seed, torch_threads=1)
        started = time.perf_counter()
        dataset = load_trace_dataset(root, dataset_name)
        test_loaded = True
        rows, steps = evaluate_calibrated_methods(
            dataset, methods, split="test", horizon=int(config["horizon"]), budget=budget,
            seed=seed + int(config["test_seed_offset"]),
            episodes_per_domain=int(config["evaluation_episodes_per_domain"]),
            gamma=float(config["gamma"]), quantile=float(config["capacity_training_quantile"]),
        )
        metrics = pd.DataFrame(rows)
        step_frame = pd.DataFrame(steps)
        metrics["training_seed"] = seed
        step_frame["training_seed"] = seed
        if not np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError("non-finite test metrics")
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        step_frame.to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
        metrics.drop(columns=["decision_ms_mean", "decision_ms_p95"], errors="ignore").to_json(run_dir / "raw_metrics.jsonl", orient="records", lines=True)
        step_frame.drop(columns=["decision_ms"], errors="ignore").to_csv(run_dir / "predictions.csv", index=False)
        write_json(run_dir / "input_hashes.json", _test_hashes(dataset))
        write_json(run_dir / "source_checkpoints.json", {
            "dap": {"models_sha256": sha256_file(dap_dir / "models.pt"), "manifest_sha256": sha256_file(dap_dir / "manifest.json")},
            "baselines": {"models_sha256": sha256_file(baseline_dir / "models.pt"), "manifest_sha256": sha256_file(baseline_dir / "manifest.json")},
        })
        write_json(run_dir / "guardrails.json", {
            "budget_overspend_max": float(metrics.budget_overspend.max()),
            "budget_safe": bool(metrics.budget_overspend.max() <= 1.0e-8),
            "all_metrics_finite": True,
            "training_or_selection_performed": False,
        })
        write_json(run_dir / "test_evidence.json", {
            "status": "PASS", "evaluation_split": "test", "current_checkpoint_first_test": True,
            "project_wide_test_previously_accessed": True, "episode_rows": len(metrics),
            "step_rows": len(step_frame), "method_count": int(metrics.method.nunique()),
        })
        write_json(run_dir / "runtime.json", {"wall_seconds": time.perf_counter() - started, "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)})
        ended_at = datetime.now(timezone.utc).isoformat()
        code_root = root / "src/dap/direct_action_planning_paper_closure"
        write_json(run_dir / "code_snapshot.json", {"code": str(code_root.relative_to(root)), "sha256": sha256_tree(code_root), "contract_sha256": sha256_file(contract_path)})
        artifact_hashes = {path.name: sha256_file(path) for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "manifest.json"}
        write_json(run_dir / "manifest.json", {
            "schema": str(config["manifest_schema"]), "status": "completed", "dataset": dataset_name,
            "budget": budget, "training_seed": seed, "attempt": attempt, "started_at": started_at,
            "ended_at": ended_at, "training_performed": False, "selection_performed": False,
            "current_checkpoint_test_accessed": True, "project_wide_test_previously_accessed": True,
            "contract_sha256": sha256_file(contract_path), "artifacts": artifact_hashes,
        })
        return run_dir
    except Exception:
        write_json(run_dir / "failure.json", {"status": "failed", "test_data_loaded": test_loaded, "traceback": traceback.format_exc()})
        raise


def run_test_matrix(project_root: str | Path, contract_path: str | Path) -> list[Path]:
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    outputs = []
    for dataset, budget, seed in sorted(_expected_cells(contract["config"])):
        outputs.append(run_test_unit(project_root, contract_path, dataset_name=dataset, budget=budget, seed=seed))
    return outputs
