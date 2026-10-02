from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml


LOCAL_ROOT = Path(__file__).resolve().parents[1]
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))

from calibration.common import ROOT, write_json


PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dap.data.trace_windows import select_trace_window  # noqa: E402
from dap.direct_action_planning_dataset_validation.data import load_trace_dataset  # noqa: E402
from dap.direct_action_planning_dataset_validation.models import FeatureNormalizer  # noqa: E402
from dap.direct_action_planning_dataset_validation.training import BranchDataset  # noqa: E402
from dap.direct_action_planning_paper_closure.models import ScaledEvidenceValueNetwork  # noqa: E402
from dap.direct_action_planning_paper_closure.training import (  # noqa: E402
    compute_value_target_scale,
    train_value_candidates,
)
from dap.direct_action_planning_paper_evidence.models import EvidenceLoadForecaster  # noqa: E402

from controller.action_mapper import ACTION_ORDER, ActionMapper
from controller.checkpoint_loader import save_checkpoint
from controller.system_model import StructuredSystemModel
from workload.trace_converter import fit_rate_scale, transform_rate


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _state_from_observation(observation: np.ndarray, ready_pods: int) -> dict[str, float]:
    return {
        "current_request_rate": float(observation[0]),
        "recent_request_rate": float(observation[1]),
        "queue_depth": float(observation[2]),
        "queue_growth_rate": float(observation[3]),
        "ready_pods": float(ready_pods),
        "busy_pod_ratio": float(observation[6]),
        "cpu_utilization": float(observation[6]),
        "mean_latency_seconds": float(observation[7]),
        "p95_latency_seconds": float(observation[8]),
        "slo_violation_rate": float(observation[9]),
        "burst_intensity": float(observation[10]),
        "queue_to_capacity": float(observation[11]),
    }


def _initial_observation(model: StructuredSystemModel, rate: float, budget: float, horizon: int) -> np.ndarray:
    capacity = model.capacity(1)
    return np.asarray(
        [rate, rate, 0.0, 0.0, capacity, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, rate / max(capacity, 1.0e-9), 1.0, 1.0],
        dtype=np.float32,
    )


def collect_branches(
    *, dataset_name: str, domain: str, profile: str, split: str,
    model: StructuredSystemModel, mapper: ActionMapper, budgets: tuple[float, ...],
    horizon: int, episodes: int, seed: int, target_peak_rps: float, max_rps: float,
    control_interval_seconds: float,
) -> BranchDataset:
    dataset = load_trace_dataset(PROJECT_ROOT, dataset_name)
    scale = fit_rate_scale(
        dataset.domains[domain]["train"], quantile=0.99,
        target_peak_rps=target_peak_rps, max_rps=max_rps,
    )
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    feasibilities: list[np.ndarray] = []
    done: list[bool] = []
    next_load: list[float] = []
    domains: list[str] = []
    starts: list[int] = []
    for budget_index, budget in enumerate(budgets):
        for episode in range(episodes):
            window_seed = seed + budget_index * 1_000_003 + episode * 9_973
            raw, start = select_trace_window(dataset.domains[domain][split], horizon, window_seed)
            rates = transform_rate(raw, scale)
            observation = _initial_observation(model, float(rates[0]), budget, horizon)
            ready = 1
            remaining = float(budget)
            for step in range(horizon):
                true_next = float(rates[step + 1]) if step + 1 < horizon else 0.0
                state = _state_from_observation(observation, ready)
                feasible = np.zeros(4, dtype=bool)
                branch_next = np.zeros((4, 14), dtype=np.float32)
                branch_reward = np.full(4, -np.inf, dtype=np.float32)
                branch_costs = np.full(4, np.inf, dtype=np.float64)
                for index, action in enumerate(ACTION_ORDER):
                    branch = model.branch(
                        observation=observation, state=state, action=action, mapper=mapper,
                        forecast_arrival_rps=true_next, total_budget_seconds=budget,
                        remaining_budget_seconds=remaining,
                        remaining_horizon_steps=horizon - step, horizon_steps=horizon,
                        control_interval_seconds=control_interval_seconds,
                    )
                    # The training feasibility mask uses the same one-cycle
                    # commitment + calibrated termination reserve as runtime.
                    target_extra = max(branch.target_replicas - 1, 0)
                    commitment = branch.expected_cost_seconds + target_extra * model.scale_down_guard_seconds
                    allowed = action == "no_op" or commitment <= remaining + 1.0e-9
                    feasible[index] = allowed
                    branch_next[index] = branch.next_observation
                    branch_reward[index] = branch.reward
                    branch_costs[index] = branch.expected_cost_seconds
                if not feasible.any():
                    raise AssertionError("base action must remain feasible")
                observations.append(observation.copy())
                next_observations.append(branch_next)
                rewards.append(branch_reward)
                feasibilities.append(feasible)
                done.append(step == horizon - 1)
                next_load.append(true_next)
                domains.append(f"{profile}:{domain}")
                starts.append(int(start))
                choices = np.flatnonzero(feasible)
                selected = int(rng.choice(choices))
                selected_action = ACTION_ORDER[selected]
                remaining = max(remaining - branch_costs[selected], 0.0)
                ready = mapper.replicas(selected_action)
                observation = branch_next[selected]
    return BranchDataset(
        observations=np.asarray(observations, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32), feasible=np.asarray(feasibilities, dtype=bool),
        done=np.asarray(done, dtype=bool), next_load=np.asarray(next_load, dtype=np.float32),
        domain=np.asarray(domains, dtype=str), window_start=np.asarray(starts, dtype=np.int64),
    )


def train_forecaster(
    train: BranchDataset, validation: BranchDataset, normalizer: FeatureNormalizer,
    *, seed: int, epochs: int,
) -> tuple[EvidenceLoadForecaster, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = EvidenceLoadForecaster(normalizer)
    # The production trace adapter is expressed in real requests/second,
    # rather than the small simulator load units used by the original network.
    # Start the positive output at a training-only persistence-scale value so
    # optimization learns deviations, not the raw unit conversion.
    target_median = float(np.median(train.next_load))
    target_scale = max(float(np.std(train.next_load)), 1.0)
    with torch.no_grad():
        final = model.network[-1]
        nn.init.zeros_(final.weight)
        final.bias.fill_(target_median)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    x = torch.as_tensor(train.observations, dtype=torch.float32)
    y = torch.as_tensor(train.next_load, dtype=torch.float32)
    vx = torch.as_tensor(validation.observations, dtype=torch.float32)
    vy = validation.next_load
    rng = np.random.default_rng(seed)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_mae = np.inf
    history = []
    for epoch in range(int(epochs)):
        losses = []
        order = rng.permutation(len(x))
        for start in range(0, len(x), 256):
            index = torch.as_tensor(order[start : start + 256], dtype=torch.long)
            loss = nn.functional.smooth_l1_loss(
                model(x[index]) / target_scale, y[index] / target_scale
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            prediction = model(
                torch.as_tensor(validation.observations, dtype=torch.float32)
            ).cpu().numpy()
        mae = float(np.mean(np.abs(prediction - vy)))
        history.append({
            "epoch": float(epoch), "training_loss": float(np.mean(losses)),
            "validation_mae": mae, "target_median_rps": target_median,
            "target_scale_rps": target_scale,
        })
        if mae < best_mae:
            best_mae = mae
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.train(False), history


def validation_return(
    data: BranchDataset, *, value: ScaledEvidenceValueNetwork, forecaster: EvidenceLoadForecaster,
    gamma: float, continuation_weight: float,
) -> float:
    next_values = value.predict(data.next_observations.reshape(-1, 14)).reshape(data.n_states, 4)
    forecast = np.asarray([forecaster.predict(obs) for obs in data.observations])
    # Forecast calibration matters but branch labels retain common random
    # exogenous outcomes. This criterion selects FVI round/continuation only
    # on validation data; it never reads a formal test trace.
    alignment = -np.mean(np.abs(forecast - data.next_load))
    q = data.rewards + gamma * continuation_weight * (~data.done[:, None]) * next_values
    q[~data.feasible] = -np.inf
    return float(np.mean(np.max(q, axis=1)) + 0.01 * alignment)


def run(config: dict[str, Any], config_path: Path) -> list[Path]:
    mapper = ActionMapper(dict(zip(ACTION_ORDER, (1, 2, 3, 5))))
    budgets = tuple(float(value) for value in config["budgets_seconds"])
    output_root = ROOT / str(config.get("checkpoint_root", "results/prototype_checkpoints"))
    outputs = []
    for profile, source in config["profiles"].items():
        model = StructuredSystemModel.load(
            ROOT / str(config["system_model"]), profile, slo_seconds=float(source["slo_seconds"])
        )
        common = {
            "dataset_name": str(source["dataset"]), "domain": str(source["domain"]),
            "profile": profile, "model": model, "mapper": mapper, "budgets": budgets,
            "horizon": int(config["horizon_steps"]), "target_peak_rps": float(source["target_peak_rps"]),
            "max_rps": float(source["max_rps"]),
            "control_interval_seconds": float(config["control_interval_seconds"]),
        }
        train = collect_branches(split="train", episodes=int(config["train_episodes"]), seed=int(config["seed"]), **common)
        validation = collect_branches(split="validation", episodes=int(config["validation_episodes"]), seed=int(config["seed"]) + 7_000_003, **common)
        normalizer = FeatureNormalizer.fit(train.observations)
        forecaster, forecaster_history = train_forecaster(
            train, validation, normalizer, seed=int(config["seed"]) + 13, epochs=int(config["forecaster_epochs"])
        )
        values, fvi_history = train_value_candidates(
            train, validation, seed=int(config["seed"]) + 17, gamma=float(config["gamma"]),
            iterations=int(config["fvi_iterations"]),
            candidate_iterations=tuple(int(item) for item in config["candidate_iterations"]),
            epochs_per_iteration=int(config["fvi_epochs_per_iteration"]), learning_rate=float(config["learning_rate"]),
            hidden_dim=int(config["hidden_dim"]), target_scale=compute_value_target_scale(train, horizon=int(config["horizon_steps"])),
            zero_initialize_output=True,
        )
        selections = []
        for iteration, value in values.items():
            for weight in config["continuation_weights"]:
                score = validation_return(
                    validation, value=value, forecaster=forecaster, gamma=float(config["gamma"]),
                    continuation_weight=float(weight),
                )
                selections.append({"iteration": int(iteration), "continuation_weight": float(weight), "validation_score": score})
        selected = max(selections, key=lambda row: row["validation_score"])
        checkpoint_dir = output_root / profile
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / "models.pt"
        if path.exists():
            raise FileExistsError(f"prototype checkpoint is append-only: {path}")
        metadata = {
            "profile": profile, "dataset": source["dataset"], "domain": source["domain"],
            "training_split": "train", "validation_split": "validation", "formal_test_accessed": False,
            "system_model_sha256": _sha256(ROOT / str(config["system_model"])),
            "train_states": train.n_states, "validation_states": validation.n_states,
            "selection": selected,
        }
        save_checkpoint(
            path, value=values[int(selected["iteration"])], forecaster=forecaster,
            gamma=float(config["gamma"]), continuation_weight=float(selected["continuation_weight"]),
            metadata=metadata,
        )
        write_json(checkpoint_dir / "diagnostics.json", {
            "schema": "dap.k8s.prototype_training.v1", "status": "completed", **metadata,
            "fvi_history": fvi_history, "forecaster_history": forecaster_history,
            "selection_candidates": selections, "config_sha256": _sha256(config_path),
        })
        outputs.append(path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = run(config, args.config.resolve())
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
