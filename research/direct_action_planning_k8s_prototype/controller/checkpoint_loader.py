from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROTOTYPE_ROOT.parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dap.direct_action_planning_dataset_validation.models import FeatureNormalizer  # noqa: E402
from dap.direct_action_planning_paper_evidence.models import EvidenceLoadForecaster  # noqa: E402
from dap.direct_action_planning_paper_closure.models import ScaledEvidenceValueNetwork  # noqa: E402


@dataclass(frozen=True)
class DAPCheckpoint:
    value: ScaledEvidenceValueNetwork
    forecaster: EvidenceLoadForecaster
    gamma: float
    continuation_weight: float
    metadata: dict[str, Any]
    sha256: str


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path: str | Path, *, hidden_dim: int = 64) -> DAPCheckpoint:
    path = Path(path)
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if payload.get("schema") != "dap.k8s.prototype_checkpoint.v1":
        raise ValueError(
            "prototype controller accepts only a prototype-adapted checkpoint; "
            "a frozen simulator checkpoint requires a train/validation-only adapter"
        )
    mean = np.asarray(payload["normalizer_mean"], dtype=np.float64)
    scale = np.asarray(payload["normalizer_scale"], dtype=np.float64)
    if mean.shape != (14,) or scale.shape != (14,):
        raise ValueError("prototype checkpoint must contain 14-dimensional normalization")
    normalizer = FeatureNormalizer(mean=mean, scale=scale)
    value = ScaledEvidenceValueNetwork(
        normalizer, hidden_dim=int(payload.get("hidden_dim", hidden_dim)),
        output_scale=float(payload["value_output_scale"]),
        zero_initialize_output=bool(payload.get("zero_initialized_output", False)),
    )
    value.load_state_dict(payload["value_state"], strict=True)
    forecaster = EvidenceLoadForecaster(normalizer)
    forecaster.load_state_dict(payload["forecaster_state"], strict=True)
    return DAPCheckpoint(
        value=value.train(False), forecaster=forecaster.train(False),
        gamma=float(payload["gamma"]), continuation_weight=float(payload["continuation_weight"]),
        metadata=dict(payload["metadata"]), sha256=_hash(path),
    )


def save_checkpoint(
    path: str | Path, *, value: ScaledEvidenceValueNetwork,
    forecaster: EvidenceLoadForecaster, gamma: float, continuation_weight: float,
    metadata: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "dap.k8s.prototype_checkpoint.v1",
        "normalizer_mean": torch.as_tensor(value.normalizer.mean, dtype=torch.float64),
        "normalizer_scale": torch.as_tensor(value.normalizer.scale, dtype=torch.float64),
        "hidden_dim": int(value.network[0].out_features),
        "value_output_scale": float(value.output_scale),
        "zero_initialized_output": bool(value.zero_initialize_output),
        "value_state": value.state_dict(),
        "forecaster_state": forecaster.state_dict(),
        "gamma": float(gamma),
        "continuation_weight": float(continuation_weight),
        "metadata": metadata,
    }
    torch.save(payload, path)
    return path
