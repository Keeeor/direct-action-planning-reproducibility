from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["_config_path"] = str(path.resolve())
    return config


def node_url(kube, port: int = 30080) -> str:
    return f"http://{kube.node_internal_ip()}:{port}"


def fixed_rate_plan(*, rate: float, duration_seconds: float, seed: int, prefix: str) -> list[dict]:
    count = int(round(float(rate) * float(duration_seconds)))
    rng = np.random.default_rng(seed)
    offsets = np.arange(count, dtype=np.float64) / float(rate)
    offsets += rng.uniform(0.0, min(0.25 / float(rate), 0.0005), size=count)
    return [
        {
            "request_id": index, "step": int(offset),
            "scheduled_offset_seconds": float(offset), "target_rps": float(rate),
            "payload": f"{prefix}:{seed}:{index}",
        }
        for index, offset in enumerate(offsets)
    ]


def jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def reset_profile(kube, profile: str, replicas: int) -> dict:
    kube.set_profile(profile)
    kube.scale(replicas)
    kube.rollout_restart()
    # A nonzero old ReplicaSet can satisfy a simple Ready-count query while a
    # restarted Pod is still replacing it. Counter deltas must begin only
    # after the new rollout is available, otherwise a legitimate reset is
    # mistaken for missing application throughput.
    kube.rollout_status()
    return kube.wait_ready(replicas)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
