from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import gymnasium
import matplotlib
import numpy
import pandas
import scipy
import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_tree(root: str | Path, suffixes=(".py", ".yaml", ".toml")) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def environment_record() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pid": os.getpid(),
        "packages": {
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
