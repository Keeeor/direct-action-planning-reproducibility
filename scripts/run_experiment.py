#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dap.experiment import run_synthetic_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", default="main")
    args = parser.parse_args()
    run_dir = run_synthetic_experiment(
        ROOT, args.config, args.method, args.budget, args.seed, args.variant
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
