from __future__ import annotations

import argparse
from pathlib import Path

from dap.direct_action_planning_dataset_validation import (
    run_gate_matrix,
    run_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "gate"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.config is None:
        name = "smoke.yaml" if args.command == "smoke" else "gate.yaml"
        config = root / "research/direct_action_planning_dataset_validation/configs" / name
    else:
        config = args.config.resolve()
    outputs = run_smoke(root, config) if args.command == "smoke" else run_gate_matrix(root, config)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
