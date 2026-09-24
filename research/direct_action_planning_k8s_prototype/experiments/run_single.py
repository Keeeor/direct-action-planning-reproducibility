from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.smoke_test import main as smoke_main


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    parser.add_argument("-h", "--help", action="store_true")
    args, _ = parser.parse_known_args()
    if args.help:
        parser.print_help()
        return 0
    if args.config is None:
        parser.error("the following arguments are required: --config")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("mode") == "smoke":
        return smoke_main()
    raise ValueError(f"unsupported run mode: {config.get('mode')!r}")


if __name__ == "__main__":
    raise SystemExit(main())
