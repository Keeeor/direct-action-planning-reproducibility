from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load one or more local images into the selected kind or Minikube cluster."
    )
    parser.add_argument("--context", required=True)
    parser.add_argument("--kind-cluster", default="dap-prototype")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()
    loader = Path(__file__).with_name("load_image.py")
    for image in args.images:
        subprocess.run(
            [
                sys.executable,
                str(loader),
                "--image",
                image,
                "--context",
                args.context,
                "--kind-cluster",
                args.kind_cluster,
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
