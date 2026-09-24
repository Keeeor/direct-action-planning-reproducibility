from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def command_ok(command: list[str]) -> bool:
    return subprocess.run(command, capture_output=True, check=False).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--kind-cluster", default="dap-prototype")
    args = parser.parse_args()
    if command_ok(["kubectl", "--context", args.context, "cluster-info"]):
        print(f"using existing Kubernetes context {args.context}")
        return 0
    if args.context.startswith("kind-") and shutil.which("kind"):
        subprocess.run(
            [
                "kind",
                "create",
                "cluster",
                "--name",
                args.kind_cluster,
                "--config",
                str(ROOT / "kubernetes/kind-config.yaml"),
            ],
            check=True,
        )
        return 0
    if args.context == "minikube" and shutil.which("minikube"):
        subprocess.run(["minikube", "start"], check=True)
        return 0
    raise SystemExit(f"context {args.context!r} is unavailable and no matching local backend exists")


if __name__ == "__main__":
    raise SystemExit(main())

