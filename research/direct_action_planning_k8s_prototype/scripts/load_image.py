from __future__ import annotations

import argparse
import shutil
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--kind-cluster", default="dap-prototype")
    args = parser.parse_args()
    subprocess.run(["docker", "image", "inspect", args.image], check=True)
    if args.context.startswith("kind-"):
        if not shutil.which("kind"):
            raise SystemExit("kind is required for a kind context")
        subprocess.run(
            ["kind", "load", "docker-image", args.image, "--name", args.kind_cluster], check=True
        )
    elif args.context == "minikube":
        if not shutil.which("minikube"):
            raise SystemExit("minikube is required for the minikube context")
        subprocess.run(["minikube", "image", "load", args.image], check=True)
    else:
        raise SystemExit("automatic local-image loading supports only kind and Minikube")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

