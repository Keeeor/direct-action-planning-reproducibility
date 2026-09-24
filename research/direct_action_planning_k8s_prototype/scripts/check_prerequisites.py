from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


COMMANDS = ("docker", "kubectl", "minikube", "kind", "k6", "helm")
PYTHON_MODULES = ("aiohttp", "yaml", "numpy", "pandas", "torch", "kubernetes")


def run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "argv": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def inspect(context: str | None) -> dict[str, Any]:
    commands = {name: shutil.which(name) for name in COMMANDS}
    modules = {name: importlib.util.find_spec(name) is not None for name in PYTHON_MODULES}
    result: dict[str, Any] = {
        "schema": "dap.k8s.prerequisites.v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "commands": commands,
        "python_modules": modules,
        "context": context,
    }
    if commands["docker"]:
        result["docker"] = run(["docker", "version", "--format", "{{json .}}"])
    if commands["kubectl"]:
        base = ["kubectl"] + (["--context", context] if context else [])
        result["cluster_info"] = run(base + ["cluster-info"])
        result["nodes"] = run(base + ["get", "nodes", "-o", "json"])
        result["scale_permission"] = run(
            base + ["auth", "can-i", "patch", "deployments.apps", "--all-namespaces"]
        )
        result["api_resources"] = run(base + ["api-resources", "-o", "name"])
    result["core_ready"] = bool(
        commands["docker"]
        and commands["kubectl"]
        and result.get("cluster_info", {}).get("returncode") == 0
        and result.get("scale_permission", {}).get("stdout", "").lower() == "yes"
        and modules["aiohttp"]
        and modules["yaml"]
        and modules["numpy"]
    )
    resources = result.get("api_resources", {}).get("stdout", "")
    result["optional"] = {
        "k6": bool(commands["k6"]),
        "minikube": bool(commands["minikube"]),
        "kind": bool(commands["kind"]),
        "python_kubernetes": modules["kubernetes"],
        "hpa_api": "horizontalpodautoscalers.autoscaling" in resources,
        "keda_crd": "scaledobjects.keda.sh" in resources,
        "prometheus_operator_crd": "servicemonitors.monitoring.coreos.com" in resources,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = inspect(args.context)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["core_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

