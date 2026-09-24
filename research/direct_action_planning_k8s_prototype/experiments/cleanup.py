from __future__ import annotations

"""Cleanup only the prototype scale target between independent trials."""

import argparse
from dataclasses import asdict, dataclass
import json
import time

from controller.kube_client import KubectlClient


@dataclass(frozen=True)
class CleanupResult:
    removed_autoscalers: tuple[str, ...]
    final_desired_replicas: int
    final_ready_replicas: int
    elapsed_seconds: float


def cleanup_target(
    kube: KubectlClient,
    *,
    base_replicas: int = 1,
    timeout_seconds: float = 90.0,
) -> CleanupResult:
    """Remove only HPA/KEDA resources targeting this Deployment and restore base.

    This intentionally does not delete a namespace, CRD, cluster addon, or
    any workload outside the isolated prototype target.
    """

    if base_replicas < 1:
        raise ValueError("base_replicas must be positive")
    started = time.monotonic()
    removed: list[str] = []
    for conflict in kube.autoscaler_conflicts():
        resource, name = conflict.split("/", 1)
        kube.run(["delete", resource, name, "--ignore-not-found"], check=False)
        removed.append(conflict)

    deadline = started + timeout_seconds
    while time.monotonic() < deadline:
        if not kube.autoscaler_conflicts():
            break
        time.sleep(0.25)
    else:
        raise TimeoutError(
            f"autoscaler resources still target {kube.deployment}: {kube.autoscaler_conflicts()}"
        )

    kube.scale(base_replicas)
    status = kube.deployment_status()
    while time.monotonic() < deadline:
        status = kube.deployment_status()
        if status["desired_replicas"] == base_replicas and status["ready_replicas"] <= base_replicas:
            break
        time.sleep(0.25)
    else:
        raise TimeoutError(f"target did not return to {base_replicas} base replicas: {status}")
    return CleanupResult(
        removed_autoscalers=tuple(removed),
        final_desired_replicas=status["desired_replicas"],
        final_ready_replicas=status["ready_replicas"],
        elapsed_seconds=time.monotonic() - started,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--deployment", default="dap-worker")
    parser.add_argument("--base-replicas", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    args = parser.parse_args()
    result = cleanup_target(
        KubectlClient(context=args.context, namespace=args.namespace, deployment=args.deployment),
        base_replicas=args.base_replicas,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
