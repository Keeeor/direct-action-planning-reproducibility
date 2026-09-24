from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
import time
from typing import Any
from urllib.parse import quote


@dataclass(frozen=True)
class CommandEvidence:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    latency_seconds: float


class KubectlClient:
    def __init__(self, *, context: str, namespace: str, deployment: str = "dap-worker"):
        self.context = context
        self.namespace = namespace
        self.deployment = deployment

    def _base(self) -> list[str]:
        return ["kubectl", "--context", self.context, "-n", self.namespace]

    def run(self, args: list[str], *, check: bool = True) -> CommandEvidence:
        argv = self._base() + args
        started = time.perf_counter()
        completed = subprocess.run(argv, text=True, capture_output=True, check=False)
        evidence = CommandEvidence(
            argv=tuple(argv),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            latency_seconds=time.perf_counter() - started,
        )
        if check and completed.returncode != 0:
            raise RuntimeError(
                f"kubectl failed ({completed.returncode}): {' '.join(argv)}\n{completed.stderr}"
            )
        return evidence

    def get_json(self, resource: str, name: str | None = None, *, labels: str | None = None) -> dict:
        args = ["get", resource]
        if name:
            args.append(name)
        if labels:
            args += ["-l", labels]
        args += ["-o", "json"]
        return json.loads(self.run(args).stdout)

    def scale(self, replicas: int) -> CommandEvidence:
        if replicas < 0:
            raise ValueError("replicas must be non-negative")
        return self.run(["scale", f"deployment/{self.deployment}", f"--replicas={replicas}"])

    def set_profile(self, profile: str) -> CommandEvidence:
        return self.run(
            ["set", "env", f"deployment/{self.deployment}", f"SERVICE_PROFILE={profile}"]
        )

    def rollout_status(self, timeout_seconds: int = 180) -> CommandEvidence:
        return self.run(
            ["rollout", "status", f"deployment/{self.deployment}", f"--timeout={timeout_seconds}s"]
        )

    def rollout_restart(self) -> CommandEvidence:
        return self.run(["rollout", "restart", f"deployment/{self.deployment}"])

    def wait_ready(self, replicas: int, timeout_seconds: float = 180.0) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout_seconds)
        latest: dict[str, Any] = {}
        while time.monotonic() < deadline:
            latest = self.deployment_status()
            if latest["ready_replicas"] >= int(replicas):
                return latest
            time.sleep(0.25)
        raise TimeoutError(f"deployment did not reach {replicas} Ready replicas: {latest}")

    def deployment_status(self) -> dict[str, Any]:
        item = self.get_json("deployment", self.deployment)
        status = item.get("status", {})
        return {
            "desired_replicas": int(item.get("spec", {}).get("replicas", 0)),
            "ready_replicas": int(status.get("readyReplicas", 0)),
            "available_replicas": int(status.get("availableReplicas", 0)),
            "updated_replicas": int(status.get("updatedReplicas", 0)),
        }

    def worker_pods(self) -> list[dict[str, Any]]:
        payload = self.get_json("pods", labels="app.kubernetes.io/name=dap-worker")
        rows = []
        for item in payload.get("items", []):
            conditions = {entry["type"]: entry for entry in item.get("status", {}).get("conditions", [])}
            ready = conditions.get("Ready", {}).get("status") == "True"
            rows.append(
                {
                    "name": item["metadata"]["name"],
                    "uid": item["metadata"]["uid"],
                    "created_at": item["metadata"].get("creationTimestamp"),
                    "phase": item.get("status", {}).get("phase"),
                    "ready": ready,
                    "ready_transition_at": conditions.get("Ready", {}).get("lastTransitionTime"),
                    "deleting": bool(item["metadata"].get("deletionTimestamp")),
                    "pod_ip": item.get("status", {}).get("podIP"),
                    "node": item.get("spec", {}).get("nodeName"),
                }
            )
        return rows

    def pod_metrics(self, pod: str, port: int = 8000) -> CommandEvidence:
        path = f"/api/v1/namespaces/{quote(self.namespace)}/pods/{quote(pod)}:{port}/proxy/metrics"
        argv = ["kubectl", "--context", self.context, "get", "--raw", path]
        started = time.perf_counter()
        completed = subprocess.run(argv, text=True, capture_output=True, check=False)
        evidence = CommandEvidence(
            argv=tuple(argv),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            latency_seconds=time.perf_counter() - started,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"pod metric proxy failed for {pod}: {completed.stderr}")
        return evidence

    def autoscaler_conflicts(self) -> list[str]:
        conflicts: list[str] = []
        hpa = self.run(["get", "hpa", "-o", "json"], check=False)
        if hpa.returncode == 0:
            for item in json.loads(hpa.stdout).get("items", []):
                if item.get("spec", {}).get("scaleTargetRef", {}).get("name") == self.deployment:
                    conflicts.append(f"hpa/{item['metadata']['name']}")
        keda = self.run(["get", "scaledobjects.keda.sh", "-o", "json"], check=False)
        if keda.returncode == 0:
            for item in json.loads(keda.stdout).get("items", []):
                if item.get("spec", {}).get("scaleTargetRef", {}).get("name") == self.deployment:
                    conflicts.append(f"scaledobject/{item['metadata']['name']}")
        return conflicts

    def apply_file(self, path: str) -> CommandEvidence:
        return self.run(["apply", "-f", path])

    def delete_file(self, path: str) -> CommandEvidence:
        return self.run(["delete", "--ignore-not-found", "-f", path], check=False)

    def patch(self, resource: str, name: str, patch: dict[str, Any]) -> CommandEvidence:
        return self.run(
            ["patch", resource, name, "--type", "merge", "-p", json.dumps(patch, separators=(",", ":"))]
        )

    def api_available(self, group_version: str) -> bool:
        evidence = self.run(["get", "--raw", f"/apis/{group_version}"], check=False)
        return evidence.returncode == 0

    def resource_available(self, resource: str) -> bool:
        evidence = self.run(["api-resources", "--no-headers"], check=False)
        return evidence.returncode == 0 and any(
            line.split()[0] == resource for line in evidence.stdout.splitlines() if line.split()
        )

    def node_internal_ip(self) -> str:
        payload = self.run(["get", "nodes", "-o", "json"])
        for item in json.loads(payload.stdout).get("items", []):
            for address in item.get("status", {}).get("addresses", []):
                if address.get("type") == "InternalIP":
                    return str(address["address"])
        raise RuntimeError("Kubernetes node has no InternalIP")
