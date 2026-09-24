from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any


def readiness_patch_payload(*, container: str, initial_delay_seconds: int) -> dict[str, Any]:
    if int(initial_delay_seconds) < 0:
        raise ValueError("readiness delay must be non-negative")
    return {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": str(container),
                            "readinessProbe": {
                                "initialDelaySeconds": int(initial_delay_seconds)
                            },
                        }
                    ]
                }
            }
        }
    }


@dataclass
class ReadinessDelayController:
    context: str
    namespace: str
    deployment: str
    container: str = "worker"

    def _base(self) -> list[str]:
        return [
            "kubectl", "--context", self.context, "-n", self.namespace,
        ]

    def current(self) -> int:
        result = subprocess.run(
            self._base()
            + [
                "get", "deployment", self.deployment,
                "-o", "jsonpath={.spec.template.spec.containers[?(@.name==\"worker\")].readinessProbe.initialDelaySeconds}",
            ],
            text=True, capture_output=True, check=True,
        )
        return int(result.stdout.strip())

    def set(self, value: int) -> None:
        payload = readiness_patch_payload(
            container=self.container, initial_delay_seconds=int(value)
        )
        subprocess.run(
            self._base()
            + [
                "patch", "deployment", self.deployment, "--type=strategic",
                "-p", json.dumps(payload, separators=(",", ":")),
            ],
            text=True, capture_output=True, check=True,
        )
        subprocess.run(
            self._base()
            + ["rollout", "status", f"deployment/{self.deployment}", "--timeout=180s"],
            text=True, capture_output=True, check=True,
        )
