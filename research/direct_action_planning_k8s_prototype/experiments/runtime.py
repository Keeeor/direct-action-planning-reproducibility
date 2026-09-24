from __future__ import annotations

from contextlib import AbstractContextManager
import json
from pathlib import Path
import socket
import subprocess
import time
from urllib.request import urlopen


class PortForward(AbstractContextManager["PortForward"]):
    def __init__(self, *, context: str, namespace: str, service: str, local_port: int, remote_port: int):
        self.argv = [
            "kubectl", "--context", context, "-n", namespace, "port-forward",
            f"service/{service}", f"{local_port}:{remote_port}",
        ]
        self.local_port = int(local_port)
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "PortForward":
        self.process = subprocess.Popen(
            self.argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(f"port-forward exited early: {stdout}\n{stderr}")
            try:
                with socket.create_connection(("127.0.0.1", self.local_port), timeout=0.2):
                    return self
            except OSError:
                time.sleep(0.2)
        self.close()
        raise TimeoutError("port-forward did not become reachable")

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

    def __exit__(self, *_: object) -> None:
        self.close()


def wait_http(url: str, timeout_seconds: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                return json.loads(response.read())
        except Exception as exc:  # pragma: no cover - real network path
            last_error = exc
            time.sleep(0.5)
    raise TimeoutError(f"HTTP endpoint did not become healthy: {last_error}")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
