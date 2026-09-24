from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import ssl
import time
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException, Request as FastAPIRequest

from .core import ProcessSample, PodUsage, UsageCache, parse_process_metrics, pod_metrics_list

class KubernetesWorkerMetricsSource:
    """Read actual metrics through the in-cluster Kubernetes Pod proxy."""

    def __init__(self) -> None:
        self.namespace = os.environ.get("METRICS_TARGET_NAMESPACE", "dap-k8s-prototype")
        self.selector = os.environ.get(
            "METRICS_TARGET_LABEL_SELECTOR", "app.kubernetes.io/name=dap-worker"
        )
        self.port = int(os.environ.get("METRICS_TARGET_PORT", "8000"))
        self.api_server = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        self.api_port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        self.token_path = Path(
            os.environ.get(
                "SERVICE_ACCOUNT_TOKEN_PATH", "/var/run/secrets/kubernetes.io/serviceaccount/token"
            )
        )
        self.ca_path = Path(
            os.environ.get(
                "SERVICE_ACCOUNT_CA_PATH", "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
            )
        )

    @property
    def base_url(self) -> str:
        return f"https://{self.api_server}:{self.api_port}"

    def _request(self, path: str) -> bytes:
        token = self.token_path.read_text(encoding="utf-8").strip()
        request = Request(
            self.base_url + path,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        context = ssl.create_default_context(cafile=str(self.ca_path))
        with urlopen(request, context=context, timeout=5.0) as response:
            return response.read()

    def ready_worker_pods(self) -> list[str]:
        selector = quote(self.selector, safe="")
        payload = json.loads(
            self._request(
                f"/api/v1/namespaces/{quote(self.namespace, safe='')}/pods?labelSelector={selector}"
            )
        )
        names: list[str] = []
        for item in payload.get("items", []):
            conditions = {
                row.get("type"): row.get("status")
                for row in item.get("status", {}).get("conditions", [])
            }
            if item.get("status", {}).get("phase") == "Running" and conditions.get("Ready") == "True":
                names.append(str(item["metadata"]["name"]))
        return sorted(names)

    def metrics_text(self, pod: str) -> str:
        path = (
            f"/api/v1/namespaces/{quote(self.namespace, safe='')}/pods/"
            f"{quote(pod, safe='')}:{self.port}/proxy/metrics"
        )
        return self._request(path).decode("utf-8")


class LiveMetricsAdapter:
    def __init__(self, source: KubernetesWorkerMetricsSource, interval_seconds: float) -> None:
        self.source = source
        self.interval_seconds = float(interval_seconds)
        self.cache = UsageCache()
        self.last_success_seconds: float | None = None
        self.last_error: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    def _sample_blocking(self) -> None:
        pods = self.source.ready_worker_pods()
        seen = set(pods)
        now = time.time()
        for pod in pods:
            cpu_seconds, memory_bytes = parse_process_metrics(self.source.metrics_text(pod))
            self.cache.record(
                ProcessSample(
                    pod=pod,
                    observed_seconds=now,
                    cpu_seconds=cpu_seconds,
                    memory_bytes=memory_bytes,
                )
            )
        self.cache.remove_except(seen)

    async def sample_once(self) -> None:
        async with self._lock:
            try:
                await asyncio.to_thread(self._sample_blocking)
            except Exception as exc:  # pragma: no cover - depends on live API failures
                self.last_error = f"{type(exc).__name__}: {exc}"
                return
            self.last_success_seconds = time.time()
            self.last_error = None

    async def start(self) -> None:
        await self.sample_once()
        self._task = asyncio.create_task(self._sample_loop(), name="metrics-api-sampler")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _sample_loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval_seconds)
            await self.sample_once()

    async def pod_metrics(self) -> list[PodUsage]:
        async with self._lock:
            return self.cache.usages()

    def ready(self) -> bool:
        if self.last_success_seconds is None:
            return False
        return time.time() - self.last_success_seconds <= max(15.0, 3.0 * self.interval_seconds)


def create_app() -> FastAPI:
    source = KubernetesWorkerMetricsSource()
    interval = float(os.environ.get("METRICS_SAMPLE_INTERVAL_SECONDS", "2"))
    adapter = LiveMetricsAdapter(source, interval)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.adapter = adapter
        await adapter.start()
        try:
            yield
        finally:
            await adapter.stop()

    app = FastAPI(title="DAP Local Kubernetes Metrics API", lifespan=lifespan)

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        if not adapter.ready():
            raise HTTPException(status_code=503, detail=adapter.last_error or "no successful metrics sample")
        return {"status": "ok"}

    @app.get("/apis/metrics.k8s.io")
    async def group() -> dict[str, Any]:
        return {
            "kind": "APIGroup",
            "apiVersion": "v1",
            "name": "metrics.k8s.io",
            "versions": [{"groupVersion": "metrics.k8s.io/v1beta1", "version": "v1beta1"}],
            "preferredVersion": {"groupVersion": "metrics.k8s.io/v1beta1", "version": "v1beta1"},
        }

    @app.get("/apis/metrics.k8s.io/v1beta1")
    async def resources() -> dict[str, Any]:
        return {
            "kind": "APIResourceList",
            "apiVersion": "v1",
            "groupVersion": "metrics.k8s.io/v1beta1",
            "resources": [
                {"name": "pods", "singularName": "", "namespaced": True, "kind": "PodMetrics", "verbs": ["get", "list"]}
            ],
        }

    async def metrics_for_namespace(namespace: str) -> dict[str, Any]:
        if namespace != source.namespace:
            return pod_metrics_list(namespace, [])
        return pod_metrics_list(namespace, await adapter.pod_metrics())

    @app.get("/apis/metrics.k8s.io/v1beta1/namespaces/{namespace}/pods")
    async def pods(namespace: str, request: FastAPIRequest) -> dict[str, Any]:
        del request  # The source's fixed label selector bounds this prototype API.
        return await metrics_for_namespace(namespace)

    @app.get("/apis/metrics.k8s.io/v1beta1/namespaces/{namespace}/pods/{pod}")
    async def pod(namespace: str, pod: str) -> dict[str, Any]:
        rows = await metrics_for_namespace(namespace)
        for row in rows["items"]:
            if row["metadata"]["name"] == pod:
                row["kind"] = "PodMetrics"
                row["apiVersion"] = "metrics.k8s.io/v1beta1"
                return row
        raise HTTPException(status_code=404, detail=f"metrics for pod {pod!r} are unavailable")

    return app


app = create_app()
