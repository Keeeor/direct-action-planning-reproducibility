from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, CONTENT_TYPE_LATEST, generate_latest

from .settings import ServiceSettings


REQUESTS_RECEIVED = Counter(
    "requests_received_total", "HTTP inference requests accepted or rejected"
)
REQUESTS_COMPLETED = Counter(
    "requests_completed_total", "Inference jobs completed by a worker"
)
REQUESTS_FAILED = Counter(
    "requests_failed_total", "Inference requests rejected, timed out, or failed", ["reason"]
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "End-to-end request latency including queueing",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
REQUEST_SLO_VIOLATIONS = Counter(
    "request_slo_violations_total", "Completed or timed-out requests exceeding the profile SLO"
)
QUEUE_DEPTH = Gauge("queue_depth", "Number of jobs waiting for a worker")
INFLIGHT_REQUESTS = Gauge("inflight_requests", "Number of jobs currently executing")
SERVICE_TIME = Histogram(
    "service_time_seconds",
    "Worker execution time including the configured minimum service time",
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)


class InferRequest(BaseModel):
    payload: str = Field(default="", max_length=4096)
    work_units: int | None = Field(default=None, ge=1, le=10)


@dataclass
class WorkItem:
    payload: bytes
    iterations: int
    accepted_at: float
    future: asyncio.Future[dict[str, Any]]


class WorkerPool:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.queue: asyncio.Queue[WorkItem] = asyncio.Queue(settings.max_queue_depth)
        self.tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        self.tasks = [
            asyncio.create_task(self._worker(index), name=f"cpu-worker-{index}")
            for index in range(self.settings.max_concurrency)
        ]

    async def stop(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

    async def submit(self, request: InferRequest) -> dict[str, Any]:
        REQUESTS_RECEIVED.inc()
        if self.queue.full():
            REQUESTS_FAILED.labels(reason="queue_full").inc()
            raise HTTPException(status_code=503, detail="worker queue is full")
        loop = asyncio.get_running_loop()
        payload = request.payload.encode("utf-8") or os.urandom(32)
        multiplier = request.work_units or 1
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        item = WorkItem(
            payload=payload,
            iterations=self.settings.cpu_iterations * multiplier,
            accepted_at=time.perf_counter(),
            future=future,
        )
        await self.queue.put(item)
        QUEUE_DEPTH.set(self.queue.qsize())
        try:
            result = await asyncio.wait_for(
                asyncio.shield(future), timeout=self.settings.request_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            REQUESTS_FAILED.labels(reason="request_timeout").inc()
            latency = time.perf_counter() - item.accepted_at
            REQUEST_LATENCY.observe(latency)
            if latency > self.settings.slo_seconds:
                REQUEST_SLO_VIOLATIONS.inc()
            raise HTTPException(status_code=504, detail="inference timed out") from exc
        latency = time.perf_counter() - item.accepted_at
        REQUEST_LATENCY.observe(latency)
        if latency > self.settings.slo_seconds:
            REQUEST_SLO_VIOLATIONS.inc()
        return result

    async def _worker(self, index: int) -> None:
        while True:
            item = await self.queue.get()
            QUEUE_DEPTH.set(self.queue.qsize())
            INFLIGHT_REQUESTS.inc()
            started = time.perf_counter()
            try:
                digest = await asyncio.to_thread(_cpu_work, item.payload, item.iterations)
                elapsed = time.perf_counter() - started
                remaining = self.settings.minimum_service_seconds - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)
                service_seconds = time.perf_counter() - started
                SERVICE_TIME.observe(service_seconds)
                REQUESTS_COMPLETED.inc()
                if not item.future.done():
                    item.future.set_result(
                        {
                            "digest": digest,
                            "worker": index,
                            "service_seconds": service_seconds,
                            "profile": self.settings.profile,
                        }
                    )
            except Exception as exc:  # pragma: no cover - defensive runtime path
                REQUESTS_FAILED.labels(reason="worker_error").inc()
                if not item.future.done():
                    item.future.set_exception(exc)
            finally:
                INFLIGHT_REQUESTS.dec()
                self.queue.task_done()


def _cpu_work(payload: bytes, iterations: int) -> str:
    digest = hashlib.sha256(payload).digest()
    for index in range(iterations):
        digest = hashlib.sha256(digest + index.to_bytes(4, "little", signed=False)).digest()
    return digest.hex()


settings = ServiceSettings.from_environment()
pool = WorkerPool(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await pool.start()
    try:
        yield
    finally:
        await pool.stop()


app = FastAPI(title="DAP Kubernetes CPU Worker", version="1.0", lifespan=lifespan)


@app.post("/infer")
async def infer(request: InferRequest) -> dict[str, Any]:
    return await pool.submit(request)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "profile": settings.profile,
        "queue_depth": pool.queue.qsize(),
        "max_concurrency": settings.max_concurrency,
        "slo_seconds": settings.slo_seconds,
    }


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
