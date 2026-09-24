from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ServiceSettings:
    profile: str
    max_concurrency: int
    max_queue_depth: int
    cpu_iterations: int
    minimum_service_seconds: float
    request_timeout_seconds: float
    slo_seconds: float

    @classmethod
    def from_environment(cls) -> "ServiceSettings":
        profile = os.getenv("SERVICE_PROFILE", "azure_http").strip().lower()
        defaults = {
            "azure_http": {
                "max_concurrency": 8,
                "max_queue_depth": 256,
                "cpu_iterations": 8_000,
                "minimum_service_seconds": 0.006,
                "request_timeout_seconds": 5.0,
                "slo_seconds": 0.25,
            },
            "gentd_inference": {
                "max_concurrency": 2,
                "max_queue_depth": 128,
                "cpu_iterations": 70_000,
                "minimum_service_seconds": 0.045,
                "request_timeout_seconds": 12.0,
                "slo_seconds": 1.0,
            },
        }
        if profile not in defaults:
            raise ValueError(f"unsupported SERVICE_PROFILE={profile!r}")
        values = defaults[profile]
        return cls(
            profile=profile,
            max_concurrency=_positive_int("MAX_CONCURRENCY", values["max_concurrency"]),
            max_queue_depth=_positive_int("MAX_QUEUE_DEPTH", values["max_queue_depth"]),
            cpu_iterations=_positive_int("CPU_ITERATIONS", values["cpu_iterations"]),
            minimum_service_seconds=_nonnegative_float(
                "MINIMUM_SERVICE_SECONDS", values["minimum_service_seconds"]
            ),
            request_timeout_seconds=_positive_float(
                "REQUEST_TIMEOUT_SECONDS", values["request_timeout_seconds"]
            ),
            slo_seconds=_positive_float("SLO_SECONDS", values["slo_seconds"]),
        )


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value
