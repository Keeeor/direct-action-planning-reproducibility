"""Stable host access to the Service after the prototype rollout step."""

from __future__ import annotations

import os
import socket
from typing import Any, Callable
from urllib.parse import urlparse

from .prototype_api import PROTOTYPE_ROOT  # noqa: F401 - establishes import path

from experiments.runtime import PortForward


def allocate_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _extend_proxy_bypass(hostnames: tuple[str, ...]) -> dict[str, str | None]:
    previous = {name: os.environ.get(name) for name in ("NO_PROXY", "no_proxy")}
    for name in ("NO_PROXY", "no_proxy"):
        entries = [
            value.strip() for value in (os.environ.get(name) or "").split(",")
            if value.strip()
        ]
        for value in hostnames:
            if value not in entries:
                entries.append(value)
        os.environ[name] = ",".join(entries)
    return previous


def _restore_proxy_bypass(previous: dict[str, str | None] | None) -> None:
    if previous is None:
        return
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class ProxyBypassedPrepare:
    """Retain the original NodePort URL and bypass host HTTP proxies for it."""

    def __init__(self, *, original_prepare: Callable[[Any, str], str]):
        self.original_prepare = original_prepare
        self._proxy_environment: dict[str, str | None] | None = None

    def __call__(self, kube: Any, profile: str) -> str:
        url = self.original_prepare(kube, profile)
        hostname = urlparse(url).hostname
        if not hostname:
            raise ValueError(f"original preparation returned an invalid URL: {url}")
        self._proxy_environment = _extend_proxy_bypass((hostname,))
        return url

    def close(self) -> None:
        _restore_proxy_bypass(self._proxy_environment)
        self._proxy_environment = None


class LocalServiceForward:
    """Run the original rollout, then expose its Service on localhost.

    The kind NodePort can become unreachable from the host when Docker bridge
    forwarding rules change.  A post-rollout Service port-forward preserves the
    real Kubernetes Service/Pod path without changing the controller or worker.
    """

    def __init__(
        self,
        *,
        original_prepare: Callable[[Any, str], str],
        context: str,
        namespace: str,
        service: str,
        forward_factory: Callable[..., Any] = PortForward,
        port_allocator: Callable[[], int] = allocate_local_port,
    ):
        self.original_prepare = original_prepare
        self.context = str(context)
        self.namespace = str(namespace)
        self.service = str(service)
        self.forward_factory = forward_factory
        self.port_allocator = port_allocator
        self.forward: Any | None = None
        self._proxy_environment: dict[str, str | None] | None = None

    def _enable_local_proxy_bypass(self) -> None:
        self._proxy_environment = _extend_proxy_bypass(("127.0.0.1", "localhost"))

    def _restore_proxy_environment(self) -> None:
        _restore_proxy_bypass(self._proxy_environment)
        self._proxy_environment = None

    def __call__(self, kube: Any, profile: str) -> str:
        self.original_prepare(kube, profile)
        port = int(self.port_allocator())
        forward = self.forward_factory(
            context=self.context,
            namespace=self.namespace,
            service=self.service,
            local_port=port,
            remote_port=8000,
        )
        forward.__enter__()
        self.forward = forward
        self._enable_local_proxy_bypass()
        return f"http://127.0.0.1:{port}"

    def close(self) -> None:
        if self.forward is not None:
            self.forward.close()
            self.forward = None
        self._restore_proxy_environment()
