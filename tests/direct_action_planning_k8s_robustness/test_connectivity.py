from __future__ import annotations

import os

from stage2_dynamic_budget.direct_action_planning_k8s_robustness.connectivity import (
    LocalServiceForward,
    ProxyBypassedPrepare,
)


def test_prepare_finishes_rollout_before_opening_local_service_forward():
    events: list[str] = []

    def original_prepare(kube, profile):
        events.append(f"prepare:{kube}:{profile}")
        return "http://unreachable-node:30080"

    class FakeForward:
        local_port = 38123

        def __enter__(self):
            events.append("forward-enter")
            return self

        def close(self):
            events.append("forward-close")

    wrapper = LocalServiceForward(
        original_prepare=original_prepare,
        context="ctx",
        namespace="ns",
        service="svc",
        forward_factory=lambda **kwargs: FakeForward(),
        port_allocator=lambda: 38123,
    )
    assert wrapper("kube", "profile") == "http://127.0.0.1:38123"
    assert events == ["prepare:kube:profile", "forward-enter"]
    wrapper.close()
    assert events[-1] == "forward-close"


def test_local_service_forward_bypasses_process_http_proxy_and_restores_environment(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "example.org")
    monkeypatch.delenv("no_proxy", raising=False)

    class FakeForward:
        def __enter__(self):
            return self

        def close(self):
            pass

    wrapper = LocalServiceForward(
        original_prepare=lambda _kube, _profile: "http://unreachable-node:30080",
        context="ctx",
        namespace="ns",
        service="svc",
        forward_factory=lambda **_kwargs: FakeForward(),
        port_allocator=lambda: 38123,
    )
    wrapper("kube", "profile")
    assert "127.0.0.1" in os.environ["NO_PROXY"].split(",")
    assert "localhost" in os.environ["NO_PROXY"].split(",")
    assert "127.0.0.1" in os.environ["no_proxy"].split(",")

    wrapper.close()
    assert os.environ["NO_PROXY"] == "example.org"
    assert "no_proxy" not in os.environ


def test_proxy_bypassed_prepare_retains_original_nodeport_and_restores_environment(monkeypatch):
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setenv("no_proxy", "example.org")
    calls: list[tuple[object, str]] = []

    def original_prepare(kube, profile):
        calls.append((kube, profile))
        return "http://172.18.0.2:30080"

    wrapper = ProxyBypassedPrepare(original_prepare=original_prepare)
    assert wrapper("kube", "azure_http") == "http://172.18.0.2:30080"
    assert calls == [("kube", "azure_http")]
    assert "172.18.0.2" in os.environ["NO_PROXY"].split(",")
    assert "172.18.0.2" in os.environ["no_proxy"].split(",")

    wrapper.close()
    assert "NO_PROXY" not in os.environ
    assert os.environ["no_proxy"] == "example.org"
