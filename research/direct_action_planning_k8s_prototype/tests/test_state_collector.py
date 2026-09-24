import math

import pytest

from controller.state_collector import StateCollector, histogram_quantile, parse_prometheus_text


def test_prometheus_parser_preserves_labels_and_special_bounds() -> None:
    samples = parse_prometheus_text(
        '# HELP ignored x\nrequest_latency_seconds_bucket{le="0.25"} 4\n'
        'request_latency_seconds_bucket{le="+Inf"} 5\nqueue_depth 3\n'
    )
    assert len(samples) == 3
    assert samples[0].labels == {"le": "0.25"}
    assert math.isinf(float(samples[1].labels["le"]))
    assert samples[2].value == 3.0


def test_histogram_quantile_interpolates_cumulative_buckets() -> None:
    assert histogram_quantile({0.1: 5, 0.2: 10, math.inf: 10}, 0.75) == pytest.approx(0.15)


def test_state_collector_uses_calibrated_non_linear_capacity_when_available() -> None:
    """Online state semantics must match the calibrated planning model."""

    collector = StateCollector(
        kube=object(),  # The capacity lookup is pure and needs no live client.
        capacity_per_pod_rps=100.0,
        capacity_by_ready_replicas={1: 100.0, 2: 180.0, 3: 230.0, 5: 240.0},
    )
    assert collector.capacity_for_ready_replicas(1) == pytest.approx(100.0)
    assert collector.capacity_for_ready_replicas(3) == pytest.approx(230.0)
    assert collector.capacity_for_ready_replicas(4) == pytest.approx(235.0)
