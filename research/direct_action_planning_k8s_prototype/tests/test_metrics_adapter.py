from app.metrics_adapter.core import ProcessSample, UsageCache, parse_process_metrics, pod_metrics_list


def test_process_metrics_parse_real_prometheus_style_lines() -> None:
    cpu, memory = parse_process_metrics(
        "# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds.\n"
        "process_cpu_seconds_total 7.25\n"
        "process_resident_memory_bytes 1048576.0\n"
    )
    assert cpu == 7.25
    assert memory == 1048576.0


def test_usage_cache_converts_cumulative_cpu_to_rate() -> None:
    cache = UsageCache()
    cache.record(ProcessSample("worker-a", 10.0, 4.0, 100.0))
    usage = cache.record(ProcessSample("worker-a", 12.0, 5.0, 120.0))
    assert usage.window_seconds == 2.0
    assert usage.cpu_cores == 0.5
    payload = pod_metrics_list("prototype", [usage])
    assert payload["items"][0]["containers"][0]["usage"] == {"cpu": "500000000n", "memory": "120"}


def test_usage_cache_removes_deleted_pods() -> None:
    cache = UsageCache()
    cache.record(ProcessSample("gone", 1.0, 1.0, 1.0))
    cache.record(ProcessSample("live", 1.0, 1.0, 1.0))
    cache.remove_except({"live"})
    assert [item.pod for item in cache.usages()] == ["live"]
