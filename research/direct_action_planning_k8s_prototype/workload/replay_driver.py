from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import aiohttp


def load_plan(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(rows[index]["scheduled_offset_seconds"] > rows[index + 1]["scheduled_offset_seconds"] for index in range(len(rows) - 1)):
        raise ValueError("request plan must be sorted by scheduled offset")
    return rows


async def replay(
    plan: list[dict[str, Any]],
    *,
    url: str,
    output: Path,
    timeout_seconds: float,
    connection_limit: int,
    force_close_connections: bool = True,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    started_wall = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    lock = asyncio.Lock()
    counts = {"scheduled": len(plan), "sent": 0, "completed": 0, "failed": 0, "timed_out": 0}
    # Kubernetes Service endpoint choice is connection-scoped. Reusing one
    # keep-alive connection can pin a whole trace to one Pod and invalidate a
    # replica-scaling comparison, so the default opens independent request
    # connections. This setting is recorded in every replay summary.
    connector = aiohttp.TCPConnector(
        limit=connection_limit,
        limit_per_host=connection_limit,
        force_close=force_close_connections,
    )
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with asyncio.TaskGroup() as group:
            for row in plan:
                group.create_task(_send_one(session, row, url, output, started, lock, counts))
    return {
        "schema": "dap.k8s.replay_summary.v1",
        **counts,
        "started_at": started_wall,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "url": url,
        "force_close_connections": bool(force_close_connections),
    }


async def _send_one(
    session: aiohttp.ClientSession,
    row: dict[str, Any],
    url: str,
    output: Path,
    started: float,
    lock: asyncio.Lock,
    counts: dict[str, int],
) -> None:
    delay = float(row["scheduled_offset_seconds"]) - (time.perf_counter() - started)
    if delay > 0:
        await asyncio.sleep(delay)
    sent_offset = time.perf_counter() - started
    counts["sent"] += 1
    status = 0
    error = None
    response_body: dict[str, Any] | None = None
    try:
        async with session.post(url, json={"payload": row["payload"]}) as response:
            status = response.status
            try:
                response_body = await response.json()
            except Exception:
                response_body = {"text": (await response.text())[:500]}
            if 200 <= status < 300:
                counts["completed"] += 1
            else:
                counts["failed"] += 1
    except asyncio.TimeoutError:
        error = "timeout"
        counts["timed_out"] += 1
        counts["failed"] += 1
    except Exception as exc:  # pragma: no cover - runtime network path
        error = f"{type(exc).__name__}:{exc}"
        counts["failed"] += 1
    completed_offset = time.perf_counter() - started
    event = {
        **row,
        "sent_offset_seconds": sent_offset,
        "completed_offset_seconds": completed_offset,
        "client_latency_seconds": completed_offset - sent_offset,
        "http_status": status,
        "error": error,
        "response": response_body,
    }
    async with lock:
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--connection-limit", type=int, default=512)
    parser.add_argument("--reuse-connections", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"append-only replay output exists: {args.output}")
    summary = asyncio.run(
        replay(
            load_plan(args.plan),
            url=args.url,
            output=args.output,
            timeout_seconds=args.timeout_seconds,
            connection_limit=args.connection_limit,
            force_close_connections=not args.reuse_connections,
        )
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["sent"] == summary["scheduled"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
