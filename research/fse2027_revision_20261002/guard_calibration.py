"""Measure direct scale API, Ready withdrawal and Pod deletion separately.

Only the revision namespace is modified. Calibration precedes policy outcomes.
The first five repetitions per profile/target calibrate; the next five validate.
No finite empirical guard is asserted to be an unconditional safety guarantee.
"""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import time

HERE = Path(__file__).resolve().parent
NS = "dap-fse-revision-20261002"
K = str(HERE / "bin/kubectl")
OUT = HERE / "guard_measurements"


def call(*args):
    return subprocess.run([K, "-n", NS, *args], check=True, capture_output=True, text=True).stdout


def snapshot():
    start = time.monotonic()
    d = json.loads(call("get", "deployment", "dap-worker", "-o", "json"))
    pods = json.loads(call("get", "pods", "-l", "app.kubernetes.io/name=dap-worker", "-o", "json"))["items"]
    return {"start": start, "end": time.monotonic(), "deployment_ready": d.get("status", {}).get("readyReplicas", 0), "desired": d["spec"]["replicas"], "pod_ready": sum(any(c.get("type") == "Ready" and c.get("status") == "True" for c in p.get("status", {}).get("conditions", [])) for p in pods), "pod_count": len(pods), "pods": [{"name": p["metadata"]["name"], "deleted": p["metadata"].get("deletionTimestamp"), "phase": p.get("status", {}).get("phase"), "conditions": p.get("status", {}).get("conditions", [])} for p in pods]}


def transition(target, direction):
    before = snapshot()
    started = time.monotonic()
    call("scale", "deployment/dap-worker", f"--replicas={target}")
    api_finished = time.monotonic()
    samples = [before]
    ready_reached = None
    end = None
    while time.monotonic() - started < 60:
        state = snapshot()
        samples.append(state)
        ready = min(state["deployment_ready"], state["pod_ready"])
        complete = ready >= target if direction == "up" else max(state["deployment_ready"], state["pod_ready"]) <= target
        if complete and ready_reached is None:
            ready_reached = state["end"]
        if complete and state["pod_count"] == target:
            end = state["end"]
            break
        time.sleep(.1)
    if ready_reached is None or end is None:
        raise RuntimeError(f"transition did not settle: target={target}, direction={direction}")
    gaps = [b["end"] - a["end"] for a,b in zip(samples,samples[1:])]
    return {"target": target, "direction": direction, "started_monotonic": started, "api_seconds": api_finished-started, "ready_reached_seconds": ready_reached-started, "pod_count_settled_seconds": end-started, "maximum_sampling_gap_seconds": max(gaps,default=0), "samples": samples}


def main():
    OUT.mkdir(exist_ok=True)
    for profile in ("azure_http", "gentd_inference"):
        call("set", "env", "deployment/dap-worker", "SERVICE_PROFILE=" + profile)
        call("scale", "deployment/dap-worker", "--replicas=1")
        call("rollout", "status", "deployment/dap-worker", "--timeout=90s")
        for repetition in range(10):
            for target in (2,3,5):
                path = OUT / f"{profile}__r{repetition:02d}__t{target}.json"
                if path.exists():
                    continue
                row = {"profile":profile, "repetition":repetition, "target":target, "partition":"calibration" if repetition<5 else "validation", "utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
                row["up"] = transition(target,"up")
                time.sleep(.5)
                row["down"] = transition(1,"down")
                path.write_text(json.dumps(row,indent=2))
                print(json.dumps({k:row[k] for k in ("profile","repetition","target","partition")}|{"up":row["up"]["ready_reached_seconds"],"down":row["down"]["ready_reached_seconds"]}),flush=True)
    rows=[json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    summary={"schema":"dap.fse2027.guard_measurements.v1", "scope":"Direct scale requests on one shared kind node, without a dedicated foreground workload; Ready cost and Pod lifetime are separate quantities.", "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "records":len(rows),"profiles":{}}
    for profile in ("azure_http","gentd_inference"):
        cal=[r for r in rows if r["profile"]==profile and r["partition"]=="calibration"]
        val=[r for r in rows if r["profile"]==profile and r["partition"]=="validation"]
        guard=max(r["down"]["ready_reached_seconds"]+r["down"]["maximum_sampling_gap_seconds"] for r in cal)+1.0
        summary["profiles"][profile]={"calibration_n":len(cal),"validation_n":len(val),"empirical_guard_seconds":guard,"validation_exceedances":sum(r["down"]["ready_reached_seconds"]>guard for r in val),"validation_max_ready_withdrawal_seconds":max(r["down"]["ready_reached_seconds"] for r in val),"validation_max_pod_deletion_seconds":max(r["down"]["pod_count_settled_seconds"] for r in val)}
    (HERE/"guard_summary.json").write_text(json.dumps(summary,indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        call("scale", "deployment/dap-worker", "--replicas=1")
