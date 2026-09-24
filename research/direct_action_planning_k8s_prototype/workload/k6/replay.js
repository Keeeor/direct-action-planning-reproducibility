import http from "k6/http";
import { check, sleep } from "k6";
import { SharedArray } from "k6/data";

const rows = new SharedArray("request plan", function () {
  return open(__ENV.PLAN_PATH).trim().split("\n").map((line) => JSON.parse(line));
});

export const options = {
  scenarios: {
    replay: {
      executor: "shared-iterations",
      vus: Number(__ENV.VUS || 64),
      iterations: rows.length,
      maxDuration: __ENV.MAX_DURATION || "20m",
    },
  },
};

const started = Date.now() / 1000;

export default function () {
  const row = rows[exec.scenario.iterationInTest];
  const delay = row.scheduled_offset_seconds - (Date.now() / 1000 - started);
  if (delay > 0) sleep(delay);
  const response = http.post(
    __ENV.TARGET_URL,
    JSON.stringify({ payload: row.payload }),
    { headers: { "Content-Type": "application/json" }, timeout: __ENV.REQUEST_TIMEOUT || "15s" },
  );
  check(response, { "inference completed": (r) => r.status === 200 });
}

