#!/usr/bin/env python3
"""用四张真实数据集图像同时验证模型，并观测实际运行请求数与显存。"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "http://127.0.0.1:19888"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
samples = []
for dataset in ("DGM4-author1000", "MMFakeBench-author1000"):
    records = json.loads((ROOT / "data" / dataset / "samples.json").read_text())
    samples.extend(records[:2])

barrier = threading.Barrier(len(samples))
done = threading.Event()
observations = []


def memory_snapshot():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits",
    ], text=True)
    return [
        {"index": int(row[0]), "used_mib": int(row[1]), "total_mib": int(row[2])}
        for line in output.splitlines() if (row := line.split(","))
    ]


def observe():
    last_gpu = 0.0
    while not done.is_set():
        item = {"time_unix": time.time()}
        try:
            with OPENER.open(BASE_URL + "/metrics", timeout=2) as response:
                metrics = response.read().decode()
            running = re.findall(r'^vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9.eE+-]+)', metrics, re.M)
            waiting = re.findall(r'^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9.eE+-]+)', metrics, re.M)
            item["running_requests"] = sum(float(value) for value in running)
            item["waiting_requests"] = sum(float(value) for value in waiting)
            if time.monotonic() - last_gpu >= 1:
                item["gpu"] = memory_snapshot()
                last_gpu = time.monotonic()
        except Exception as exc:
            item["monitor_error"] = type(exc).__name__
        observations.append(item)
        done.wait(0.2)


def request(sample):
    image_path = Path(sample["image_path"])
    data = image_path.read_bytes()
    image_url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
    payload = {
        "model": "Qwen3.8-27B",
        "messages": [{
            "role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Describe the visible image in about 100 words. Return only a JSON object with the keys main_subject and visible_details. Do not assess any associated news claim."},
            ],
        }],
        "temperature": 0,
        "max_tokens": 384,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    row = {
        "sample_id": sample["sample_id"],
        "image_sha256": hashlib.sha256(data).hexdigest(),
        "labels_sent": False,
    }
    barrier.wait()
    started = time.monotonic()
    row["started_at_unix"] = time.time()
    try:
        request_object = urllib.request.Request(
            BASE_URL + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with OPENER.open(request_object, timeout=300) as response:
            result = json.load(response)
        choice = result["choices"][0]
        content = choice["message"]["content"]
        row.update(
            content=content, parsed_json=json.loads(content),
            usage=result.get("usage"), finish_reason=choice["finish_reason"],
            status="success" if choice["finish_reason"] == "stop" else "incomplete",
        )
    except Exception as exc:
        row.update(status="error", error_type=type(exc).__name__, error=str(exc)[:300])
    row["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return row


before = memory_snapshot()
monitor = threading.Thread(target=observe, daemon=True)
monitor.start()
started = time.monotonic()
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(samples)) as pool:
        results = list(pool.map(request, samples))
finally:
    done.set()
    monitor.join(timeout=5)

report = {
    "model": "Qwen3.8-27B",
    "weights": "Qwen/Qwen3.8-27B-FP8",
    "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
    "base_url": BASE_URL + "/v1",
    "concurrency_requested": len(samples),
    "max_running_requests_observed": max((row.get("running_requests", 0) for row in observations), default=0),
    "elapsed_seconds": round(time.monotonic() - started, 3),
    "success_count": sum(row["status"] == "success" for row in results),
    "gpu_before": before,
    "gpu_after": memory_snapshot(),
    "peak_gpu_used_mib": {
        str(index): max(
            (gpu["used_mib"] for row in observations for gpu in row.get("gpu", []) if gpu["index"] == index),
            default=0,
        )
        for index in (0, 1)
    },
    "results": results,
    "observations": observations,
}
report["passed"] = report["success_count"] == len(samples) and report["max_running_requests_observed"] >= 4
(ROOT / "outputs" / "model-concurrency-smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({
    key: report[key] for key in (
        "passed", "concurrency_requested", "max_running_requests_observed",
        "elapsed_seconds", "success_count", "peak_gpu_used_mib",
    )
}, ensure_ascii=False))
if not report["passed"]:
    raise SystemExit(1)
