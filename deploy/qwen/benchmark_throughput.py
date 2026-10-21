#!/usr/bin/env python3
"""使用固定输入和固定生成 token 数比较本地模型并发吞吐。"""
from __future__ import annotations

import argparse
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
BASE = "http://127.0.0.1:19888"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--concurrency", type=int, required=True)
parser.add_argument("--tokens", type=int, default=256)
args = parser.parse_args()
sample = json.loads((ROOT / "data/DGM4-author1000/samples.json").read_text())[0]
image_bytes = Path(sample["image_path"]).read_bytes()
prompt = (
    "Describe all visible people, clothing, objects and background details in this image. "
    "Write a detailed factual description using only visible evidence and avoiding speculation."
)
payload = {
    "model": "Qwen3.8-27B",
    "messages": [{
        "role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()}},
            {"type": "text", "text": prompt},
        ],
    }],
    "temperature": 0,
    "max_tokens": args.tokens,
    "min_tokens": args.tokens,
    "ignore_eos": True,
    "chat_template_kwargs": {"enable_thinking": False},
}


def metrics():
    with OPENER.open(BASE + "/metrics", timeout=5) as response:
        text = response.read().decode()
    output = {}
    for name in ("num_requests_running", "num_requests_waiting", "num_preemptions_total", "kv_cache_usage_perc"):
        values = re.findall(rf'^vllm:{name}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)', text, re.M)
        if values:
            output[name] = sum(float(value) for value in values)
    return output


def gpu():
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits",
    ], text=True)
    return [
        {"index": int(parts[0]), "used_mib": int(parts[1]), "total_mib": int(parts[2])}
        for line in text.splitlines() if (parts := line.split(","))
    ]


def complete(body):
    request = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with OPENER.open(request, timeout=300) as response:
        return json.load(response)


# 两个并发设置都用相同的短请求预热图像编码和前缀。
warm_payload = {**payload, "max_tokens": 32, "min_tokens": 32}
warmup_started = time.monotonic()
warmup = complete(warm_payload)
warmup_seconds = time.monotonic() - warmup_started
before = metrics()
barrier = threading.Barrier(args.concurrency)
stop = threading.Event()
observations = []


def monitor():
    while not stop.is_set():
        try:
            observations.append({"time_unix": time.time(), **metrics(), "gpu": gpu()})
        except Exception as exc:
            observations.append({"monitor_error": type(exc).__name__})
        stop.wait(0.3)


def one(index):
    barrier.wait()
    started = time.monotonic()
    try:
        result = complete(payload)
        row = {
            "index": index, "status": "success",
            "usage": result.get("usage"),
            "finish_reason": result["choices"][0]["finish_reason"],
        }
    except Exception as exc:
        row = {"index": index, "status": "error", "error_type": type(exc).__name__, "error": str(exc)[:300]}
    row["elapsed_seconds"] = round(time.monotonic() - started, 4)
    return row


observer = threading.Thread(target=monitor, daemon=True)
observer.start()
batch_started = time.monotonic()
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(one, range(args.concurrency)))
    wall = time.monotonic() - batch_started
finally:
    stop.set()
    observer.join(timeout=10)
after = metrics()
tokens = sum(row.get("usage", {}).get("completion_tokens", 0) for row in rows)
report = {
    "model": "Qwen3.8-27B",
    "server_launch": json.loads((ROOT / "outputs/model-launch.json").read_text()),
    "concurrency": args.concurrency,
    "output_tokens_per_request_target": args.tokens,
    "total_output_tokens": tokens,
    "batch_wall_seconds": round(wall, 4),
    "aggregate_output_tokens_per_second": round(tokens / wall, 3),
    "warmup_seconds": round(warmup_seconds, 4),
    "input_image_sha256": hashlib.sha256(image_bytes).hexdigest(),
    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    "labels_sent": False,
    "sampling": {"temperature": 0, "max_tokens": args.tokens, "min_tokens": args.tokens, "ignore_eos": True, "enable_thinking": False},
    "max_running_observed": max((row.get("num_requests_running", 0) for row in observations), default=0),
    "max_kv_cache_usage_observed": max((row.get("kv_cache_usage_perc", 0) for row in observations), default=0),
    "preemptions_delta": after.get("num_preemptions_total", 0) - before.get("num_preemptions_total", 0),
    "peak_gpu_used_mib": {
        str(index): max(
            (g["used_mib"] for row in observations for g in row.get("gpu", []) if g["index"] == index),
            default=0,
        ) for index in (0, 1)
    },
    "requests": rows,
    "observations": observations,
}
report["passed"] = (
    all(row["status"] == "success" and row["usage"]["completion_tokens"] == args.tokens for row in rows)
    and report["max_running_observed"] == args.concurrency
    and report["preemptions_delta"] == 0
)
output = ROOT / "outputs" / f"model-throughput-{args.concurrency}.json"
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({key: report[key] for key in (
    "concurrency", "passed", "total_output_tokens", "batch_wall_seconds",
    "aggregate_output_tokens_per_second", "max_running_observed",
    "max_kv_cache_usage_observed", "preemptions_delta", "peak_gpu_used_mib",
)}, ensure_ascii=False))
if not report["passed"]:
    raise SystemExit(1)
