#!/usr/bin/env python3
"""以独立后台进程启动本项目模型服务，保存 PID 与日志。"""

import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PID_PATH = ROOT / "outputs" / "model-server.pid"
LOG_PATH = ROOT / "outputs" / "model-server.log"

if PID_PATH.exists():
    old_pid = int(PID_PATH.read_text().strip())
    command_path = Path(f"/proc/{old_pid}/cmdline")
    if command_path.exists():
        command = command_path.read_bytes()
        if str(ROOT).encode() in command and b"vllm" in command:
            print(f"模型进程已存在：PID {old_pid}")
            sys.exit(0)
try:
    with socket.create_connection(("127.0.0.1", 19888), timeout=2):
        raise SystemExit("端口 19888 已有服务，未启动新进程")
except OSError:
    pass

ROOT.joinpath("outputs").mkdir(exist_ok=True)
with LOG_PATH.open("ab") as log:
    process = subprocess.Popen(
        ["bash", str(ROOT / "scripts" / "serve_qwen.sh")],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
PID_PATH.write_text(f"{process.pid}\n")
(ROOT / "outputs" / "model-launch.json").write_text(json.dumps({
    "pid": process.pid,
    "started_at_unix": time.time(),
    "base_url": "http://127.0.0.1:19888/v1",
    "served_model": "Qwen3.8-27B",
    "weights": "Qwen/Qwen3.8-27B-FP8",
    "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
    "max_num_seqs": int(os.environ.get("MAR_QWEN_MAX_SEQS", "8")),
    "gpu_memory_utilization": float(os.environ.get("MAR_QWEN_GPU_MEMORY", "0.78")),
    "max_model_len": int(os.environ.get("MAR_QWEN_CONTEXT", "16384")),
    "cuda_visible_devices": os.environ.get("MAR_QWEN_GPU", "0"),
    "log": str(LOG_PATH),
}, ensure_ascii=False, indent=2) + "\n")
print(f"已启动模型进程 PID {process.pid}，日志 {LOG_PATH}")
