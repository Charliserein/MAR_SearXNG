#!/usr/bin/env python3
"""真实图像请求测试；不向模型传入数据集标签。"""

import base64
import hashlib
import json
import mimetypes
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "http://127.0.0.1:19888/v1"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SAMPLES = Path("/home/c/czh/MAR_KDD/dataset/DGM4-author1000/samples.json")
sample = json.loads(SAMPLES.read_text())[0]
image_path = Path(sample["image_path"])
image_bytes = image_path.read_bytes()
mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
image_url = f"data:{mime_type};base64," + base64.b64encode(image_bytes).decode()
with OPENER.open(f"{BASE_URL}/models", timeout=10) as response:
    served_models = json.load(response)
assert any(item["id"] == "Qwen3.8-27B" for item in served_models["data"])
payload = {
    "model": "Qwen3.8-27B",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "Describe the visible image. Return only a JSON object with the keys main_subject and visible_details. Do not infer whether any associated news is true or false."},
        ],
    }],
    "max_tokens": 512,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
started = time.time()
request = urllib.request.Request(
    f"{BASE_URL}/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with OPENER.open(request, timeout=300) as response:
    result = json.load(response)
content = result["choices"][0]["message"]["content"]
try:
    structured = json.loads(content)
except json.JSONDecodeError:
    structured = None
record = {
    "base_url": BASE_URL,
    "served_model": "Qwen3.8-27B",
    "weights": "Qwen/Qwen3.8-27B-FP8",
    "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
    "image_path": str(image_path),
    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
    "labels_sent": False,
    "elapsed_seconds": round(time.time() - started, 3),
    "models_response": served_models,
    "completion_response": result,
    "parsed_json": structured,
    "passed": bool(content and structured and result["choices"][0]["finish_reason"] == "stop"),
}
output = ROOT / "outputs" / "model-vision-smoke.json"
output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"passed": record["passed"], "elapsed_seconds": record["elapsed_seconds"], "content": content}, ensure_ascii=False))
if not record["passed"]:
    raise SystemExit(1)
