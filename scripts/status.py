"""Print only operational state, never runtime credentials."""
import json
import os
from pathlib import Path
import sys
from collections import Counter
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from retrieval.cache import EvidenceCache

load_dotenv(ROOT / ".env")
report = {"project": str(ROOT), "model": os.getenv("MAR_MODEL", "Qwen3.8-27B"),
          "mode": os.getenv("MAR_CACHE_MODE", "online"), "datasets": {}}
for name in ["DGM4-author1000", "MMFakeBench-author1000"]:
    path = ROOT / "data" / name / "reverse-evidence.json"
    report["datasets"][name] = json.loads(path.read_text()).get("summary", {}) if path.exists() else {"status": "not_prepared"}
with httpx.Client(trust_env=False, timeout=5) as client:
    for name, url in [
        ("searxng", os.getenv("MAR_SEARXNG_URL", "http://127.0.0.1:18888") + "/healthz"),
        ("model_service", os.getenv("MAR_BASE_URL", "http://127.0.0.1:19888/v1") + "/models")]:
        try:
            response = client.get(url)
            report[name] = {"http_status": response.status_code,
                            "ready": response.status_code == 200}
            if name == "model_service" and response.status_code == 200:
                report[name]["models"] = [m["id"] for m in response.json().get("data", [])]
        except Exception as exc:
            report[name] = {"ready": False, "error_type": type(exc).__name__}
report["cache"] = EvidenceCache(os.getenv("MAR_CACHE_DB", str(ROOT / "cache/evidence.sqlite3"))).stats()
checkpoint = ROOT / "outputs/search-warmup.jsonl"
if checkpoint.exists():
    latest = {}
    for line in checkpoint.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # Another process may still be appending the final line.
        latest[row["key_hash"]] = row
    report["image_candidate_warmup"] = {
        "distinct_queries_attempted": len(latest),
        "statuses": dict(Counter(row["status"] for row in latest.values())),
        "kind": "text_to_image_candidates",
        "summary_file": str(ROOT / "outputs/search-warmup.summary.json"),
    }
print(json.dumps(report, ensure_ascii=False, indent=2))
