"""Small command-line interface to cache-backed SearXNG search."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from retrieval.cache import EvidenceCache
from retrieval.search import SearchClient

load_dotenv(ROOT / ".env")
parser = argparse.ArgumentParser(description="查询本地检索缓存或 SearXNG")
parser.add_argument("query")
parser.add_argument("--category", choices=["general", "images"], default="general")
parser.add_argument("--mode", choices=["online", "offline"], default=os.getenv("MAR_CACHE_MODE", "online"))
args = parser.parse_args()
client = SearchClient(cache=EvidenceCache(os.getenv("MAR_CACHE_DB", str(ROOT / "cache/evidence.sqlite3"))),
                      base_url=os.getenv("MAR_SEARXNG_URL", "http://127.0.0.1:18888"),
                      mode=args.mode)
rows, events = client.search(args.query, category=args.category)
print(json.dumps({"results": rows, "events": events}, ensure_ascii=False, indent=2))
sys.exit(1 if not rows and any(e.get("status") in {"error", "cache_miss"} for e in events) else 0)
