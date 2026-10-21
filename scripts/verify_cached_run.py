"""只读重放已记录的网页查询，验证离线缓存，不调用搜索或模型服务。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.cache import EvidenceCache
from retrieval.search import SearchClient

READY = {"ok", "no_results"}


class ReadOnlyCache:
    """The SearchClient cache protocol without schema changes or writes."""

    key_hash = staticmethod(EvidenceCache.key_hash)

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError("The existing cache database was not found")
        self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self.connection.execute("PRAGMA query_only = ON")

    def get(self, namespace, key):
        row = self.connection.execute(
            "SELECT value_json FROM evidence_cache WHERE namespace=? AND key_hash=?",
            (namespace, self.key_hash(key)),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def recorded_key(self, key_hash):
        if not key_hash:
            return None
        row = self.connection.execute(
            "SELECT key_json FROM evidence_cache WHERE namespace='search' AND key_hash=?",
            (key_hash,),
        ).fetchone()
        if row is None:
            return None
        key = json.loads(row[0])
        if self.key_hash(key) != key_hash:
            raise ValueError("Stored search key does not match its hash")
        return key

    def put(self, *args, **kwargs):
        raise AssertionError("Offline verification must not write the cache")

    def close(self):
        self.connection.close()


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _queries(record):
    grouped, skipped = {}, Counter()
    for event in record.get("retrieval_diagnostics", []):
        if event.get("stage") != "search":
            continue
        if event.get("category", "general") != "general":
            skipped["non_general_category"] += 1
            continue
        if event.get("provider", "searxng") != "searxng":
            skipped["non_searxng_provider"] += 1
            continue
        query = event.get("query")
        if not isinstance(query, str) or not query.strip():
            skipped["missing_query"] += 1
            continue
        identity = (query.strip(), event.get("cache_key"))
        item = grouped.setdefault(identity, {
            "query": query.strip(), "category": "general",
            "recorded_cache_key": event.get("cache_key"), "original_event_statuses": [],
        })
        item["original_event_statuses"].append(event.get("status", "unrecorded"))
    return list(grouped.values()), dict(skipped)


def _verify_query(item, config, cache):
    stored = cache.recorded_key(item["recorded_cache_key"])
    configured = config.get("search_config") or {}
    if not isinstance(configured, dict):
        raise ValueError("config.search_config must be an object")
    parameters = {**configured, **(stored or {})}
    base_url = parameters.get("base_url", config.get(
        "searxng_url", config.get("search_base_url", "http://127.0.0.1:18888")))
    max_results = parameters.get("max_results", config.get("search_max_results", 10))
    client = SearchClient(
        cache=cache, base_url=base_url, mode="offline",
        timeout=config.get("search_timeout", 20), max_results=max_results)
    client.language = parameters.get("language", client.language)
    client.normalizer_version = parameters.get("normalizer_version", client.normalizer_version)
    key = client.cache_key(item["query"], "general")
    for name in ("safesearch", "pageno"):
        if name in parameters:
            key[name] = parameters[name]
    # Reproduce the frozen request parameters for this offline lookup, including
    # fields that the current SearchClient otherwise supplies as fixed defaults.
    client.cache_key = lambda query, category="general": {
        **key, "query": query.strip(), "category": category}
    key_hash = cache.key_hash(key)
    original_success = any(status in READY for status in item["original_event_statuses"])
    result = {
        **item, "original_query_succeeded": original_success,
        "parameter_source": ("recorded_cache_key_json" if stored else
                             "run_search_config" if configured else "run_config_with_client_defaults"),
        "parameters": key, "computed_cache_key": key_hash, "mode": "offline",
    }
    if (parameters.get("category", "general") != "general"
            or (item["recorded_cache_key"] and item["recorded_cache_key"] != key_hash)
            or (stored is not None and stored != key)):
        return {**result, "status": "parameter_mismatch", "cache": "not_checked",
                "result_count": 0, "verified_reusable": False}
    rows, events = client.search(item["query"], category="general")
    final = events[-1]
    hit = final.get("cache") == "hit" and final.get("status") in READY
    return {
        **result, "status": final["status"], "cache": final.get("cache"),
        "result_count": len(rows), "partial": bool(final.get("partial")),
        # A newly present cache entry never rewrites a historical failed query
        # into a successful original query or a completed model evaluation.
        "verified_reusable": bool(original_success and hit),
        "retrieved_at": final.get("retrieved_at"),
    }


def _summarize(records):
    rows = [query for record in records for query in record["queries"]]
    successful = [row for row in rows if row["original_query_succeeded"]]
    return {
        "records": len(records),
        "sample_ids": [record["sample_id"] for record in records],
        "unique_queries_per_record": len(rows),
        "original_successful_queries": len(successful),
        "original_failed_or_unrecorded_queries": len(rows) - len(successful),
        "cache_hits": sum(row.get("cache") == "hit" for row in rows),
        "cache_misses": sum(row["status"] == "cache_miss" for row in rows),
        "parameter_mismatches": sum(row["status"] == "parameter_mismatch" for row in rows),
        "verified_successful_query_reuse": sum(row["verified_reusable"] for row in rows),
        "all_original_successful_queries_reusable": (
            all(row["verified_reusable"] for row in successful) if successful else None),
        "all_recorded_queries_reusable": (
            all(row["verified_reusable"] for row in rows) if rows else None),
    }


def verify_run(run_directory, cache_db):
    run_directory = Path(run_directory).expanduser().resolve()
    config_path = run_directory / "config.json"
    config = json.loads(config_path.read_text())
    network_attempts = 0
    verified = []
    cache = ReadOnlyCache(cache_db)

    def forbidden_request(*args, **kwargs):
        nonlocal network_attempts
        network_attempts += 1
        raise AssertionError("Network requests are forbidden during offline verification")

    try:
        with patch.object(SearchClient, "_request", forbidden_request):
            for path in sorted((run_directory / "records").glob("*.json")):
                record = json.loads(path.read_text())
                queries, skipped = _queries(record)
                verified.append({
                    "sample_id": record["sample_id"], "record_status": record.get("status", "unrecorded"),
                    "record_sha256": _sha256(path), "skipped_search_events": skipped,
                    "queries": [_verify_query(item, config, cache) for item in queries],
                })
    finally:
        cache.close()
    completed = [record for record in verified if record["record_status"] == "ok"]
    failed = [record for record in verified if record["record_status"] != "ok"]
    completed_summary, failed_summary = _summarize(completed), _summarize(failed)
    return {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_directory), "config_sha256": _sha256(config_path),
        "cache_database": str(Path(cache_db).expanduser().resolve()),
        "cache_open_mode": "read_only", "mode": "offline",
        "network_attempts": network_attempts,
        "network_guard": "SearchClient._request replaced with a counting exception",
        "completed_records": completed_summary, "failed_records": failed_summary,
        "completed_run_queries_verified": (
            network_attempts == 0
            and completed_summary["all_recorded_queries_reusable"] is True),
        "scope": "Only persisted general search queries are replayed; no model inference is rerun.",
        "records": verified,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="只读验证实际评估查询能否离线复用")
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--cache-db", type=Path, default=ROOT / "cache/evidence.sqlite3")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    run_directory = args.run_directory.resolve()
    output = (args.output or ROOT / "outputs" / f"{run_directory.name}-cache-verification.json").resolve()
    if output == run_directory or run_directory in output.parents:
        parser.error("验证报告必须位于原评估目录之外")
    report = verify_run(run_directory, args.cache_db)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".cache-verification-", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({
        "report": str(output), "network_attempts": report["network_attempts"],
        "completed_records": report["completed_records"],
        "failed_records": report["failed_records"],
        "completed_run_queries_verified": report["completed_run_queries_verified"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["completed_run_queries_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
