"""预缓存作者测试集的 SearXNG 查询；images 是文本搜图候选，不是原生反向搜图。"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from retrieval.cache import EvidenceCache
from retrieval.reverse import claim_query
from retrieval.search import SearchClient

DATASETS = ("DGM4-author1000", "MMFakeBench-author1000")
READY = {"ok", "no_results"}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def build_tasks(data_root: Path, datasets, categories) -> tuple[list[dict], int]:
    """Only the original claim and sample identifiers leave the input loader."""
    tasks = {}
    sample_total = 0
    for name in datasets:
        rows = json.loads((data_root / name / "samples.json").read_text())
        if not isinstance(rows, list):
            raise ValueError(f"{name}: samples.json must contain a list")
        sample_total += len(rows)
        for row in rows:
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"{name}: sample text must be a non-empty string")
            query = claim_query(row["text"])
            reference = {"dataset": name, "sample_id": str(row["sample_id"])}
            for category in categories:
                task = tasks.setdefault(
                    (category, query),
                    {"query": query, "category": category, "samples": []},
                )
                task["samples"].append(reference)
    return list(tasks.values()), sample_total


def load_checkpoint(path: Path) -> dict:
    """Recover an incomplete final append, but report corruption in older lines."""
    latest = {}
    if not path.exists():
        return latest
    data = path.read_bytes()
    offset = 0
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise ValueError(f"Invalid checkpoint JSON at line {index + 1}") from None
            with path.open("r+b") as file:
                file.truncate(offset)
            break
        key = record.get("key_hash")
        if not isinstance(key, str) or record.get("kind") != "search_result":
            raise ValueError(f"Unexpected checkpoint record at line {index + 1}")
        # An offline miss must not erase a previously failed online attempt:
        # retrying that backend still requires --retry-errors.
        if record["status"] != "cache_miss" or key not in latest:
            latest[key] = record
        offset += len(line)
    else:
        if data and not data.endswith(b"\n"):
            with path.open("ab") as file:
                file.write(b"\n")
    return latest


def ready_cached(client: SearchClient, key: dict) -> dict | None:
    value = client.cache.get(client.namespace, key)
    if (value is not None and value.get("status") in READY
            and isinstance(value.get("rows"), list)):
        return value
    return None


def warm_tasks(tasks: list[dict], client: SearchClient, checkpoint: Path, *,
               workers=2, max_new=None, retry_errors=False, progress=None) -> dict:
    """Warm a bounded stream, reuse successful queries, and checkpoint each result.

    ``max_new`` limits uncached queries scheduled in this invocation. Cache hits
    do not consume it. ``attempted`` and outcome counters describe this run;
    ``coverage`` describes all requested tasks, including earlier runs.
    """
    if not 1 <= workers <= 4:
        raise ValueError("workers must be 1..4")
    if max_new is not None and max_new < 0:
        raise ValueError("max_new must be non-negative")
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.with_suffix(checkpoint.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _warm_locked(tasks, client, checkpoint, workers=workers,
                            max_new=max_new, retry_errors=retry_errors,
                            progress=progress)


def _warm_locked(tasks, client, checkpoint, *, workers, max_new, retry_errors, progress):
    latest = load_checkpoint(checkpoint)
    run_id = uuid.uuid4().hex
    started_at = timestamp()
    counts = Counter(attempted=0, success=0, empty=0, errors=0, cache_misses=0,
                     cachehits=0, network_attempts=0, skipped_errors=0,
                     new_scheduled=0, partial_success=0)
    prepared = []
    for task in tasks:
        key = client.cache_key(task["query"], task["category"])
        prepared.append((task, key, client.cache.key_hash(key)))
    consecutive_errors = 0
    stop_reason = None
    deferred = []
    run_outcomes = {}

    def execute(task, key, key_hash):
        try:
            rows, events = client.search(task["query"], category=task["category"])
            last = events[-1]
            status = last["status"]
            if status not in READY | {"error", "cache_miss"}:
                raise ValueError("Unexpected search event status")
            cache_hit = last.get("cache") == "hit"
            network_attempted = client.mode == "online" and not cache_hit
            return {
                "schema_version": 1, "kind": "search_result",
                "run_id": run_id, "finished_at": timestamp(), "key_hash": key_hash,
                "key": key, "query": task["query"], "category": task["category"],
                "samples": task["samples"], "mode": client.mode,
                "status": status, "count": len(rows), "cache_hit": cache_hit,
                "network_attempted": network_attempted,
                "partial": bool(last.get("partial")), "events": events,
            }
        except Exception as exc:
            return {
                "schema_version": 1, "kind": "search_result",
                "run_id": run_id, "finished_at": timestamp(), "key_hash": key_hash,
                "key": key, "query": task["query"], "category": task["category"],
                "samples": task["samples"], "mode": client.mode,
                "status": "error", "count": 0, "cache_hit": False,
                "network_attempted": False, "partial": False,
                "events": [{"status": "error", "stage": "warmup",
                            "error_type": type(exc).__name__}],
            }

    with checkpoint.open("a", encoding="utf-8") as output:
        def record_result(record):
            nonlocal consecutive_errors, stop_reason
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
            key_hash = record["key_hash"]
            if record["status"] != "cache_miss" or key_hash not in latest:
                latest[key_hash] = record
            run_outcomes[key_hash] = record
            counts["attempted"] += 1
            counts[{"ok": "success", "no_results": "empty", "error": "errors",
                    "cache_miss": "cache_misses"}[record["status"]]] += 1
            counts["cachehits"] += int(record["cache_hit"])
            counts["partial_success"] += int(record["status"] == "ok" and record["partial"])
            if record["network_attempted"]:
                counts["network_attempts"] += 1
                consecutive_errors = consecutive_errors + 1 if record["status"] == "error" else 0
                if consecutive_errors >= 10:
                    stop_reason = "ten_consecutive_backend_errors"
            if progress is not None:
                progress(record, dict(counts))

        # Existing success data is authoritative even when a stale checkpoint
        # contains an earlier failure, and does not consume the new-query budget.
        for task, key, key_hash in prepared:
            if ready_cached(client, key) is not None:
                record_result(execute(task, key, key_hash))
            elif (client.mode == "online" and not retry_errors
                  and latest.get(key_hash, {}).get("status") == "error"):
                counts["skipped_errors"] += 1
            else:
                deferred.append((task, key, key_hash))

        pending_iter = iter(deferred)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            exhausted = False
            while futures or not exhausted:
                while not exhausted and not stop_reason and len(futures) < workers:
                    if max_new is not None and counts["new_scheduled"] >= max_new:
                        exhausted = True
                        break
                    item = next(pending_iter, None)
                    if item is None:
                        exhausted = True
                        break
                    futures[pool.submit(execute, *item)] = item
                    counts["new_scheduled"] += 1
                if not futures:
                    break
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in finished:
                    futures.pop(future)
                    record_result(future.result())
                if stop_reason:
                    # The bounded queue has at most workers-1 already-running
                    # requests; allow these to finish and persist their outcomes.
                    exhausted = True

    coverage = Counter(success=0, empty=0, errors=0, cache_misses=0, pending=0,
                       partial_success=0)
    for task, key, key_hash in prepared:
        cached = ready_cached(client, key)
        if cached is not None:
            coverage["success" if cached["status"] == "ok" else "empty"] += 1
            coverage["partial_success"] += int(cached["status"] == "ok" and bool(cached.get("partial")))
            continue
        previous = run_outcomes.get(key_hash) or latest.get(key_hash, {})
        status = previous.get("status")
        coverage["errors" if status == "error"
                 else "cache_misses" if status == "cache_miss" else "pending"] += 1
    return {
        "schema_version": 1, "run_id": run_id, "started_at": started_at,
        "finished_at": timestamp(), "mode": client.mode,
        "provider": client.provider, "base_url": client.base_url,
        "max_results": client.max_results, "language": client.language,
        "search_kind": "text_queries; images are not native reverse-image lookup",
        "total": len(tasks), **dict(counts),
        "coverage": {**dict(coverage),
                     "complete": coverage["success"] + coverage["empty"] == len(tasks)},
        "stopped": bool(stop_reason), "stop_reason": stop_reason,
        "consecutive_backend_errors": consecutive_errors,
        "max_inflight_at_stop": workers - 1,
        "checkpoint": str(checkpoint),
    }


def write_summary(path: Path, summary: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="预缓存作者样本的 SearXNG 网页或文本搜图结果，不下载候选图片。")
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--categories", default="images",
                        help="images、general 或 general,images；默认 images")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-new", type=int,
                        help="本次未缓存查询上限，缓存命中不占上限；0 仅核对已有缓存")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--retry-errors", action="store_true",
                        help="重试已记录的失败；否则继续处理未运行的查询")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/search-warmup.jsonl")
    args = parser.parse_args()
    categories = list(dict.fromkeys(part.strip() for part in args.categories.split(",")))
    if not categories or any(c not in {"general", "images"} for c in categories):
        parser.error("categories 仅支持 general 和 images")
    if not 1 <= args.workers <= 4 or (args.max_new is not None and args.max_new < 0):
        parser.error("workers 必须为 1..4，max-new 必须非负")
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    tasks, sample_total = build_tasks(args.data_root, datasets, categories)
    cache_path = Path(os.getenv("MAR_CACHE_DB", str(ROOT / "cache/evidence.sqlite3")))
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    with EvidenceCache(cache_path) as cache:
        client = SearchClient(
            cache=cache, base_url=os.getenv("MAR_SEARXNG_URL", "http://127.0.0.1:18888"),
            mode=args.mode, timeout=float(os.getenv("MAR_SEARCH_TIMEOUT", "20")),
            max_results=10,
        )

        def progress(record, counts):
            print(json.dumps({"completed": counts["attempted"], "total": len(tasks),
                              "category": record["category"], "status": record["status"],
                              "cache_hit": record["cache_hit"],
                              "count": record["count"]}, ensure_ascii=False), flush=True)

        summary = warm_tasks(
            tasks, client, args.output, workers=args.workers,
            max_new=args.max_new, retry_errors=args.retry_errors, progress=progress,
        )
        summary.update(datasets=list(datasets), categories=categories,
                       sample_total=sample_total,
                       counter_note="attempted/success/empty/errors/cachehits 属于本轮；coverage 覆盖全部去重查询")
        write_summary(args.output.with_suffix(".summary.json"), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["coverage"]["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
