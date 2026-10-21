"""Build resumable evidence for both author test sets without mutating their data."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv
from evidence_manifest import atomic_json, summarize, READY_STATUSES, text_digest
from retrieval.cache import EvidenceCache
from retrieval.search import SearchClient
from retrieval.reverse import CachedReverseSearch, claim_query


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial_record(sample):
    caption = sample.get("captions", "")
    return {"sample_id": sample["sample_id"], "image_sha256": sample["image_sha256"],
            "captions": caption, "captions_sha256": text_digest(caption),
            "status": "author_annotation" if caption.strip() else "pending",
            "source": "author_annotation" if caption.strip() else "none",
            "entries": [], "events": []}


def valid_record(record, sample):
    caption = record.get("captions", "")
    return (record.get("image_sha256") == sample["image_sha256"]
            and isinstance(caption, str)
            and record.get("captions_sha256") == text_digest(caption)
            and record.get("status") in READY_STATUSES
            and bool(caption.strip()) == (record["status"] in {"ok", "author_annotation"})
            and (record["status"] != "author_annotation" or caption == sample.get("captions", "")))


def sample_identity(sample):
    """Keep input text changes separate without including evaluation labels."""
    return {"version": 1, "sample_id": sample["sample_id"],
            "image_sha256": sample["image_sha256"],
            "text_sha256": text_digest(sample["text"]),
            "author_captions_sha256": text_digest(sample.get("captions", ""))}


def generation_specs(sample, reverse, backend):
    query = claim_query(sample["text"])
    backends = ["searxng_images", "serper_lens"] if backend == "hybrid" else [backend]
    return {
        name: {"version": 1, "sample_input": sample_identity(sample),
               "query": query, "backend": name,
               "retrieval": reverse.key(sample["image_path"], name, query)}
        for name in backends
    }


def generation_cache_key(spec):
    return {"version": 2, "sample_input": spec["sample_input"],
            "generation_fingerprint": EvidenceCache.key_hash(spec)}


def valid_generated(record, sample, specs):
    if not valid_record(record, sample) or record.get("evidence_origin") != "generated":
        return False
    expected = specs.get(record.get("source"))
    return (expected is not None and record.get("generation_spec") == expected
            and record.get("generation_fingerprint") == EvidenceCache.key_hash(expected))


def valid_frozen(record, sample):
    provenance = record.get("import_provenance", {})
    return (valid_record(record, sample)
            and record.get("evidence_origin") == "frozen_import"
            and record.get("sample_input") == sample_identity(sample)
            and isinstance(provenance, dict)
            and all(isinstance(provenance.get(key), str) and provenance[key]
                    for key in ("path", "manifest_sha256", "input_sha256", "record_sha256")))


def freeze_import(record, sample, provenance):
    """An explicitly verified source-project record remains a frozen baseline."""
    return {**record, "evidence_origin": "frozen_import",
            "sample_input": sample_identity(sample),
            "imported_from": provenance["path"],
            "import_provenance": {**provenance,
                                  "record_sha256": EvidenceCache.key_hash(record)}}


def select_record(sample, *, cache, specs, previous=None, original=None,
                  provenance=None):
    """Prefer exact author annotations, then valid generated or frozen evidence."""
    initial = initial_record(sample)
    if initial["status"] == "author_annotation":
        return {**initial, "evidence_origin": "author_annotation"}
    previous = previous or {}
    frozen = cache.get("frozen_sample_evidence", sample_identity(sample)) or {}
    if not valid_frozen(frozen, sample):
        frozen = {}
    if original and provenance and valid_record(original, sample):
        frozen = freeze_import(original, sample, provenance)
        cache.put("frozen_sample_evidence", sample_identity(sample), frozen)
    elif valid_frozen(previous, sample):
        frozen = previous
    # A currently generated record wins over an older frozen empty result.
    if valid_generated(previous, sample, specs):
        return previous
    candidates = []
    for spec in specs.values():
        cached = cache.get("sample_evidence", generation_cache_key(spec))
        if cached is not None and valid_generated(cached, sample, specs):
            candidates.append(cached)
    if candidates:
        return next((r for r in candidates if r.get("captions", "").strip()), candidates[-1])
    # Legacy imports may have only imported_from. Accept them as frozen only
    # when their original record has just been independently verified.
    same_original = (original and valid_record(original, sample)
                     and all(previous.get(k) == v for k, v in original.items()))
    if valid_record(previous, sample) and not valid_frozen(previous, sample) and not same_original:
        return {**initial, "reason": "generation_configuration_changed_or_missing"}
    if frozen:
        return frozen
    return initial


def needs_search(record, *, backend, mode, retry_empty=False, refresh=False):
    if record["status"] not in READY_STATUSES:
        return True
    if record["status"] == "author_annotation":
        return False
    if refresh and record.get("evidence_origin") == "generated":
        return True
    if record["status"] in {"no_results", "no_caption"}:
        return retry_empty or (record.get("source") == "searxng_images"
                               and backend in {"hybrid", "serper_lens"} and mode == "offline")
    return False


def mark_generated(record, sample, specs):
    source = record.get("source")
    if source not in specs:
        return record
    # Do not inherit the provenance of an empty imported record we replaced.
    for field in ("imported_from", "import_provenance", "sample_input"):
        record.pop(field, None)
    record.update(evidence_origin="generated", generation_spec=specs[source],
                  generation_fingerprint=EvidenceCache.key_hash(specs[source]))
    return record


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="导入旧证据并补齐持久缓存")
    parser.add_argument("--dataset", choices=["DGM4-author1000", "MMFakeBench-author1000", "all"], default="all")
    parser.add_argument("--mode", choices=["import", "online", "offline"], default="import")
    parser.add_argument("--backend", choices=["hybrid", "searxng_images", "serper_lens"], default="hybrid")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--allow-paid", action="store_true", help="允许首次使用已配置的 Serper/ImgBB")
    parser.add_argument("--warm-web", action="store_true", help="另外缓存输入新闻文本的网页搜索")
    parser.add_argument("--retry-empty", action="store_true", help="重新处理空证据，便于切换检索后端；仍优先使用原始搜索缓存")
    parser.add_argument("--refresh", action="store_true",
                        help="重新提取本项目生成的证据，复用原始候选和下载缓存；保留作者图注及冻结导入")
    parser.add_argument("--source-root", type=Path, default=Path("/home/c/czh/MAR_KDD"))
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8 or (args.max_new is not None and args.max_new < 1):
        parser.error("workers must be 1..8 and max-new must be positive")
    if args.refresh and args.mode != "online":
        parser.error("--refresh 仅适用于 --mode online；离线模式不会重新抓取或提取")
    if args.max_results < 1:
        parser.error("max-results must be positive")
    (ROOT / "cache").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    cache_path = Path(os.getenv("MAR_CACHE_DB", str(ROOT / "cache/evidence.sqlite3")))
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    with (ROOT / "cache/prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with EvidenceCache(cache_path) as cache:
            return prepare_datasets(args, cache)


def prepare_datasets(args, cache):
    names = ["DGM4-author1000", "MMFakeBench-author1000"] if args.dataset == "all" else [args.dataset]
    search = SearchClient(cache=cache, base_url=os.getenv("MAR_SEARXNG_URL", "http://127.0.0.1:18888"),
                          mode="offline" if args.mode == "offline" else "online",
                          timeout=float(os.getenv("MAR_SEARCH_TIMEOUT", "20")),
                          max_results=args.max_results)
    first_key = lambda name: next((s.strip() for s in os.getenv(name, "").split(",") if s.strip()), "")
    reverse = CachedReverseSearch(cache, mode="offline" if args.mode == "offline" else "online",
                                  search_client=search, max_results=args.max_results,
                                  serper_key=first_key("MAR_SERPER_API_KEYS") if args.allow_paid else "",
                                  imgbb_key=first_key("MAR_IMGBB_API_KEYS") if args.allow_paid else "")
    manifests, inputs, paths, specs_by_sample = {}, {}, {}, {}
    imported = 0
    for name in names:
        sample_path = ROOT / "data" / name / "samples.json"
        samples = json.loads(sample_path.read_text())
        if len({s["sample_id"] for s in samples}) != len(samples):
            raise ValueError("Duplicate sample IDs")
        inputs[name] = samples
        path = sample_path.with_name("reverse-evidence.json")
        paths[name] = path
        old_path = args.source_root / "dataset" / name / "reverse-evidence.json"
        donors = {}
        source_provenance = None
        input_sha256 = digest(sample_path)
        for donor in [old_path, path]:
            if donor.is_file():
                value = json.loads(donor.read_text())
                if value.get("input_sha256") != input_sha256:
                    if donor == path:
                        raise ValueError("Existing evidence refers to another sample file")
                    continue
                donors[donor] = value.get("records", {})
                if donor == old_path:
                    source_provenance = {"path": str(old_path),
                                         "manifest_sha256": digest(old_path),
                                         "input_sha256": input_sha256}
        records = {}
        for sample in samples:
            if digest(sample["image_path"]) != sample["image_sha256"]:
                raise ValueError("Image changed: " + sample["sample_id"])
            specs = generation_specs(sample, reverse, args.backend)
            specs_by_sample[(name, sample["sample_id"])] = specs
            record = select_record(
                sample, cache=cache, specs=specs,
                previous=donors.get(path, {}).get(sample["sample_id"]),
                original=donors.get(old_path, {}).get(sample["sample_id"]),
                provenance=source_provenance,
            )
            if record["status"] in READY_STATUSES:
                imported += 1
                if record.get("evidence_origin") in {"author_annotation", "frozen_import"}:
                    cache.put("imported_evidence", {**sample_identity(sample),
                              "captions_sha256": record["captions_sha256"]}, record)
                elif valid_generated(record, sample, specs):
                    cache.put("sample_evidence", generation_cache_key(record["generation_spec"]), record)
            records[sample["sample_id"]] = record
        manifests[name] = {"schema_version": 1, "input_sha256": digest(sample_path),
                           "records": records,
                           "preparation": {"implementation": "cached-evidence-v2",
                                           "searxng_images_is_native_reverse": False}}

    def save(name):
        manifest = manifests[name]
        manifest["summary"] = summarize(manifest["records"])
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(paths[name], manifest)

    def should_search(record):
        return needs_search(record, backend=args.backend, mode=args.mode,
                            retry_empty=args.retry_empty, refresh=args.refresh) or (
            record["status"] in {"no_results", "no_caption"}
            and record.get("source") == "searxng_images"
            and args.backend in {"hybrid", "serper_lens"} and args.allow_paid)

    for name in names:
        save(name)
    pending = [(name, sample) for name in names for sample in inputs[name]
               if should_search(manifests[name]["records"][sample["sample_id"]])
               or args.warm_web]
    if args.max_new:
        pending = pending[:args.max_new]

    def worker(item):
        name, sample = item
        record = dict(manifests[name]["records"][sample["sample_id"]])
        query = claim_query(sample["text"])
        if args.warm_web:
            rows, events = search.search(query, category="general")
            record["web_seed"] = {"query": query, "results": len(rows), "events": events}
        if should_search(record):
            results = []
            backends = (["searxng_images", "serper_lens"] if args.backend == "hybrid"
                        else [args.backend])
            for backend in backends:
                if backend == "serper_lens" and not args.allow_paid and args.mode != "offline":
                    raw = cache.get("image_matches", reverse.raw_key(sample["image_path"], backend, query))
                    final = (None if args.refresh else cache.get(
                        "image_evidence", reverse.key(sample["image_path"], backend, query)))
                    if raw is None and (final is None or final.get("status") not in READY_STATUSES):
                        continue
                result = reverse.search(sample["image_path"], backend=backend, query=query,
                                        refresh=args.refresh)
                results.append(result)
                if result.get("captions", "").strip():
                    break
            result = next((r for r in results if r.get("captions", "").strip()), None)
            if result is None:
                # Any retrieval failure remains visible instead of being called a successful empty search.
                result = next((r for r in reversed(results) if r.get("status") in {"error", "cache_miss"}),
                              results[-1] if results else {"status": "not_configured", "captions": "",
                                                          "source": "none", "events": []})
            record.update(result)
            record["backend_attempts"] = [{"source": r.get("source"), "status": r["status"],
                                          "cache": r.get("cache")} for r in results]
            record["captions_sha256"] = text_digest(record.get("captions", ""))
            record["checked_at"] = datetime.now(timezone.utc).isoformat()
            mark_generated(record, sample, specs_by_sample[(name, sample["sample_id"])])
        return name, sample["sample_id"], record

    if args.mode != "import":
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(worker, item): item for item in pending}
            for n, future in enumerate(as_completed(futures), 1):
                name, sample = futures[future]
                try:
                    name, sid, record = future.result()
                except Exception as exc:
                    sid = sample["sample_id"]
                    record = dict(manifests[name]["records"][sid])
                    record.update(status="error", error_type=type(exc).__name__)
                manifests[name]["records"][sid] = record
                if valid_generated(record, sample, specs_by_sample[(name, sid)]):
                    cache.put("sample_evidence", generation_cache_key(record["generation_spec"]), record)
                save(name)
                print(f"{n}/{len(pending)} {sid} {record['status']} cache={record.get('cache', '-')}", flush=True)
    report = {"datasets": {name: manifests[name]["summary"] for name in names},
              "imported_or_already_ready": imported, "mode": args.mode,
              "refresh_generated": args.refresh,
              "paid_reverse_attempts": reverse.network_calls,
              "paid_count_note": "上传与搜图组合尝试数；不等于供应商实际计费请求数",
              "paid_backend_disabled": reverse.paid_disabled,
              "cache": cache.stats()}
    atomic_json(ROOT / "outputs/cache-preparation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(m["summary"]["ready"] == len(inputs[n]) for n, m in manifests.items()) else 2


if __name__ == "__main__":
    sys.exit(main())
