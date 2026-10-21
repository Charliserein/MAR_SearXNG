"""Resumable MAR evaluation; labels are recorded but never passed to inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MAR_framework"))
from evidence_manifest import attach_evidence
from retrieval.provenance import evidence_kind
LABELS = ("original", "textual_veracity_distortion", "visual_veracity_distortion", "mismatch")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class CallRecorder(BaseCallbackHandler):
    def __init__(self, path):
        self.path, self.starts, self.events = path, {}, []
        self.raise_error = True

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.starts[str(run_id)] = time.monotonic()

    def append(self, record):
        self.events.append(record)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_llm_end(self, response, *, run_id, **kwargs):
        message = response.generations[0][0].message
        usage = getattr(message, "usage_metadata", None) or {}
        self.append({"timestamp": timestamp(), "status": "ok", "run_id": str(run_id),
                     "seconds": time.monotonic() - self.starts.pop(str(run_id), time.monotonic()),
                     "usage": usage, "response_metadata": message.response_metadata,
                     "content": message.content})

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.append({"timestamp": timestamp(), "status": "error", "run_id": str(run_id),
                     "seconds": time.monotonic() - self.starts.pop(str(run_id), time.monotonic()),
                     "error_type": type(error).__name__, "error": str(error)})


def run_sample(analyzer, sample):
    """Keep ground truth outside NewsAnalyzer's prompt and state."""
    started = time.monotonic()
    record = {"sample_id": sample["sample_id"], "news_id": sample["id"],
              "ground_truth": sample["ground_truth"], "source_class": sample["fake_cls"],
              "source_dataset": sample.get("source_dataset"),
              "text": sample["text"], "image_path": sample["image_path"],
              "image_sha256": sample["image_sha256"], "started_at": timestamp(),
              "reverse_image_status": sample.get("reverse_image_status", "missing"),
              "reverse_image_source": sample.get("reverse_image_source"),
              "image_evidence_kind": evidence_kind(sample),
              "reverse_image_used": bool(sample.get("captions", "").strip()),
              "reverse_image_captions_sha256": hashlib.sha256(sample.get("captions", "").encode()).hexdigest()}
    try:
        result = analyzer.analyze(sample["text"], sample["image_path"], sample.get("captions", ""),
                                  evidence_kind=evidence_kind(sample))
        if result.get("label") not in LABELS:
            raise ValueError("Analyzer returned an invalid label")
        record.update(status="ok", prediction=result["label"],
                      result={k: v for k, v in result.items() if k != "image"})
    except Exception as exc:
        record.update(status="error", prediction=None, error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
    record.update(finished_at=timestamp(), seconds=time.monotonic() - started)
    return record


def remaining_samples(samples, output, retry_errors=False):
    remaining = []
    for sample in samples:
        path = output / "records" / f"{sample['sample_id']}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("status") == "ok" or not retry_errors:
                continue
        remaining.append(sample)
    return remaining


def code_hashes():
    paths = [*sorted((ROOT / "MAR_framework").rglob("*.py")),
             *sorted((ROOT / "retrieval").rglob("*.py")), Path(__file__),
             ROOT / "scripts/evidence_manifest.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def next_attempt(output, sample_id):
    paths = [*(output / "attempts").glob(f"{sample_id}-*.json"),
             *(output / "calls").glob(f"{sample_id}-*.jsonl")]
    return max((int(p.stem.rsplit("-", 1)[1]) for p in paths), default=0) + 1


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/DGM4-author1000/samples.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--max-new", type=int, help="Stop after this many pending samples (pilot)")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--reverse-evidence", type=Path, help="Frozen evidence manifest (defaults to input's sibling reverse-evidence.json)")
    parser.add_argument("--strict-evidence", action=argparse.BooleanOptionalAction, default=True,
                        help="Require prepared reverse evidence, three rounds and a working web provider (default); disable only for diagnostics")
    parser.add_argument("--strict-web", action=argparse.BooleanOptionalAction, default=True,
                        help="Fail attempts whose requested web searches all fail, even when image evidence is optional")
    parser.add_argument("--preflight-only", action="store_true", help="Check configuration and web search without invoking the model")
    args = parser.parse_args()
    if args.rounds < 1 or args.attempts < 1 or (args.max_new is not None and args.max_new < 1):
        parser.error("rounds, attempts and max-new must be positive")
    samples = json.loads(args.input.read_text())
    if len({s['sample_id'] for s in samples}) != len(samples):
        parser.error("Duplicate sample IDs")
    for sample in samples:
        if sample['ground_truth'] not in LABELS or digest(sample['image_path']) != sample['image_sha256']:
            parser.error(f"Invalid sample or changed image: {sample['sample_id']}")
    evidence_path = args.reverse_evidence or args.input.with_name("reverse-evidence.json")
    evidence_manifest = json.loads(evidence_path.read_text()) if evidence_path.exists() else None
    if args.reverse_evidence and evidence_manifest is None:
        parser.error("Requested reverse evidence file does not exist")
    if args.strict_evidence and (evidence_manifest is None or args.rounds != 3):
        parser.error("Strict evaluation requires a reverse evidence manifest and exactly three rounds")
    if evidence_manifest is not None:
        try:
            samples = attach_evidence(samples, evidence_manifest, digest(args.input), args.strict_evidence)
        except ValueError as exc:
            parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for directory in ("records", "attempts", "calls"):
        (args.output / directory).mkdir(exist_ok=True)
    from web_search import WebSearchClient
    search_client = WebSearchClient()
    if args.strict_evidence or args.strict_web or args.preflight_only:
        preflight = search_client.preflight()
        atomic_json(args.output / "preflight.json", {"checked_at": timestamp(), "web_search": preflight,
                    "reverse_image": evidence_manifest.get("summary", {}) if evidence_manifest else {"status": "not_prepared"}})
        if preflight['status'] != 'ok':
            parser.error("Web search preflight failed; see preflight.json. No inference started")
        if args.preflight_only:
            print(json.dumps({"preflight": "ok", "provider": search_client.provider, "count": len(samples)}))
            return 0
    config = {"input_sha256": digest(args.input), "count": len(samples), "rounds": args.rounds,
              "model": os.getenv("MAR_MODEL", "Qwen3.8-27B"),
              "base_url": os.getenv("MAR_BASE_URL", "http://127.0.0.1:19888/v1"),
              "cache_mode": os.getenv("MAR_CACHE_MODE", "online"),
              "model_max_tokens": int(os.getenv("MAR_MODEL_MAX_TOKENS", "2048")),
              "model_extra_body": json.loads(os.getenv("MAR_MODEL_EXTRA_BODY", "{}")),
              "temperature": 0.0, "workers": args.workers, "code_sha256": code_hashes(),
              "search_backends": os.getenv("MAR_SEARCH_BACKENDS", "yahoo,wikipedia"),
              "search_provider": search_client.provider, "search_timeout": search_client.timeout,
              "search_config": {k: v for k, v in search_client.client.cache_key("__config__").items()
                                if k != "query"},
              "search_attempts": search_client.attempts, "search_proxy_configured": bool(search_client.proxy),
              "strict_evidence": args.strict_evidence,
              "strict_web": args.strict_web,
              "reverse_evidence_sha256": digest(evidence_path) if evidence_manifest else None,
              "reverse_image_caption_count": sum(bool(s.get('captions', '').strip()) for s in samples),
              "embedding_model": os.getenv("MAR_EMBEDDING_MODEL", "~/model_path/bge-base-en-v1.5/"),
              "relevance_threshold": 0.85, "max_pre_filtered": 10,
              "request_timeout_seconds": 300,
              "packages": {name: importlib.metadata.version(name) for name in
                           ("langchain-openai", "langgraph", "ddgs", "sentence-transformers", "torch", "httpx", "scikit-learn")},
              "reverse_image_evidence": any(s.get('captions', '').strip() for s in samples),
              "embedding_device": os.getenv("MAR_EMBEDDING_DEVICE", "cpu")}
    config_path = args.output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            parser.error("Configuration or code changed; use a new output directory")
    else:
        atomic_json(config_path, config)
        atomic_json(args.output / "samples.json", samples)
        atomic_json(args.output / "dataset-manifest.json", json.loads(args.input.with_name("manifest.json").read_text()))
        if evidence_manifest:
            atomic_json(args.output / "reverse-evidence.json", evidence_manifest)
        for relative in config['code_sha256']:
            target = args.output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        service_manifest = args.input.with_name("model-service.json")
        if service_manifest.exists():
            shutil.copyfile(service_manifest, args.output / "model-service.json")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    from NewsAnalyzer import NewsAnalyzer
    local = threading.local()
    progress_lock, active = threading.Lock(), {}

    def worker(sample):
        if not hasattr(local, "analyzer"):
            local.analyzer = NewsAnalyzer(base_url=config['base_url'], api_key=os.getenv("MAR_API_KEY", "sk-local"),
                                         model=config['model'], temperature=0.0, n_rounds=args.rounds)
            # Bounded network waits; model generation and prompts remain unchanged.
            local.analyzer.llm.request_timeout = 300
            local.analyzer.retriever.llm.request_timeout = 300
            local.analyzer.retriever.strict_evidence = args.strict_web or args.strict_evidence
        analyzer = local.analyzer
        sid = sample["sample_id"]
        first_attempt = next_attempt(args.output, sid)
        for attempt in range(first_attempt, first_attempt + args.attempts):
            recorder = CallRecorder(args.output / "calls" / f"{sid}-{attempt}.jsonl")
            analyzer.llm.callbacks = [recorder]
            analyzer.retriever.llm.callbacks = [recorder]
            analyzer.retriever.diagnostics = []
            with progress_lock:
                active[sid] = {"attempt": attempt, "started": time.monotonic(), "recorder": recorder}
            record = run_sample(analyzer, sample)
            record.update(attempt=attempt, retrieval_diagnostics=list(analyzer.retriever.diagnostics),
                          llm_calls=len(recorder.events),
                          llm_errors=sum(e['status'] == 'error' for e in recorder.events),
                          input_tokens=sum(e.get('usage', {}).get('input_tokens', 0) for e in recorder.events),
                          output_tokens=sum(e.get('usage', {}).get('output_tokens', 0) for e in recorder.events))
            atomic_json(args.output / "attempts" / f"{sid}-{attempt}.json", record)
            atomic_json(args.output / "records" / f"{sid}.json", record)
            if record["status"] == "ok":
                break
            print(f"RETRY {sid} attempt={attempt} {record['error_type']}: {record['error'][:250]}", flush=True)
        with progress_lock:
            active.pop(sid, None)
        return record

    pending = remaining_samples(samples, args.output, args.retry_errors)
    if args.max_new:
        pending = pending[:args.max_new]
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(15):
            with progress_lock:
                state = {sid: {"attempt": a['attempt'], "seconds": round(time.monotonic() - a['started']),
                               "llm_calls": len(a['recorder'].events)} for sid, a in active.items()}
            atomic_json(args.output / "progress.json", {"timestamp": timestamp(), "active": state,
                        "completed": len(list((args.output / "records").glob("*.json"))), "total": len(samples)})

    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    started = timestamp()
    print(f"START {started} pending={len(pending)} rounds={args.rounds} workers={args.workers}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(worker, sample) for sample in pending]
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                print(f"DONE {index}/{len(pending)} {record['sample_id']} {record['status']} prediction={record['prediction']} seconds={record['seconds']:.1f}", flush=True)
    finally:
        stop.set()
        monitor.join()
        with (args.output / "sessions.jsonl").open("a") as stream:
            stream.write(json.dumps({"started_at": started, "finished_at": timestamp(), "scheduled": len(pending)}) + "\n")
    records = [json.loads(p.read_text()) for p in sorted((args.output / "records").glob("*.json"))]
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    status = {"completed": len(records), "ok": sum(r['status'] == 'ok' for r in records), "total": len(samples), "finished_at": timestamp()}
    atomic_json(args.output / "status.json", status)
    atomic_json(args.output / "progress.json", {**status, "active": {}})
    print(json.dumps(status), flush=True)
    return int(any(r['status'] != 'ok' for r in records))


if __name__ == "__main__":
    sys.exit(main())
