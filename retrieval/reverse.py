"""Cache native reverse lookup separately from text-to-image discovery."""
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import threading
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reverse_image"))
from serper_lens import SerperLensClient, SerperLensResult, LensError

READY = {"ok", "no_results", "no_caption"}


def now():
    return datetime.now(timezone.utc).isoformat()


def image_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def claim_query(text):
    """Use only the input claim, never its label or a reference answer."""
    return " ".join(text.split())[:240]


class CachedReverseSearch:
    def __init__(self, cache, *, mode="online", search_client=None,
                 serper_key="", imgbb_key="", max_results=10,
                 similarity_cutoff=10, hashing_cutoff=5):
        if mode not in {"online", "offline"}:
            raise ValueError("mode must be online or offline")
        self.cache, self.mode, self.search_client = cache, mode, search_client
        self.serper_key, self.imgbb_key = serper_key, imgbb_key
        self.max_results = max_results
        self.similarity_cutoff, self.hashing_cutoff = similarity_cutoff, hashing_cutoff
        self.paid_disabled = False
        self.network_calls = 0
        self._lock = threading.Lock()
        self._locks = {}

    def raw_key(self, path, backend, query=""):
        key = {"version": 2, "image_sha256": image_digest(path), "backend": backend,
               "query": query if backend == "searxng_images" else "",
               "max_results": self.max_results}
        if backend == "searxng_images" and self.search_client is not None:
            key["search_config"] = self.search_client.cache_key(query, "images")
        return key

    def key(self, path, backend, query=""):
        from retrieval.extract import (EXTRACTION_VERSION, MAX_CANDIDATES,
                                       MAX_IMAGE_BYTES, MAX_HTML_BYTES, MAX_IMAGE_PIXELS)
        return {**self.raw_key(path, backend, query),
                "extraction": {"version": EXTRACTION_VERSION, "max_candidates": MAX_CANDIDATES,
                               "max_image_bytes": MAX_IMAGE_BYTES, "max_html_bytes": MAX_HTML_BYTES,
                               "max_image_pixels": MAX_IMAGE_PIXELS, "max_seconds": 45,
                               "request_timeout": 8},
                "similarity_cutoff": self.similarity_cutoff,
                "hashing_cutoff": self.hashing_cutoff}

    def _key_lock(self, key):
        import json
        name = json.dumps(key, sort_keys=True)
        with self._lock:
            return self._locks.setdefault(name, threading.Lock())

    def _matches(self, path, backend, query):
        if backend == "searxng_images":
            if self.search_client is None:
                raise RuntimeError("SearXNG search client is not configured")
            rows, events = self.search_client.search(query, category="images")
            if not rows and any(e.get("status") in {"error", "cache_miss"} for e in events):
                return None, events
            return [SerperLensResult(title=r.get("title", ""),
                        source=urlparse(r["href"]).netloc, link=r["href"],
                        image_url=r["image_url"], thumbnail_url=r.get("thumbnail", ""))
                    for r in rows if r.get("image_url")], events
        if self.paid_disabled:
            return None, [{"status": "error", "stage": "serper_lens",
                           "reason": "paid_backend_disabled_after_auth_or_quota_error"}]
        if not self.serper_key or not self.imgbb_key:
            return None, [{"status": "error", "stage": "serper_lens",
                           "reason": "credentials_missing"}]
        try:
            with self._lock:
                self.network_calls += 1
            matches = asyncio.run(SerperLensClient(self.serper_key, self.max_results,
                                                   timeout=20, attempts=1).search(
                                                       Path(path), self.imgbb_key))
            return matches, [{"status": "ok", "stage": "serper_lens"}]
        except LensError as exc:
            message = str(exc)  # LensError deliberately strips keys and provider bodies.
            if "quota_exhausted" in message or any(f"HTTP {code}" in message for code in (401, 402, 403, 429)):
                self.paid_disabled = True
            return None, [{"status": "error", "stage": "reverse_lookup", "error": message}]

    def _extract(self, path, matches):
        from retrieval.extract import extract_matches_cached
        return extract_matches_cached(path, matches, cache=self.cache,
                                      blob_dir=ROOT / "cache/blobs",
                                      max_seconds=45, timeout=8,
                                      similarity_cutoff=self.similarity_cutoff)

    def search(self, path, *, backend="serper_lens", query="", refresh=False):
        if backend not in {"serper_lens", "searxng_images"}:
            raise ValueError("Unknown image search backend")
        if backend == "searxng_images" and not query.strip():
            raise ValueError("Text-to-image discovery requires a claim query")
        if refresh and self.mode == "offline":
            raise ValueError("Refreshing extracted evidence requires online mode")
        key = self.key(path, backend, query)
        raw_key = self.raw_key(path, backend, query)
        with self._key_lock(key):
            cached = self.cache.get("image_evidence", key)
            if not refresh and cached is not None and cached.get("status") in READY:
                return {**cached, "cache": "hit"}
            if self.mode == "offline":
                return {"status": "cache_miss", "captions": "", "source": backend,
                        "cache": "miss", "entries": [], "events": [],
                        "reason": "offline_mode_never_calls_search_or_upload"}
            raw = self.cache.get("image_matches", raw_key)
            lookup_events = []
            if raw is None:
                matches, lookup_events = self._matches(path, backend, query)
                if matches is None:
                    return {"status": "error", "captions": "", "source": backend,
                            "cache": "miss", "entries": [], "events": lookup_events}
                raw = {"status": "ok" if matches else "no_results",
                       "matches": [asdict(m) for m in matches],
                       "created_at": now(), "source": backend,
                       "events": lookup_events}
                self.cache.put("image_matches", raw_key, raw)
            matches = [SerperLensResult(**m) for m in raw["matches"]]
            if matches:
                result = self._extract(path, matches)
            else:
                result = {"status": "no_results", "captions": "", "matches": 0,
                          "entries": [], "events": []}
            result.update(source=backend, created_at=now(),
                          retrieval_kind=("native_reverse_image" if backend == "serper_lens"
                                          else "text_to_image_with_visual_verification"))
            result["events"] = list(raw.get("events", [])) + result.get("events", [])
            # Preserve raw matches even if individual websites cannot be fetched.
            # This makes retries free of further paid Lens calls.
            self.cache.put("image_evidence" if result["status"] in READY else "image_errors",
                           key, result)
            return {**result, "cache": "miss", "raw_cache": "hit" if not lookup_events else "miss"}
