"""SearXNG JSON search with durable successful-response caching."""
from __future__ import annotations

from datetime import datetime, timezone
from html import unescape
import re
import time
from urllib.parse import urlparse

import httpx

from .cache import EvidenceCache


def web_url(value) -> bool:
    return (isinstance(value, str) and urlparse(value).scheme in ("http", "https")
            and bool(urlparse(value).netloc))


def _text(value) -> str:
    if not isinstance(value, str):
        return ""
    return unescape(re.sub(r"<[^>]*>", "", value)).strip()


def normalize_rows(raw: list, category: str, max_results: int) -> list[dict]:
    """Normalize snippets without fetching pages or downloading result images."""
    rows = []
    seen = set()
    for result in raw:
        if not isinstance(result, dict):
            continue
        href = result.get("url") or result.get("href")
        image_url = result.get("img_src") or result.get("image_url") or ""
        if category == "images" and not web_url(image_url):
            continue
        if not web_url(href):
            if category == "images" and web_url(image_url):
                href = image_url
            else:
                continue
        title = _text(result.get("title"))
        body = _text(result.get("content") or result.get("body")) or title
        if category == "general" and not body:
            continue
        identity = (href, image_url) if category == "images" else href
        if identity in seen:
            continue
        seen.add(identity)
        row = {"title": title, "href": href, "body": body}
        if category == "images":
            thumbnail = result.get("thumbnail_src") or result.get("thumbnail") or ""
            row.update(image_url=image_url if web_url(image_url) else "",
                       thumbnail=thumbnail if web_url(thumbnail) else "")
        rows.append(row)
        if len(rows) == max_results:
            break
    return rows


class SearchClient:
    """Cache-first SearXNG transport.

    Only successful responses are reused. Transport errors and responses with
    no usable results plus failed engines remain retryable. ``images`` is a
    text-query image search category, not an upload/reverse-image search API.
    """

    namespace = "search"
    provider = "searxng"
    language = "en"
    normalizer_version = 1

    def __init__(self, *, cache: EvidenceCache, base_url: str,
                 mode: str = "online", timeout: float = 20,
                 max_results: int = 10):
        if mode not in ("online", "offline"):
            raise ValueError("MAR_CACHE_MODE must be online or offline")
        if not web_url(base_url):
            raise ValueError("SearXNG base_url must be an HTTP(S) URL")
        parsed = urlparse(base_url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("SearXNG base_url must not contain credentials, a query, or a fragment")
        if timeout <= 0 or not isinstance(max_results, int) or max_results <= 0:
            raise ValueError("Search timeout and max_results must be positive")
        self.cache = cache
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.timeout = timeout
        self.max_results = max_results

    def cache_key(self, query: str, category: str = "general") -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must be a non-empty string")
        if category not in ("general", "images"):
            raise ValueError("Search category must be general or images")
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "query": query.strip(),
            "category": category,
            "max_results": self.max_results,
            "language": self.language,
            "safesearch": 1,
            "pageno": 1,
            "normalizer_version": self.normalizer_version,
        }

    def _request(self, query: str, category: str) -> dict:
        # Do not send a localhost service request through inherited HTTP proxies.
        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            response = client.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json", "categories": category,
                        "language": self.language, "safesearch": 1, "pageno": 1},
            )
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ValueError("Invalid SearXNG response: expected results array")
        return data

    def search(self, query: str, category: str = "general") -> tuple[list, list]:
        started = time.monotonic()
        key = self.cache_key(query, category)
        event = {
            "stage": "search", "provider": self.provider,
            "query": key["query"], "category": category, "mode": self.mode,
            "cache_key": self.cache.key_hash(key),
        }
        cached = self.cache.get(self.namespace, key)
        if (cached is not None and cached.get("status") in ("ok", "no_results")
                and isinstance(cached.get("rows"), list)):
            rows = cached["rows"]
            event.update(
                status=cached["status"], count=len(rows), cache="hit", cached=True,
                retrieved_at=cached.get("retrieved_at"),
                partial=bool(cached.get("partial")),
                unresponsive_engines=cached.get("unresponsive_engines", []),
                seconds=time.monotonic() - started,
            )
            return rows, [event]
        if self.mode == "offline":
            event.update(status="cache_miss", error_type="OfflineCacheMiss",
                         reason="offline_cache_miss", cache="miss", cached=False,
                         retryable=False, count=0, seconds=time.monotonic() - started)
            return [], [event]

        event.update(cache="miss", cached=False)
        try:
            data = self._request(key["query"], category)
            raw = data["results"]
            failures = data.get("unresponsive_engines") or []
            if not isinstance(failures, list):
                raise ValueError("Invalid SearXNG unresponsive_engines field")
            rows = normalize_rows(raw, category, self.max_results)
            if not rows and failures:
                # JSON does not enumerate all healthy engines. No results plus
                # failures is not a verified zero-hit response.
                event.update(status="error", error_type="SearXNGEngineError",
                             reason="no_usable_results_with_engine_failures",
                             unresponsive_engines=failures, partial=False,
                             count=0, retryable=True,
                             seconds=time.monotonic() - started)
                return [], [event]
            if raw and not rows:
                raise ValueError("SearXNG returned no usable result records")
            status = "ok" if rows else "no_results"
            retrieved_at = datetime.now(timezone.utc).isoformat()
            value = {
                "status": status, "provider": self.provider,
                "retrieved_at": retrieved_at, "rows": rows,
                "partial": bool(failures), "unresponsive_engines": failures,
                "raw_response": data,
            }
            self.cache.put(self.namespace, key, value)
            event.update(status=status, count=len(rows), partial=bool(failures),
                         unresponsive_engines=failures, retrieved_at=retrieved_at,
                         seconds=time.monotonic() - started)
            return rows, [event]
        except Exception as exc:
            # Exception messages can contain response bodies or secrets.
            event.update(status="error", error_type=type(exc).__name__,
                         count=0, retryable=True, seconds=time.monotonic() - started)
            if isinstance(exc, httpx.HTTPStatusError):
                status_code = exc.response.status_code
                event.update(http_status=status_code,
                             retryable=status_code == 429 or status_code >= 500)
            return [], [event]
