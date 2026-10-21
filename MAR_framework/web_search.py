"""Cached SearXNG search facade shared by MAR and its evaluation preflight."""
import os
from pathlib import Path
import sys
import time

# Original MAR scripts are also launched directly from MAR_framework/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.cache import EvidenceCache
from retrieval.search import SearchClient, web_url


def api_keys(name):
    return [key.strip() for key in os.getenv(name, "").split(",") if key.strip()]


class SearchUnavailable(RuntimeError):
    pass


class WebSearchClient:
    def __init__(self, provider=None, proxy=None, timeout=None, max_results=10):
        self.provider = (provider or os.getenv("MAR_SEARCH_PROVIDER", "searxng")).lower()
        if self.provider == "auto":
            self.provider = "searxng"
        if self.provider != "searxng":
            raise SearchUnavailable(
                "This project requires MAR_SEARCH_PROVIDER=searxng; "
                "no Serper or DDGS fallback is enabled")
        # Keep attributes consumed by evaluate_batch.py. Upstream engine proxies
        # belong in SearXNG deployment; this client connects to the service directly.
        self.proxy = proxy or os.getenv("MAR_SEARCH_PROXY") or None
        self.timeout = float(timeout if timeout is not None
                             else os.getenv("MAR_SEARCH_TIMEOUT", "20"))
        self.max_results = max_results
        self.attempts = int(os.getenv("MAR_SEARCH_ATTEMPTS", "2"))
        if self.attempts not in (1, 2, 3) or self.timeout <= 0:
            raise ValueError("Search attempts must be 1..3 and timeout must be positive")
        self.mode = os.getenv("MAR_CACHE_MODE", "online")
        cache_path = Path(os.getenv("MAR_CACHE_DB", str(PROJECT_ROOT / "cache" / "evidence.sqlite3")))
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path
        self.cache = EvidenceCache(cache_path)
        self.client = SearchClient(
            cache=self.cache,
            base_url=os.getenv("MAR_SEARXNG_URL", "http://127.0.0.1:18888"),
            mode=self.mode, timeout=self.timeout, max_results=self.max_results,
        )

    def search(self, query):
        """Return normalized rows and events without hiding provider failures."""
        events = []
        for attempt in range(1, self.attempts + 1):
            rows, current = self.client.search(query)
            for event in current:
                event["attempt"] = attempt
            events.extend(current)
            last = current[-1]
            if last["status"] != "error" or not last.get("retryable"):
                return rows, events
            if attempt < self.attempts:
                time.sleep(min(attempt, 2))
        return [], events

    def preflight(self):
        rows, events = self.search("Apollo 11 NASA moon landing")
        return {
            "status": "ok" if rows else "error",
            "provider": self.provider,
            "urls": [r["href"] for r in rows],
            "events": events,
        }
