import json
from pathlib import Path
import tempfile
import unittest

from retrieval.cache import EvidenceCache
from retrieval.search import SearchClient
from scripts.verify_cached_run import verify_run


class VerifyCachedRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        (self.run / "records").mkdir(parents=True)
        (self.run / "config.json").write_text(json.dumps({
            "search_provider": "searxng", "search_timeout": 20,
        }))
        self.db = self.root / "cache.sqlite3"
        self.cache = EvidenceCache(self.db)
        self.addCleanup(self.cache.close)
        self.client = SearchClient(cache=self.cache, base_url="http://127.0.0.1:19999",
                                   max_results=7)

    def event(self, query, status="ok", *, cached=True):
        key = self.client.cache_key(query)
        if cached:
            self.cache.put("search", key, {
                "status": "ok", "rows": [{"title": "Evidence", "href": "https://example.org",
                                         "body": "Cached evidence"}],
            })
        return {"stage": "search", "provider": "searxng", "category": "general",
                "query": query, "status": status, "cache_key": self.cache.key_hash(key)}

    def record(self, sample_id, status, events):
        (self.run / "records" / f"{sample_id}.json").write_text(json.dumps({
            "sample_id": sample_id, "status": status, "retrieval_diagnostics": events,
        }))

    def test_reconstructs_exact_parameters_and_keeps_failed_query_separate(self):
        (self.run / "config.json").write_text(json.dumps({
            "search_config": {"base_url": "http://127.0.0.1:18888", "max_results": 10},
        }))
        good = self.event("good query")
        failed = self.event("earlier failed query", status="error")
        self.record("DGM", "ok", [good, good])
        self.record("MM", "error", [failed])
        report = verify_run(self.run, self.db)
        self.assertEqual(report["network_attempts"], 0)
        self.assertTrue(report["completed_run_queries_verified"])
        self.assertEqual(report["completed_records"]["unique_queries_per_record"], 1)
        self.assertEqual(report["completed_records"]["verified_successful_query_reuse"], 1)
        self.assertEqual(report["failed_records"]["cache_hits"], 1)
        self.assertEqual(report["failed_records"]["verified_successful_query_reuse"], 0)
        params = report["records"][0]["queries"][0]["parameters"]
        self.assertEqual(params["max_results"], 7)
        self.assertEqual(params["base_url"], "http://127.0.0.1:19999")
        self.assertEqual(report["cache_open_mode"], "read_only")

    def test_missing_entry_never_calls_network_or_counts_as_completed(self):
        client = SearchClient(cache=self.cache, base_url="http://127.0.0.1:18888")
        key = client.cache_key("missing query")
        self.record("DGM", "ok", [{
            "stage": "search", "query": "missing query", "status": "ok",
            "cache_key": self.cache.key_hash(key),
        }])
        report = verify_run(self.run, self.db)
        self.assertEqual(report["network_attempts"], 0)
        self.assertEqual(report["completed_records"]["cache_misses"], 1)
        self.assertFalse(report["completed_run_queries_verified"])
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_missing_entry_restores_nondefault_search_config(self):
        key = self.client.cache_key("failed query")
        key.update(language="zh", safesearch=0, pageno=2, normalizer_version=3)
        (self.run / "config.json").write_text(json.dumps({
            "search_config": {name: value for name, value in key.items() if name != "query"},
        }))
        self.record("MM", "error", [{
            "stage": "search", "query": "failed query", "status": "error",
            "cache_key": self.cache.key_hash(key),
        }])
        report = verify_run(self.run, self.db)
        self.assertEqual(report["network_attempts"], 0)
        self.assertEqual(report["failed_records"]["cache_misses"], 1)
        self.assertEqual(report["failed_records"]["parameter_mismatches"], 0)
        result = report["records"][0]["queries"][0]
        self.assertEqual(result["parameter_source"], "run_search_config")
        self.assertEqual(result["parameters"], key)
        self.assertFalse(result["verified_reusable"])


if __name__ == "__main__":
    unittest.main()
