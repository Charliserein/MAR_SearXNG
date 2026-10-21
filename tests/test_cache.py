"""Offline deterministic tests: no real provider or model is contacted."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from retrieval.cache import EvidenceCache
from retrieval.search import SearchClient
from MAR_framework.web_search import WebSearchClient


RESULT = {"title": "Apollo &amp; Moon", "url": "https://www.nasa.gov/apollo/",
          "content": "The <b>Apollo</b> mission.", "engine": "example",
          "img_src": "https://www.nasa.gov/apollo.jpg"}
SUCCESS = {"results": [RESULT], "unresponsive_engines": []}


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "cache.sqlite3"
        self.cache = EvidenceCache(self.path)

    def tearDown(self):
        self.cache.close()
        self.directory.cleanup()

    def client(self, **kwargs):
        options = dict(cache=self.cache, base_url="http://127.0.0.1:18888")
        options.update(kwargs)
        return SearchClient(**options)

    def test_canonical_keys_persist_and_values_are_detached(self):
        key = {"b": ["中文", 3], "a": 1}
        self.cache.put("test", key, {"status": "ok", "rows": [{"n": 1}]})
        value = self.cache.get("test", {"a": 1, "b": ["中文", 3]})
        value["rows"][0]["n"] = 9
        with EvidenceCache(self.path) as other:
            self.assertEqual(other.get("test", key)["rows"][0]["n"], 1)
            self.assertEqual(other.stats()["by_namespace"], {"test": 1})
        self.assertIsNone(self.cache.get("different", key))

    def test_shared_instance_threaded_writes_are_atomic(self):
        def write(i):
            self.cache.put("thread", {"i": i}, {"status": "ok", "payload": [i] * 50})
            return self.cache.get("thread", {"i": i})
        with ThreadPoolExecutor(max_workers=8) as pool:
            values = list(pool.map(write, range(48)))
        self.assertEqual([v["payload"][0] for v in values], list(range(48)))
        self.assertEqual(self.cache.stats()["entries"], 48)

    def test_cached_second_search_uses_no_backend(self):
        with patch.object(SearchClient, "_request", return_value=SUCCESS) as backend:
            rows, first = self.client().search("Apollo")
            again, second = self.client().search("Apollo")
            offline, third = self.client(mode="offline").search("Apollo")
        self.assertEqual(backend.call_count, 1)
        self.assertEqual(rows, again)
        self.assertEqual(rows, offline)
        self.assertEqual(rows[0]["body"], "The Apollo mission.")
        self.assertEqual(rows[0]["title"], "Apollo & Moon")
        self.assertEqual(first[-1]["cache"], "miss")
        self.assertEqual(second[-1]["cache"], "hit")
        self.assertTrue(third[-1]["cached"])

    def test_offline_cache_miss_never_contacts_backend(self):
        with patch.object(SearchClient, "_request", side_effect=AssertionError("network")) as backend:
            rows, events = self.client(mode="offline").search("missing")
        backend.assert_not_called()
        self.assertEqual(rows, [])
        self.assertEqual(events[-1]["status"], "cache_miss")
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_cache_key_separates_parameters(self):
        with patch.object(SearchClient, "_request", return_value=SUCCESS) as backend:
            self.client().search("Apollo")
            self.client(max_results=3).search("Apollo")
            self.client(base_url="http://127.0.0.1:18889").search("Apollo")
            self.client().search("Apollo", "images")
            self.client().search("Apollo 11")
            language_client = self.client()
            language_client.language = "zh"
            language_client.search("Apollo")
            self.client().search("Apollo")
        self.assertEqual(backend.call_count, 6)
        self.assertEqual(self.cache.stats()["entries"], 6)

    def test_http_error_is_not_cached_and_can_be_retried(self):
        request = httpx.Request("GET", "http://127.0.0.1:18888/search")
        response = httpx.Response(503, request=request)
        error = httpx.HTTPStatusError("unavailable", request=request, response=response)
        with patch.object(SearchClient, "_request", side_effect=[error, SUCCESS]) as backend:
            rows, events = self.client().search("Apollo")
            self.assertEqual(rows, [])
            self.assertEqual(events[-1]["status"], "error")
            self.assertEqual(events[-1]["http_status"], 503)
            self.assertEqual(self.cache.stats()["entries"], 0)
            rows, events = self.client().search("Apollo")
        self.assertEqual(backend.call_count, 2)
        self.assertTrue(rows)
        self.assertEqual(events[-1]["status"], "ok")

    def test_engine_failure_and_successful_zero_results_are_distinct(self):
        failures = [["google", "CAPTCHA", False], ["bing", "timeout", True]]
        responses = [
            {"results": [], "unresponsive_engines": failures},
            {"results": [RESULT], "unresponsive_engines": failures[:1]},
            {"results": [], "unresponsive_engines": []},
        ]
        with patch.object(SearchClient, "_request", side_effect=responses):
            rows, failed = self.client().search("failed")
            self.assertEqual(rows, [])
            self.assertEqual(failed[-1]["status"], "error")
            self.assertEqual(failed[-1]["unresponsive_engines"], failures)
            self.assertEqual(self.cache.stats()["entries"], 0)
            rows, partial = self.client().search("partial")
            self.assertTrue(rows)
            self.assertTrue(partial[-1]["partial"])
            rows, empty = self.client().search("empty")
            self.assertEqual(rows, [])
            self.assertEqual(empty[-1]["status"], "no_results")
        with patch.object(SearchClient, "_request", side_effect=AssertionError("network")):
            _, partial_hit = self.client().search("partial")
            _, empty_hit = self.client().search("empty")
        self.assertTrue(partial_hit[-1]["partial"])
        self.assertEqual(empty_hit[-1]["status"], "no_results")
        self.assertEqual(self.cache.stats()["entries"], 2)

    def test_malformed_nonempty_results_are_not_cached_as_zero(self):
        with patch.object(SearchClient, "_request", return_value={"results": [{"url": "javascript:void(0)"}]}):
            _, events = self.client().search("malformed")
        self.assertEqual(events[-1]["status"], "error")
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_image_results_include_canonical_image_fields(self):
        data = {"results": [
            {"title": "Moon", "url": "https://nasa.gov/photo", "content": "",
             "img_src": "https://nasa.gov/moon.jpg", "thumbnail_src": "https://nasa.gov/thumb.jpg"},
            {"title": "Earth", "img_src": "https://nasa.gov/earth.jpg",
             "thumbnail": "data:image/jpeg;base64,ignored"},
            {"title": "Duplicate", "url": "https://nasa.gov/photo",
             "img_src": "https://nasa.gov/moon.jpg"},
        ]}
        with patch.object(SearchClient, "_request", return_value=data):
            rows, events = self.client().search("Moon photos", category="images")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["image_url"], "https://nasa.gov/moon.jpg")
        self.assertEqual(rows[0]["thumbnail"], "https://nasa.gov/thumb.jpg")
        self.assertEqual(rows[1]["href"], "https://nasa.gov/earth.jpg")
        self.assertEqual(rows[1]["thumbnail"], "")
        self.assertEqual(events[-1]["category"], "images")

    def test_searxng_request_uses_json_and_english_without_proxy(self):
        response = httpx.Response(
            200, json=SUCCESS,
            request=httpx.Request("GET", "http://127.0.0.1:18888/search"))
        with patch("retrieval.search.httpx.Client") as factory:
            client = factory.return_value.__enter__.return_value
            client.get.return_value = response
            rows, _ = self.client().search("Apollo", "images")
        self.assertTrue(rows)
        self.assertFalse(factory.call_args.kwargs["trust_env"])
        args, kwargs = client.get.call_args
        self.assertEqual(args[0], "http://127.0.0.1:18888/search")
        self.assertEqual(kwargs["params"]["format"], "json")
        self.assertEqual(kwargs["params"]["language"], "en")
        self.assertEqual(kwargs["params"]["categories"], "images")

    def test_facade_offline_preflight_is_safe_and_no_serper_fallback(self):
        env = {"MAR_CACHE_DB": str(self.path), "MAR_CACHE_MODE": "offline",
               "MAR_SEARCH_PROVIDER": "auto", "MAR_SERPER_API_KEYS": "must-not-be-used"}
        with patch.dict("os.environ", env), patch.object(SearchClient, "_request") as backend:
            facade = WebSearchClient()
            try:
                result = facade.preflight()
            finally:
                facade.cache.close()
        backend.assert_not_called()
        self.assertEqual(facade.provider, "searxng")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["events"][-1]["status"], "cache_miss")


if __name__ == "__main__":
    unittest.main()
