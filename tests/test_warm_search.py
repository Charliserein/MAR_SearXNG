import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from retrieval.cache import EvidenceCache
from retrieval.search import SearchClient
from scripts.warm_search_cache import build_tasks, load_checkpoint, warm_tasks

SUCCESS = {"results": [{"title": "Result", "url": "https://example.org/page",
                        "content": "Evidence", "img_src": "https://example.org/image.jpg"}],
           "unresponsive_engines": []}
FAILED = {"results": [], "unresponsive_engines": [["engine", "timeout", False]]}


class WarmSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = EvidenceCache(self.root / "cache.sqlite3")
        self.addCleanup(self.cache.close)
        self.client = SearchClient(cache=self.cache, base_url="http://127.0.0.1:18888")
        self.checkpoint = self.root / "warmup.jsonl"

    def tasks(self, count=3):
        return [{"query": f"claim {n}", "category": "images",
                 "samples": [{"dataset": "test", "sample_id": str(n)}]}
                for n in range(count)]

    def run_tasks(self, **kwargs):
        return warm_tasks(kwargs.pop("tasks", self.tasks()), self.client,
                          self.checkpoint, **kwargs)

    def test_loader_deduplicates_category_query_and_never_includes_labels(self):
        for name, label in (("first", "real"), ("second", "fake")):
            path = self.root / name
            path.mkdir()
            (path / "samples.json").write_text(json.dumps([
                {"sample_id": name, "text": " A news\n claim ", "ground_truth": label,
                 "fake_cls": label, "captions": "Secret reference"}
            ]))
        tasks, count = build_tasks(self.root, ["first", "second"], ["general", "images"])
        self.assertEqual(count, 2)
        self.assertEqual(len(tasks), 2)
        self.assertEqual({t["category"] for t in tasks}, {"general", "images"})
        self.assertTrue(all(t["query"] == "A news claim" for t in tasks))
        self.assertTrue(all(len(t["samples"]) == 2 for t in tasks))
        serialized = json.dumps(tasks)
        self.assertNotIn("ground_truth", serialized)
        self.assertNotIn("fake_cls", serialized)
        self.assertNotIn("Secret reference", serialized)

    def test_restart_hits_cache_without_backend(self):
        with patch.object(SearchClient, "_request", return_value=SUCCESS) as backend:
            first = self.run_tasks()
        self.assertEqual(backend.call_count, 3)
        self.assertEqual(first["success"], 3)
        with patch.object(SearchClient, "_request", side_effect=AssertionError("network")) as backend:
            repeated = self.run_tasks()
        backend.assert_not_called()
        self.assertEqual(repeated["attempted"], 3)
        self.assertEqual(repeated["cachehits"], 3)
        self.assertEqual(repeated["network_attempts"], 0)
        self.assertTrue(repeated["coverage"]["complete"])
        self.assertEqual(len(load_checkpoint(self.checkpoint)), 3)

    def test_new_limit_skips_cached_work_and_advances_on_restart(self):
        with patch.object(SearchClient, "_request", return_value=SUCCESS) as backend:
            first = self.run_tasks(max_new=1)
            second = self.run_tasks(max_new=1)
        self.assertEqual(first["attempted"], 1)
        self.assertEqual(first["coverage"]["pending"], 2)
        self.assertEqual(second["cachehits"], 1)
        self.assertEqual(second["network_attempts"], 1)
        self.assertEqual(second["coverage"]["success"], 2)
        self.assertEqual(second["coverage"]["pending"], 1)
        self.assertEqual(backend.call_count, 2)

    def test_failure_requires_explicit_retry_and_is_not_zero_results(self):
        with patch.object(SearchClient, "_request", return_value=FAILED):
            first = self.run_tasks(tasks=self.tasks(1))
        self.assertEqual(first["errors"], 1)
        self.assertEqual(first["empty"], 0)
        with patch.object(SearchClient, "_request", side_effect=AssertionError("network")) as backend:
            repeated = self.run_tasks(tasks=self.tasks(1))
        backend.assert_not_called()
        self.assertEqual(repeated["skipped_errors"], 1)
        self.assertEqual(repeated["coverage"]["errors"], 1)
        self.assertFalse(repeated["coverage"]["complete"])
        with patch.object(SearchClient, "_request", return_value=SUCCESS) as backend:
            retried = self.run_tasks(tasks=self.tasks(1), retry_errors=True)
        backend.assert_called_once()
        self.assertEqual(retried["success"], 1)
        self.assertTrue(retried["coverage"]["complete"])

    def test_offline_miss_never_calls_network_or_erases_old_failure(self):
        with patch.object(SearchClient, "_request", return_value=FAILED):
            self.run_tasks(tasks=self.tasks(1))
        self.client.mode = "offline"
        with patch.object(SearchClient, "_request", side_effect=AssertionError("network")) as backend:
            result = self.run_tasks()
        backend.assert_not_called()
        self.assertEqual(result["cache_misses"], 3)
        self.assertEqual(result["network_attempts"], 0)
        self.assertFalse(result["stopped"])
        self.client.mode = "online"
        with patch.object(SearchClient, "_request", return_value=SUCCESS) as backend:
            online = self.run_tasks()
        self.assertEqual(backend.call_count, 2)
        self.assertEqual(online["skipped_errors"], 1)

    def test_ten_consecutive_backend_errors_stop_the_batch(self):
        with patch.object(SearchClient, "_request", return_value=FAILED) as backend:
            summary = self.run_tasks(tasks=self.tasks(30), workers=1)
        self.assertEqual(backend.call_count, 10)
        self.assertEqual(summary["errors"], 10)
        self.assertEqual(summary["coverage"]["pending"], 20)
        self.assertTrue(summary["stopped"])
        self.assertEqual(summary["stop_reason"], "ten_consecutive_backend_errors")

    def test_parallel_error_circuit_has_bounded_inflight_requests(self):
        with patch.object(SearchClient, "_request", return_value=FAILED) as backend:
            summary = self.run_tasks(tasks=self.tasks(40), workers=4)
        self.assertGreaterEqual(backend.call_count, 10)
        self.assertLessEqual(backend.call_count, 13)
        self.assertTrue(summary["stopped"])

    def test_true_zero_results_is_successful_empty_cache(self):
        with patch.object(SearchClient, "_request", return_value={"results": []}):
            first = self.run_tasks(tasks=self.tasks(1))
        self.assertEqual(first["empty"], 1)
        self.assertEqual(first["errors"], 0)
        self.assertTrue(first["coverage"]["complete"])
        with patch.object(SearchClient, "_request", side_effect=AssertionError("network")) as backend:
            second = self.run_tasks(tasks=self.tasks(1))
        backend.assert_not_called()
        self.assertEqual(second["empty"], 1)
        self.assertEqual(second["cachehits"], 1)

    def test_partial_append_is_recovered_using_durable_cache(self):
        with patch.object(SearchClient, "_request", return_value=SUCCESS):
            self.run_tasks(tasks=self.tasks(1))
        with self.checkpoint.open("ab") as output:
            output.write(b'{"kind":"search_result","key_hash":"unfinished')
        with patch.object(SearchClient, "_request", side_effect=AssertionError("network")) as backend:
            result = self.run_tasks(tasks=self.tasks(1))
        backend.assert_not_called()
        self.assertEqual(result["cachehits"], 1)
        self.assertEqual(len(load_checkpoint(self.checkpoint)), 1)
        self.assertEqual(len(self.checkpoint.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
