import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from retrieval.cache import EvidenceCache
from retrieval.reverse import CachedReverseSearch, SerperLensResult, claim_query


class ReverseCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root / "image.png"
        self.image.write_bytes(b"image-content")
        self.cache = EvidenceCache(self.root / "cache.sqlite3")

    def client(self, mode="online"):
        c = CachedReverseSearch(self.cache, mode=mode)
        c._matches = Mock(return_value=([SerperLensResult(
            "title", "example.org", "https://example.org/page", "https://example.org/a.png")], []))
        c._extract = Mock(return_value={"status": "ok", "captions": "Evidence",
                                        "entries": [], "events": [], "matches": 1})
        return c

    def test_repeated_image_reuses_all_evidence_without_network(self):
        c = self.client()
        first = c.search(self.image)
        second = self.client()
        repeated = second.search(self.image)
        self.assertEqual(first["captions"], repeated["captions"])
        self.assertEqual(repeated["cache"], "hit")
        second._matches.assert_not_called()
        second._extract.assert_not_called()

    def test_offline_miss_never_uploads_or_extracts(self):
        c = self.client("offline")
        self.assertEqual(c.search(self.image)["status"], "cache_miss")
        c._matches.assert_not_called()
        c._extract.assert_not_called()

    def test_extraction_failure_retries_without_paying_again(self):
        c = self.client()
        c._extract.return_value = {"status": "error", "captions": "", "entries": [], "events": []}
        self.assertEqual(c.search(self.image)["status"], "error")
        second = self.client()
        self.assertEqual(second.search(self.image)["status"], "ok")
        second._matches.assert_not_called()
        second._extract.assert_called_once()

    def test_filter_change_reuses_raw_paid_search(self):
        first = self.client()
        first.search(self.image)
        second = self.client()
        second.similarity_cutoff = 12
        second.search(self.image)
        second._matches.assert_not_called()
        second._extract.assert_called_once()

    def test_different_image_and_text_queries_do_not_share_results(self):
        c = self.client()
        c.search(self.image, backend="searxng_images", query="first")
        c.search(self.image, backend="searxng_images", query="second")
        self.assertEqual(c._matches.call_count, 2)
        self.image.write_bytes(b"different-image")
        c.search(self.image, backend="searxng_images", query="first")
        self.assertEqual(c._matches.call_count, 3)

    def test_claim_query_uses_only_input_text(self):
        self.assertEqual(claim_query("  A news\n claim  "), "A news claim")


if __name__ == "__main__":
    unittest.main()
