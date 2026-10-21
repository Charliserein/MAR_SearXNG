import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

from retrieval.cache import EvidenceCache
from retrieval.extract import extract_matches_cached
from retrieval.reverse import SerperLensResult


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root / "original.png"
        Image.new("RGB", (32, 32), color="red").save(self.image)
        self.image_bytes = self.image.read_bytes()
        self.cache = EvidenceCache(self.root / "cache.sqlite3")
        self.addCleanup(self.cache.close)
        self.blobs = self.root / "blobs"
        self.match = SerperLensResult(
            "Fallback title", "example.org", "https://example.org/news",
            "https://example.org/image.png")
        self.calls = []
        self.actual_client = httpx.Client

    def run_extract(self, handler, *, matches=None, **kwargs):
        def tracked(request):
            self.calls.append(str(request.url))
            return handler(request)
        transport = httpx.MockTransport(tracked)
        def factory(**options):
            options.pop("proxy", None)
            return self.actual_client(transport=transport, **options)
        with patch("retrieval.extract.httpx.Client", side_effect=factory):
            return extract_matches_cached(
                self.image, [self.match] if matches is None else matches,
                cache=self.cache, blob_dir=self.blobs, **kwargs)

    def handler(self, html):
        def respond(request):
            if request.url.path == "/image.png":
                return httpx.Response(200, content=self.image_bytes,
                                      headers={"Content-Type": "image/png"})
            return httpx.Response(200, text=html,
                                  headers={"Content-Type": "text/html"})
        return respond

    def test_figcaption_and_persistent_body_cache(self):
        handler = self.handler(
            "<title>News source</title><figure><img src='/image.png' alt='Short alt'>"
            "<figcaption>Observed <b>event</b> caption.</figcaption></figure>"
            "<img src='/unrelated.jpg' alt='Must not fetch'>")
        first = self.run_extract(handler)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["entries"][0]["caption"], "Observed event caption.")
        self.assertEqual(first["entries"][0]["caption_origin"], "figcaption")
        self.assertEqual(len(self.calls), 2)
        self.assertIn("[Re1] Source: example.org\nTitle: News source\n", first["captions"])
        with patch("retrieval.extract.httpx.Client.stream",
                   side_effect=AssertionError("Cache hit must not download")):
            second = extract_matches_cached(
                self.image, [self.match], cache=self.cache, blob_dir=self.blobs)
        self.assertEqual(second["captions"], first["captions"])
        self.assertEqual(second["events"][-1]["image_cache"], "hit")
        self.assertEqual(second["events"][-1]["page_cache"], "hit")
        self.assertEqual(len(list(self.blobs.iterdir())), 2)

    def test_relative_img_alt_used_without_unrelated_caption(self):
        result = self.run_extract(self.handler(
            "<img src='/unrelated.png' alt='Wrong image'>"
            "<img data-src='/image.png' alt=' Correct   event caption '>"
            "<figcaption>Unrelated later caption</figcaption>"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["entries"][0]["caption"], "Correct event caption")
        self.assertEqual(result["entries"][0]["caption_origin"], "alt")

    def test_unmatched_page_image_is_not_evidence(self):
        result = self.run_extract(self.handler(
            "<title>Title alone is not evidence</title>"
            "<figure><img src='/other.png'><figcaption>Wrong image caption</figcaption></figure>"))
        self.assertEqual(result["status"], "no_caption")
        self.assertEqual(result["captions"], "")

    def test_visually_different_image_never_fetches_page(self):
        with patch("retrieval.extract.compare_images", return_value=(False, 31)):
            result = self.run_extract(self.handler("<img src='/image.png' alt='Wrong'>"))
        self.assertEqual(result["status"], "no_caption")
        self.assertEqual(self.calls, [self.match.image_url])
        self.assertEqual(result["events"][0]["status"], "filtered")

    def test_http_error_remains_retryable_and_is_not_cached(self):
        result = self.run_extract(lambda request: httpx.Response(503))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["events"][0]["http_status"], 503)
        self.assertEqual(self.cache.stats()["entries"], 0)
        retry = self.run_extract(self.handler("<img src='/image.png' alt='Recovered caption'>"))
        self.assertEqual(retry["status"], "ok")
        self.assertEqual(len(self.calls), 3)

    def test_page_http_failure_does_not_redownload_successful_image(self):
        first = self.run_extract(lambda request:
            httpx.Response(200, content=self.image_bytes,
                           headers={"Content-Type": "image/png"})
            if request.url.path == "/image.png" else httpx.Response(403))
        self.assertEqual(first["status"], "error")
        second = self.run_extract(self.handler("<img src='/image.png' title='Recovered'>"))
        self.assertEqual(second["status"], "ok")
        self.assertEqual(self.calls.count(self.match.image_url), 1)
        self.assertEqual(second["entries"][0]["caption_origin"], "title")

    def test_size_limit_is_error_and_not_saved(self):
        with patch("retrieval.extract.MAX_IMAGE_BYTES", 20):
            result = self.run_extract(lambda request:
                httpx.Response(200, content=b"x" * 21,
                               headers={"Content-Type": "image/png"}))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["events"][0]["reason"], "body_too_large")
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_stream_size_limit_without_content_length(self):
        class Stream(httpx.SyncByteStream):
            def __iter__(self):
                yield b"x" * 12
                yield b"x" * 12
        with patch("retrieval.extract.MAX_IMAGE_BYTES", 20):
            result = self.run_extract(lambda request:
                httpx.Response(200, stream=Stream(),
                               headers={"Content-Type": "image/png"}))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["events"][0]["reason"], "body_too_large")

    def test_expired_budget_does_not_start_http(self):
        with patch("retrieval.extract.time.monotonic", side_effect=[0.0, 46.0]):
            result = self.run_extract(lambda request: self.fail("Must not download"))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["events"][0]["reason"], "time_budget_exceeded")
        self.assertEqual(self.calls, [])

    def test_candidate_count_is_bounded(self):
        matches = [SerperLensResult(
            "Title", "example.org", f"https://example.org/page{i}",
            f"https://example.org/image{i}.png") for i in range(10)]
        result = self.run_extract(lambda request: httpx.Response(404), matches=matches)
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(result["extraction_config"]["max_candidates"], 5)
        self.assertEqual(result["status"], "error")


if __name__ == "__main__":
    unittest.main()
