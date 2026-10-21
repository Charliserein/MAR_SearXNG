"""Evidence preparation fingerprints and refresh behavior; no live requests."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from retrieval.cache import EvidenceCache
from retrieval.reverse import CachedReverseSearch, SerperLensResult
from retrieval.search import SearchClient
from scripts import prepare_cached_evidence as prepare


class PrepareEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root / "image.jpg"
        self.image.write_bytes(b"immutable-image")
        self.sample = {
            "sample_id": "sample-1", "text": "The original news claim.",
            "image_path": str(self.image),
            "image_sha256": hashlib.sha256(self.image.read_bytes()).hexdigest(),
            "captions": "", "ground_truth": "original", "fake_cls": "orig",
        }
        self.cache = EvidenceCache(self.root / "cache.sqlite3")
        self.addCleanup(self.cache.close)
        self.client = self.reverse()

    def reverse(self, *, base_url="http://127.0.0.1:18888", max_results=10,
                similarity_cutoff=10, mode="online"):
        search = SearchClient(cache=self.cache, base_url=base_url,
                              max_results=max_results, mode=mode)
        return CachedReverseSearch(self.cache, search_client=search, mode=mode,
                                   max_results=max_results,
                                   similarity_cutoff=similarity_cutoff)

    def specs(self, sample=None, client=None, backend="searxng_images"):
        return prepare.generation_specs(sample or self.sample, client or self.client, backend)

    def record(self, source="searxng_images", captions="Verified caption"):
        return {**prepare.initial_record(self.sample), "source": source,
                "status": "ok" if captions else "no_caption",
                "captions": captions, "captions_sha256": prepare.text_digest(captions)}

    def generated(self):
        return prepare.mark_generated(self.record(), self.sample, self.specs())

    def test_generated_record_is_reused_only_for_matching_configuration(self):
        record = self.generated()
        self.cache.put("sample_evidence",
                       prepare.generation_cache_key(record["generation_spec"]), record)
        selected = prepare.select_record(self.sample, cache=self.cache, specs=self.specs())
        self.assertEqual(selected["captions"], "Verified caption")
        self.assertTrue(prepare.valid_generated(selected, self.sample, self.specs()))
        for client in (self.reverse(base_url="http://127.0.0.1:18889"),
                       self.reverse(max_results=5),
                       self.reverse(similarity_cutoff=3)):
            with self.subTest(client=client.search_client.base_url,
                              max_results=client.max_results):
                selected = prepare.select_record(
                    self.sample, cache=self.cache, specs=self.specs(client=client), previous=record)
                self.assertEqual(selected["status"], "pending")

    def test_query_and_extractor_version_invalidate_generated_sample_evidence(self):
        record = self.generated()
        changed = {**self.sample, "text": "A different query for the same image"}
        selected = prepare.select_record(changed, cache=self.cache,
                                         specs=self.specs(sample=changed), previous=record)
        self.assertEqual(selected["status"], "pending")
        with patch("retrieval.extract.EXTRACTION_VERSION", 99):
            selected = prepare.select_record(
                self.sample, cache=self.cache, specs=self.specs(), previous=record)
        self.assertEqual(selected["status"], "pending")

    def test_labels_do_not_affect_fingerprints_or_enter_the_specs(self):
        modified = {**self.sample, "ground_truth": "mismatch", "fake_cls": "test_label"}
        self.assertEqual(self.specs(), self.specs(sample=modified))
        encoded = json.dumps(self.specs())
        self.assertNotIn("ground_truth", encoded)
        self.assertNotIn("fake_cls", encoded)
        self.assertNotIn("test_label", encoded)

    def test_legacy_generated_record_is_not_treated_as_a_frozen_import(self):
        record = self.record()
        record["imported_from"] = "/unverified/old-project/reverse-evidence.json"
        selected = prepare.select_record(self.sample, cache=self.cache,
                                         specs=self.specs(), previous=record)
        self.assertEqual(selected["status"], "pending")

    def test_fixed_author_annotation_is_preserved_even_during_refresh(self):
        sample = {**self.sample, "captions": "Exact author annotation"}
        selected = prepare.select_record(sample, cache=self.cache,
                                         specs=self.specs(sample=sample), previous=self.generated())
        self.assertEqual(selected["captions"], sample["captions"])
        self.assertEqual(selected["status"], "author_annotation")
        self.assertFalse(prepare.needs_search(
            selected, backend="hybrid", mode="online", refresh=True, retry_empty=True))

    def test_verified_original_evidence_is_frozen_across_config_changes(self):
        original = self.record(source="serper_lens")
        provenance = {"path": "/original/reverse-evidence.json",
                      "manifest_sha256": "a" * 64, "input_sha256": "b" * 64}
        selected = prepare.select_record(
            self.sample, cache=self.cache, specs=self.specs(),
            original=original, provenance=provenance)
        self.assertEqual(selected["evidence_origin"], "frozen_import")
        self.assertFalse(prepare.needs_search(
            selected, backend="searxng_images", mode="online", refresh=True))
        changed_specs = self.specs(client=self.reverse(base_url="http://127.0.0.1:18889"))
        recovered = prepare.select_record(
            self.sample, cache=self.cache, specs=changed_specs)
        self.assertEqual(recovered, selected)
        changed_input = {**self.sample, "text": "This is a different input claim"}
        invalid = prepare.select_record(
            changed_input, cache=self.cache,
            specs=self.specs(sample=changed_input), previous=selected)
        self.assertEqual(invalid["status"], "pending")

    def test_generated_improvement_replaces_frozen_empty_without_old_provenance(self):
        original = self.record(source="serper_lens", captions="")
        provenance = {"path": "/original/reverse-evidence.json",
                      "manifest_sha256": "a" * 64, "input_sha256": "b" * 64}
        frozen = prepare.freeze_import(original, self.sample, provenance)
        improved = {**frozen, **self.record()}
        prepare.mark_generated(improved, self.sample, self.specs())
        self.assertNotIn("imported_from", improved)
        self.assertNotIn("import_provenance", improved)
        selected = prepare.select_record(
            self.sample, cache=self.cache, specs=self.specs(),
            previous=improved, original=original, provenance=provenance)
        self.assertEqual(selected["evidence_origin"], "generated")
        self.assertEqual(selected["captions"], "Verified caption")
        self.assertTrue(prepare.needs_search(
            selected, backend="searxng_images", mode="online", refresh=True))

    def test_refresh_reruns_extraction_but_reuses_paid_raw_candidates(self):
        client = self.reverse()
        client._matches = Mock(return_value=([SerperLensResult(
            "Title", "example.org", "https://example.org/page", "https://example.org/image.jpg")], []))
        client._extract = Mock(side_effect=[
            {"status": "ok", "captions": "first", "entries": [], "events": []},
            {"status": "ok", "captions": "refreshed", "entries": [], "events": []},
        ])
        first = client.search(self.image, backend="serper_lens")
        second = client.search(self.image, backend="serper_lens", refresh=True)
        self.assertEqual(first["captions"], "first")
        self.assertEqual(second["captions"], "refreshed")
        self.assertEqual(second["raw_cache"], "hit")
        client._matches.assert_called_once()
        self.assertEqual(client._extract.call_count, 2)
        client.mode = "offline"
        with self.assertRaises(ValueError):
            client.search(self.image, backend="serper_lens", refresh=True)
        self.assertEqual(client._extract.call_count, 2)

    def test_cli_preserves_ready_then_invalidates_configuration_and_refreshes(self):
        data_dir = self.root / "data/DGM4-author1000"
        data_dir.mkdir(parents=True)
        (data_dir / "samples.json").write_text(json.dumps([self.sample]))
        (self.root / "cache").mkdir()
        (self.root / "outputs").mkdir()
        env = {"MAR_CACHE_DB": str(self.root / "integration.sqlite3"),
               "MAR_SEARXNG_URL": "http://127.0.0.1:18888"}
        argv = ["prepare", "--dataset", "DGM4-author1000", "--mode", "online",
                "--backend", "searxng_images", "--source-root", str(self.root / "source")]
        match = SerperLensResult("Title", "example.org", "https://example.org/page",
                                "https://example.org/image.jpg")
        with patch.object(prepare, "ROOT", self.root), patch.dict("os.environ", env), \
             patch.object(CachedReverseSearch, "_matches", return_value=([match], [])) as matches, \
             patch.object(CachedReverseSearch, "_extract", return_value={
                 "status": "ok", "captions": "Evidence", "entries": [], "events": []}) as extract, \
             redirect_stdout(io.StringIO()):
            with patch("sys.argv", argv):
                self.assertEqual(prepare.main(), 0)
                self.assertEqual(prepare.main(), 0)
            self.assertEqual(matches.call_count, 1)
            self.assertEqual(extract.call_count, 1)
            with patch.dict("os.environ", {"MAR_SEARXNG_URL": "http://127.0.0.1:18889"}):
                with patch("sys.argv", argv):
                    self.assertEqual(prepare.main(), 0)
                self.assertEqual(matches.call_count, 2)
                self.assertEqual(extract.call_count, 2)
                with patch("sys.argv", argv + ["--refresh"]):
                    self.assertEqual(prepare.main(), 0)
            self.assertEqual(matches.call_count, 2)
            self.assertEqual(extract.call_count, 3)
        record = json.loads((data_dir / "reverse-evidence.json").read_text())["records"]["sample-1"]
        self.assertEqual(record["evidence_origin"], "generated")
        self.assertEqual(record["generation_spec"]["retrieval"]["search_config"]["base_url"],
                         "http://127.0.0.1:18889")

    def test_cli_refuses_refresh_in_offline_mode_before_opening_cache(self):
        with patch("sys.argv", ["prepare", "--mode", "offline", "--refresh"]), \
             patch.object(prepare, "EvidenceCache") as cache, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                prepare.main()
        self.assertEqual(error.exception.code, 2)
        cache.assert_not_called()

    def test_cli_refreshes_native_cached_matches_without_allow_paid(self):
        data_dir = self.root / "data/DGM4-author1000"
        data_dir.mkdir(parents=True)
        (data_dir / "samples.json").write_text(json.dumps([self.sample]))
        env = {"MAR_CACHE_DB": str(self.root / "native.sqlite3")}
        argv = ["prepare", "--dataset", "DGM4-author1000", "--mode", "online",
                "--backend", "serper_lens", "--source-root", str(self.root / "source")]
        match = SerperLensResult("Title", "example.org", "https://example.org/page",
                                "https://example.org/image.jpg")
        with patch.object(prepare, "ROOT", self.root), patch.dict("os.environ", env), \
             patch.object(CachedReverseSearch, "_matches", return_value=([match], [])) as matches, \
             patch.object(CachedReverseSearch, "_extract", return_value={
                 "status": "ok", "captions": "Evidence", "entries": [], "events": []}) as extract, \
             redirect_stdout(io.StringIO()):
            with patch("sys.argv", argv + ["--allow-paid"]):
                self.assertEqual(prepare.main(), 0)
            with patch("sys.argv", argv + ["--refresh"]):
                self.assertEqual(prepare.main(), 0)
        self.assertEqual(matches.call_count, 1)
        self.assertEqual(extract.call_count, 2)


if __name__ == "__main__":
    unittest.main()
