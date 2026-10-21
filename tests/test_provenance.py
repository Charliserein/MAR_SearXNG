import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "MAR_framework"))
from evidence_manifest import attach_evidence, text_digest
from retrieval.provenance import evidence_context, evidence_kind


class ProvenanceTests(unittest.TestCase):
    def test_manifest_keeps_method_and_caption_bytes_separate(self):
        sample = {"sample_id": "example", "image_sha256": "image", "captions": ""}
        caption = "[Re1] Original caption\n"
        manifest = {
            "schema_version": 1, "input_sha256": "input",
            "records": {"example": {"image_sha256": "image", "captions": caption,
                                   "captions_sha256": text_digest(caption),
                                   "status": "ok", "source": "searxng_images"}},
        }
        before = copy.deepcopy(manifest)
        row = attach_evidence([sample], manifest, "input", strict=True)[0]
        self.assertEqual(row["captions"], caption)
        self.assertEqual(evidence_kind(row), "text_to_image_with_visual_verification")
        context = evidence_context(row["captions"], evidence_kind(row))
        self.assertIn("not native reverse-image search", context)
        self.assertIn("depends on the claim", context)
        self.assertTrue(context.endswith(caption))
        self.assertEqual(manifest, before)
        self.assertEqual(sample["captions"], "")

    def test_author_annotations_and_unknown_inputs_are_not_fresh_search(self):
        author = evidence_context("caption", evidence_kind({"source": "author_annotation"}))
        self.assertIn("authors' released dataset", author)
        self.assertIn("No fresh image search", author)
        self.assertIn("unspecified", evidence_context("caption"))
        self.assertIn("native reverse-image search",
                      evidence_context("caption", evidence_kind({"source": "serper_lens"})))

    def test_empty_evidence_is_not_a_claim_of_no_prior_source(self):
        self.assertIn("Do not infer", evidence_context("", "native_reverse_image"))

    def test_batch_passes_method_without_ground_truth_to_analyzer(self):
        from evaluate_batch import run_sample
        analyzer = Mock()
        analyzer.analyze.return_value = {"label": "original"}
        sample = {
            "sample_id": "example", "id": 1, "ground_truth": "mismatch",
            "fake_cls": "text_swap", "text": "Claim", "image_path": "/tmp/a.jpg",
            "image_sha256": "image", "captions": "caption",
            "image_evidence_kind": "text_to_image_with_visual_verification",
        }
        record = run_sample(analyzer, sample)
        analyzer.analyze.assert_called_once_with(
            "Claim", "/tmp/a.jpg", "caption",
            evidence_kind="text_to_image_with_visual_verification")
        self.assertEqual(record["image_evidence_kind"], sample["image_evidence_kind"])
        self.assertEqual(record["status"], "ok")

    def test_analyzer_preserves_provenance_in_graph_state(self):
        from NewsAnalyzer import NewsAnalyzer
        analyzer = NewsAnalyzer.__new__(NewsAnalyzer)
        analyzer.encode_image = Mock(return_value="image-bytes")
        analyzer.app = Mock()
        analyzer.app.invoke.side_effect = lambda state: state
        result = analyzer.analyze("claim", "/tmp/a.jpg", "raw caption",
                                  evidence_kind="text_to_image_with_visual_verification")
        self.assertEqual(result["captions"], "raw caption")
        self.assertEqual(result["image_evidence_kind"], "text_to_image_with_visual_verification")
        self.assertNotIn("ground_truth", result)


if __name__ == "__main__":
    unittest.main()
