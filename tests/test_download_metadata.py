import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "download_metadata", Path(__file__).resolve().parents[1] / "scripts/download_metadata.py")
metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metadata)


class DownloadMetadataTests(unittest.TestCase):
    def test_fixed_author_rows_keep_labels_captions_and_image_identity(self):
        body = json.dumps([{"text": "claim", "fake_cls": "orig", "captions": "caption"}]).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample-index.json").write_text(json.dumps({"count": 1, "records": [{
                "author_index": 0, "sample_id": "fixed-0000", "archive": "images.zip",
                "zip_member": "original.jpg", "image_sha256": "known-image-hash"}]}))
            with patch.dict(metadata.SOURCES, {"fixture": ("test.json", hashlib.sha256(body).hexdigest())}):
                samples_bytes, evidence = metadata.build_metadata("fixture", root, body)
            row = json.loads(samples_bytes)[0]
            self.assertEqual(row["ground_truth"], "original")
            self.assertEqual(row["captions"], "caption")
            self.assertEqual(row["image_sha256"], "known-image-hash")
            self.assertEqual(evidence["input_sha256"], hashlib.sha256(samples_bytes).hexdigest())
            self.assertEqual(evidence["summary"]["ready"], 1)

    def test_changed_author_download_is_rejected_before_parsing(self):
        with self.assertRaisesRegex(ValueError, "哈希"):
            metadata.build_metadata("DGM4-author1000", Path("/missing"), b"changed")


if __name__ == "__main__":
    unittest.main()
