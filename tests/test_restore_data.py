"""发布副本的数据恢复必须保持标签、图片哈希和证据绑定。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "restore_data", Path(__file__).resolve().parents[1] / "scripts/restore_data.py")
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


class RestoreDataTests(unittest.TestCase):
    def prepare(self, root, body=b"image bytes", expected=None):
        source = root / "source"
        source.mkdir()
        samples = [{"sample_id": "example-0001", "image_path": "/missing/original.jpg",
                    "image_sha256": expected or hashlib.sha256(body).hexdigest(),
                    "ground_truth": "original", "captions": "fixed caption",
                    "archive": "origin/sample.zip", "zip_member": "nested/image.jpg"}]
        (source / "samples.json").write_text(json.dumps(samples))
        (source / "manifest.json").write_text(json.dumps({"count": 1}))
        (source / "reverse-evidence.json").write_text(json.dumps({
            "input_sha256": restore.sha256(source / "samples.json"),
            "records": {"example-0001": {"captions": "fixed caption"}}}))
        archive = root / "downloads/origin/sample.zip"
        archive.parent.mkdir(parents=True)
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("nested/image.jpg", body)
        return source, samples

    def test_restore_preserves_selection_and_rebinds_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, original = self.prepare(root)
            original_bytes = (source / "samples.json").read_bytes()
            result = restore.restore_dataset(source, root / "restored", root / "downloads")
            rows = json.loads((root / "restored/samples.json").read_text())
            self.assertEqual(rows[0]["ground_truth"], original[0]["ground_truth"])
            self.assertEqual(Path(rows[0]["image_path"]).read_bytes(), b"image bytes")
            evidence = json.loads((root / "restored/reverse-evidence.json").read_text())
            self.assertEqual(evidence["input_sha256"], result["samples_sha256"])
            self.assertEqual((source / "samples.json").read_bytes(), original_bytes)
            # 重跑可直接利用已校验图片，即使原 ZIP 已不可用。
            (root / "downloads/origin/sample.zip").unlink()
            restore.restore_dataset(source, root / "restored", root / "downloads")

    def test_hash_mismatch_does_not_publish_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self.prepare(root, expected="0" * 64)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                restore.restore_dataset(source, root / "restored", root / "downloads")
            self.assertFalse((root / "restored/samples.json").exists())

    def test_in_place_requires_explicit_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self.prepare(root)
            with self.assertRaisesRegex(ValueError, "in-place"):
                restore.restore_dataset(source, source, root / "downloads")

    def test_wrong_evidence_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self.prepare(root)
            (source / "reverse-evidence.json").write_text('{"input_sha256":"wrong"}')
            with self.assertRaisesRegex(ValueError, "证据哈希"):
                restore.restore_dataset(source, root / "restored", root / "downloads")


if __name__ == "__main__":
    unittest.main()
