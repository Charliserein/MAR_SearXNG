import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

from scripts.report_evaluation import generate_report, generate_initial_report, run_timing


class ReportEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)
        (self.output / "records").mkdir()
        self.samples = [
            {"sample_id": "sample-1", "ground_truth": "original", "fake_cls": "original",
             "text": "A sample news claim.", "image_size": [100, 80]},
            {"sample_id": "sample-2", "ground_truth": "mismatch", "fake_cls": "mismatch",
             "text": "Another sample news claim.", "image_size": [120, 90]},
        ]
        self.write("samples.json", self.samples)
        self.write("config.json", {
            "model": "Qwen3.8-27B", "rounds": 1, "workers": 1,
            "temperature": 0, "base_url": "http://127.0.0.1:19888/v1",
            "embedding_device": "cpu", "input_sha256": "test-samples-hash",
            "search_provider": "searxng", "cache_mode": "offline",
        })

    def write(self, path, value):
        (self.output / path).write_text(json.dumps(value, ensure_ascii=False))

    def records(self):
        for sample in self.samples:
            self.write("records/" + sample["sample_id"] + ".json", {
                "sample_id": sample["sample_id"], "ground_truth": sample["ground_truth"],
                "prediction": sample["ground_truth"], "status": "ok",
                "text": sample["text"], "attempt": 1, "seconds": 1.0,
                "started_at": "2026-10-20T12:00:00+00:00",
                "finished_at": "2026-10-20T12:00:01+00:00",
                "result": {"current_round": 1, "short_term_memories": ["summary"],
                           "retrieved_knowledge_only": []},
            })

    def report(self, manifest, *, complete=True):
        self.write("dataset-manifest.json", manifest)
        if complete:
            self.records()
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = generate_report(self.output)
        return metrics, (self.output / "report.html").read_text()

    def test_mmfakebench_optional_manifest_fields_and_source(self):
        metrics, page = self.report({
            "dataset": "Averdgr/MAR_KDD MMFakeBench processed test",
            "split": "author_test", "official_split": "val",
            "source_metadata_sha256": "mm-source-metadata",
            "source_url": "https://github.com/Averdgr/MAR_KDD/blob/main/data_processing/MMFakeBench/test_set.json",
        })
        self.assertEqual(metrics["valid"], 2)
        self.assertEqual(metrics["dataset"]["metadata_sha256"], "mm-source-metadata")
        self.assertEqual(metrics["subclasses"]["original"]["selected"], 1)
        self.assertIn("MMFakeBench 官方数据", page)
        self.assertNotIn("DGM4", page)
        self.assertNotIn("固定种子", page)
        self.assertIn("清单未记录随机种子", page)
        self.assertIn("官方原始划分：val", page)

    def test_mixed_smoke_report_does_not_claim_full_evaluation(self):
        metrics, page = self.report({
            "dataset": "DGM4 and MMFakeBench author tests",
            "count": 2,
            "purpose": "integration_smoke_only_not_accuracy_estimation",
            "selection": "first_available_caption_per_dataset",
            "source_samples": ["/data/DGM4/samples.json", "/data/MMFakeBench/samples.json"],
        })
        self.assertTrue(metrics["dataset"]["diagnostic_only"])
        self.assertEqual(metrics["selection"]["selection"], "first_available_caption_per_dataset")
        self.assertIn("不用于估计模型准确率", page)
        self.assertIn("DGM4 官方数据", page)
        self.assertIn("MMFakeBench 官方数据", page)
        self.assertIn("按配置循环 1 轮", page)
        self.assertIn("配置的样本并发数为 1", page)
        self.assertNotIn("重新混合划分", page)
        self.assertNotIn("本次使用项目支持的公开 DGM4", page)
        self.assertNotIn("循环三轮", page)
        self.assertNotIn("两条样本之间并行", page)

    def test_subclass_counts_use_actual_selected_samples(self):
        metrics, page = self.report({
            "dataset": "DGM4 author test subset", "seed": 42,
            "subclass_counts": {"original": 300, "mismatch": 300},
            "metadata_sha256": {"test": "official-metadata"},
        })
        self.assertEqual(metrics["subclasses"]["original"]["selected"], 1)
        self.assertEqual(metrics["subclasses"]["mismatch"]["selected"], 1)
        self.assertEqual(metrics["subclasses"]["original"]["valid"], 1)
        self.assertIn("固定种子 42", page)
        self.assertIn("official-metadata", page)
        self.assertNotIn("MMFakeBench 官方数据", page)

    def test_unidentified_dataset_is_not_assumed_to_be_dgm4(self):
        metrics, page = self.report({"dataset": "Other dataset"}, complete=False)
        self.assertEqual(metrics["valid"], 0)
        self.assertEqual(metrics["pending"], 2)
        self.assertEqual(metrics["dataset"]["datasets"], [])
        self.assertIn("Other dataset", page)
        self.assertNotIn("DGM4", page)
        self.assertNotIn("MMFakeBench", page)

    def test_stopped_report_with_no_success_does_not_require_class_metrics(self):
        self.write("status.json", {"status": "stopped_by_user", "interrupted": [],
                                   "not_started": 2})
        metrics, page = self.report({"dataset": "MMFakeBench"}, complete=False)
        self.assertEqual(metrics["valid"], 0)
        self.assertIn("不代表原定 2 条结果", page)
        self.assertNotIn("原定100条", page)

    def test_zero_results_word_is_valid_and_metrics_are_unavailable(self):
        metrics, page = self.report({"dataset": "MMFakeBench"}, complete=False)
        self.assertIsNone(metrics["four_class"]["accuracy"])
        self.assertIsNone(metrics["by_dataset"]["MMFakeBench"]["binary"]["macro_f1"])
        with zipfile.ZipFile(self.output / "report.docx") as archive:
            self.assertIsNone(archive.testzip())
            for name in archive.namelist():
                ET.fromstring(archive.read(name))
            document = ET.fromstring(archive.read("word/document.xml"))
            content = "".join(document.itertext())
        self.assertIn("尚无成功预测时显示“—”", content)
        self.assertIn("完整三轮", content)
        self.assertIn("Macro-Precision", content)
        self.assertIn("report.docx", page)

    def test_parallel_timing_uses_supervisor_wall_clock(self):
        records = [
            {"sample_id": str(i), "status": "ok", "seconds": 120,
             "started_at": "2026-10-20T12:00:00+00:00",
             "finished_at": "2026-10-20T12:02:00+00:00",
             "result": {"current_round": 3, "short_term_memories": ["a", "b", "c"]}}
            for i in range(4)
        ]
        timing = run_timing([{}] * 8, records, {"workers": 4, "rounds": 3}, [],
                            {}, {"status": "running", "elapsed_seconds": 180}, {})
        self.assertEqual(timing["session_seconds"], 180)
        self.assertEqual(timing["effective_seconds_per_successful_sample"], 45)
        self.assertEqual(timing["full_three_round_sample_wall_seconds"]["mean"], 120)
        self.assertEqual(timing["successful_samples_per_hour"], 80)
        self.assertEqual(timing["eta_remaining_seconds"], 180)

    def test_initial_two_thousand_report_does_not_create_evaluation_config(self):
        source_dir = self.output / "inputs/combined"
        source_dir.mkdir(parents=True)
        samples = [
            {**self.samples[0], "sample_id": f"author-{prefix}-{index:04d}",
             "source_dataset": name}
            for name, prefix in (("DGM4", "dgm4"), ("MMFakeBench", "mmfakebench"))
            for index in range(1000)
        ]
        (source_dir / "samples.json").write_text(json.dumps(samples))
        (source_dir / "manifest.json").write_text(json.dumps({
            "dataset": "DGM4 + MMFakeBench author test lists", "count": 2000,
        }))
        self.write("experiment.json", {
            "combined_input": str(source_dir / "samples.json"),
            "evaluation_output": str(self.output / "evaluation"),
        })
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = generate_initial_report(self.output)
        self.assertEqual(metrics["selected"], 2000)
        self.assertEqual(metrics["pending"], 2000)
        self.assertEqual(metrics["selection_scope"], "full_author_test_lists")
        self.assertEqual(metrics["by_dataset"]["DGM4"]["selected"], 1000)
        self.assertEqual(metrics["by_dataset"]["MMFakeBench"]["selected"], 1000)
        self.assertIsNone(metrics["binary"]["accuracy"])
        self.assertFalse((self.output / "evaluation/config.json").exists())
        with zipfile.ZipFile(self.output / "report.docx") as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn("全量变体", archive.read("word/document.xml").decode())


if __name__ == "__main__":
    unittest.main()
