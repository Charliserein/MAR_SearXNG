from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_experiment as runner


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "experiment"
        self.image = self.root / "image.jpg"
        self.image.write_bytes(b"input-image")

    def rows(self, name):
        labels = ["original"] * 6 + ["textual_veracity_distortion"] * 6 + [
            "visual_veracity_distortion"] * 2 + ["mismatch"] * 6
        return [{"sample_id": f"{name}-{i:03}", "id": i, "text": f"Claim {i}",
                 "ground_truth": label, "fake_cls": label, "captions": "",
                 "image_path": str(self.image),
                 "image_sha256": hashlib.sha256(self.image.read_bytes()).hexdigest()}
                for i, label in enumerate(labels)]

    def setup_sources(self):
        for name in runner.DATASETS:
            rows = self.rows(name)
            root = self.root / "data" / name
            runner.write_json(root / "samples.json", rows)
            runner.write_json(root / "manifest.json", {"dataset": name})

    def test_largest_remainder_is_deterministic_and_full_selection_keeps_order(self):
        rows = self.rows("test")
        selected, quotas = runner.stratified_sample(rows, 7, 20261020, "test")
        again, _ = runner.stratified_sample(rows, 7, 20261020, "test")
        self.assertEqual(selected, again)
        self.assertEqual(sum(quotas.values()), 7)
        self.assertEqual(Counter(r["ground_truth"] for r in selected), quotas)
        full, counts = runner.stratified_sample(rows, len(rows), 20261020, "test")
        self.assertEqual([r["sample_id"] for r in full], [r["sample_id"] for r in rows])
        self.assertEqual(counts, Counter(r["ground_truth"] for r in rows))

    def test_prepare_interleaves_and_preserves_pending_evidence(self):
        self.setup_sources()
        with patch.object(runner, "ROOT", self.root):
            plan = runner.prepare_experiment(self.output, per_dataset=10)
            again = runner.prepare_experiment(self.output, per_dataset=10)
        self.assertEqual(plan, again)
        rows = runner.read_json(plan["combined_input"])
        self.assertEqual(len(rows), 20)
        self.assertEqual([r["source_dataset"] for r in rows[:4]],
                         list(runner.DATASETS) * 2)
        manifest = runner.read_json(Path(plan["combined_input"]).with_name("reverse-evidence.json"))
        self.assertEqual(manifest["input_sha256"], runner.digest(plan["combined_input"]))
        self.assertEqual(manifest["summary"]["statuses"], {"pending": 20})
        self.assertEqual(manifest["summary"]["ready"], 0)
        self.assertEqual(set(manifest["records"]), {r["sample_id"] for r in rows})

    def test_changed_frozen_inputs_are_refused(self):
        self.setup_sources()
        with patch.object(runner, "ROOT", self.root):
            plan = runner.prepare_experiment(self.output, per_dataset=10)
        Path(plan["combined_input"]).write_text("[]")
        with self.assertRaisesRegex(ValueError, "冻结输入"):
            runner.verify_inputs(self.output, plan)

    def test_snapshot_change_is_refused(self):
        self.output.mkdir()
        runner.write_json(self.output / "experiment.json", {"test": True})
        plan = {"input_files": {}}
        with patch.object(runner, "code_snapshot", return_value={}):
            runner.freeze_or_verify(self.output, plan, 8)
            runner.freeze_or_verify(self.output, plan, 8)
            with self.assertRaisesRegex(ValueError, "禁止混合续跑"):
                runner.freeze_or_verify(self.output, plan, 4)
            with patch.dict("os.environ", {"MAR_MODEL": "different-model"}):
                with self.assertRaisesRegex(ValueError, "禁止混合续跑"):
                    runner.freeze_or_verify(self.output, plan, 8)

    def test_launch_is_detached_and_does_not_expose_credentials(self):
        self.output.mkdir()
        runner.write_json(self.output / "experiment.json", {"input_files": {}})
        runner.write_json(self.output / "run-snapshot.json", {})
        with patch.object(runner, "freeze_or_verify", return_value={}), \
             patch.object(runner, "running", return_value=False), \
             patch.object(runner, "process_token", return_value="1234"), \
             patch.object(runner.subprocess, "Popen") as launch:
            launch.return_value.pid = 456
            result = runner.start(self.output, 8)
        argv = launch.call_args.args[0]
        self.assertEqual(argv[:2], ["nohup", "setsid"])
        self.assertIn("_supervise", argv)
        self.assertEqual(launch.call_args.kwargs["stdin"], runner.subprocess.DEVNULL)
        self.assertEqual(launch.call_args.kwargs["stderr"], runner.subprocess.STDOUT)
        self.assertEqual(result["pid"], 456)

    def test_evaluator_requires_web_success_but_accepts_missing_image_evidence(self):
        command = runner.evaluation_command({"combined_input": "/input/samples.json",
                                             "evaluation_output": "/output"}, 8, True)
        self.assertIn("--strict-web", command)
        self.assertIn("--no-strict-evidence", command)
        self.assertIn("--retry-errors", command)
        self.assertEqual(command[command.index("--workers") + 1], "8")
        self.assertEqual(command[command.index("--rounds") + 1], "3")

    def test_orphan_evaluator_with_wrong_process_group_is_not_signaled(self):
        self.output.mkdir()
        runner.write_json(self.output / "evaluator.json", {"pid": 123, "process_token": "token"})
        with patch.object(runner, "process_token", return_value="token"), \
             patch.object(runner.os, "getpgid", return_value=999), \
             patch.object(runner, "running", return_value=False), \
             patch.object(runner.os, "killpg") as signal_process:
            result = runner.stop(self.output)
        signal_process.assert_not_called()
        self.assertEqual(result["status"], "not_running")

    def test_benchmark_uses_overlapping_wall_clock_not_sum_of_sample_seconds(self):
        ids = [str(i) for i in range(4)]
        plan = {"first4_sample_ids": ids, "first20_sample_ids": ids + ["later"]}
        records = [{"sample_id": str(i), "status": "ok", "seconds": 10 + i,
                    "started_at": "2026-10-20T10:00:00+00:00",
                    "finished_at": f"2026-10-20T10:00:{10+i:02}+00:00"} for i in range(4)]
        benchmark = runner.benchmarks(plan, records)
        self.assertEqual(benchmark["first4"]["wall_seconds"], 13)
        self.assertEqual(benchmark["first4"]["ok"], 4)
        self.assertFalse(benchmark["first20"]["available"])


if __name__ == "__main__":
    unittest.main()
