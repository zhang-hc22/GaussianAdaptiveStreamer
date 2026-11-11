"""Failure-safe retention of the YUV payload, without CUDA or a network."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quality_store as store
from run_quality import finish_quality, quality_environment, quality_jobs


class RunFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rel = "train/NTHU/train/user3_train/test_x1/rtt20/webrtc-push/rep1"
        self.directory = self.root / "runs" / self.rel
        (self.directory / "frames").mkdir(parents=True)
        self.meta = self.directory / "run.json"
        self.meta.write_text(json.dumps(dict(status="ok", viewpoints=24, args=dict(mode="webrtc"))))
        (self.directory / "frames.ndjson").write_text("".join(json.dumps(dict(i=i, standin_i=0) if i == 12 else dict(i=i, dec=20+i)) + "\n" for i in range(24)))
        self.yuv = [self.directory / "frames/000000.yuv", self.directory / "frames/000012_standin.yuv"]
        for p in self.yuv:
            p.write_bytes(b"original payload")
        (self.directory / "sender.log").write_text("controller and encoder records\n")
        (self.directory / "frames/000000.jpg").write_bytes(b"preserve image")
        self.csv = self.root / "metrics" / (self.rel + ".csv")
        self.cfg = dict(server=dict(metric_stride=12, resolution="1280x720", paper_metrics=False))
        self.run = dict(rel=self.rel, mode="webrtc")

    def write_metrics(self, count=2, nan=False):
        rows = [dict(i=i, standin=int(i == 12), **{key: (float("nan") if nan and key == "vmaf" else 1.0) for key in store.METRICS})
                for i in (0, 12)][:count]
        store.atomic_csv(self.csv, rows, ["i", "standin", *store.METRICS])

    def assert_payload_retained(self):
        self.assertTrue(all(p.exists() for p in self.yuv))

    def test_complete_csv_preserves_data_and_removes_only_yuv(self):
        self.write_metrics()
        protected = [self.meta, self.csv, self.directory / "frames.ndjson", self.directory / "sender.log", self.directory / "frames/000000.jpg"]
        before = {p: p.read_bytes() for p in protected}
        unlink = Path.unlink

        def checked_unlink(path, *args, **kwargs):
            if path.suffix == ".yuv":
                checkpoint = json.loads((self.directory / "quality_complete.json").read_text())
                self.assertEqual(checkpoint["state"], "quality_complete")
                self.assertEqual(next(iter(checkpoint["quality"].values()))["sha256"], store.sha(self.csv))
            return unlink(path, *args, **kwargs)

        with mock.patch.object(Path, "unlink", checked_unlink):
            result = store.complete_and_remove_yuv(self.root, self.rel, 12)
        self.assertEqual(result["deleted_files"], 2)
        self.assertEqual(before, {p: p.read_bytes() for p in protected})
        self.assertTrue(json.loads((self.directory / "quality_complete.json").read_text())["yuv_removed"])

    def test_partial_csv_retains_every_yuv(self):
        self.write_metrics(count=1)
        with self.assertRaises(store.QualityError):
            store.complete_and_remove_yuv(self.root, self.rel, 12)
        self.assert_payload_retained()

    def test_incomplete_viewpoint_log_prevents_cleanup(self):
        self.write_metrics()
        (self.directory / "frames.ndjson").write_text(json.dumps(dict(i=0, dec=20)) + "\n")
        with self.assertRaises(store.QualityError):
            store.complete_and_remove_yuv(self.root, self.rel, 12)
        self.assert_payload_retained()

    def test_nonfinite_csv_retains_every_yuv(self):
        self.write_metrics(nan=True)
        with self.assertRaises(store.QualityError):
            store.complete_and_remove_yuv(self.root, self.rel, 12)
        self.assert_payload_retained()

    def test_duplicate_sample_is_rejected(self):
        self.write_metrics()
        with self.csv.open("a") as fh:
            fh.write("0,0,1,1,1,1,1,1\n")
        with self.assertRaises(store.QualityError):
            store.complete_and_remove_yuv(self.root, self.rel, 12)
        self.assert_payload_retained()

    def test_interrupted_cleanup_resumes_without_recomputing_quality(self):
        self.write_metrics()
        unlink = Path.unlink
        calls = []

        def interrupted(path, *args, **kwargs):
            if path.suffix == ".yuv":
                calls.append(path)
                if len(calls) == 2:
                    raise KeyboardInterrupt()
            return unlink(path, *args, **kwargs)

        with mock.patch.object(Path, "unlink", interrupted), self.assertRaises(KeyboardInterrupt):
            store.complete_and_remove_yuv(self.root, self.rel, 12)
        self.assertEqual(sum(p.exists() for p in self.yuv), 1)
        invoke = mock.Mock(side_effect=AssertionError("must reuse complete metrics"))
        finish_quality(self.cfg, str(self.root), self.run, invoke, lambda _: None)
        invoke.assert_not_called()
        self.assertFalse(any(p.exists() for p in self.yuv))

    def test_completed_cleanup_is_idempotent(self):
        self.write_metrics()
        store.complete_and_remove_yuv(self.root, self.rel, 12)
        invoke = mock.Mock(side_effect=AssertionError("no payload must be rescored"))
        result = finish_quality(self.cfg, str(self.root), self.run, invoke, lambda _: None)
        self.assertTrue(result["cached"])
        invoke.assert_not_called()

    def test_symlink_prevents_all_deletion(self):
        self.write_metrics()
        external = self.root / "original.yuv"
        external.write_bytes(b"preserve outside")
        self.yuv[1].unlink()
        self.yuv[1].symlink_to(external)
        with self.assertRaises(store.QualityError):
            store.complete_and_remove_yuv(self.root, self.rel, 12)
        self.assert_payload_retained()
        self.assertTrue(external.exists())

    def test_success_exit_with_partial_metrics_cannot_delete_payload(self):
        def invoke(cmd):
            if "exp/metrics.py" in cmd:
                self.write_metrics(count=1)
            return 0, "scorer exited normally"
        with self.assertRaises(store.QualityError):
            finish_quality(self.cfg, str(self.root), self.run, invoke, lambda _: None)
        self.assert_payload_retained()

    def test_scorer_failure_retains_payload(self):
        with self.assertRaises(store.QualityError):
            finish_quality(self.cfg, str(self.root), self.run, lambda _: (1, "interrupted scorer"), lambda _: None)
        self.assert_payload_retained()

    def test_disk_guard_fires_before_scorer_or_new_files(self):
        low = type("Space", (), dict(f_bavail=1, f_frsize=2**30))()
        invoke = mock.Mock()
        with mock.patch.object(store.os, "statvfs", return_value=low), self.assertRaises(store.QualityError):
            finish_quality(self.cfg, str(self.root), self.run, invoke, lambda _: None)
        invoke.assert_not_called()
        self.assert_payload_retained()

    def test_existing_csv_is_not_replaced_by_failed_atomic_write(self):
        self.write_metrics()
        before = self.csv.read_bytes()
        def broken_rows():
            yield {"i": 0}
            raise RuntimeError("interrupted write")
        with self.assertRaises(RuntimeError):
            store.atomic_csv(self.csv, broken_rows(), ["i"])
        self.assertEqual(self.csv.read_bytes(), before)


class QualityBudget(unittest.TestCase):
    def test_cpu_budget_is_capped(self):
        with mock.patch("os.cpu_count", return_value=16):
            env = quality_environment(dict(server={}))
            self.assertEqual(quality_jobs(dict(server={})), 4)
            self.assertEqual(quality_jobs(dict(server=dict(quality_jobs=8))), 8)
        with mock.patch("os.cpu_count", return_value=2):
            self.assertEqual(quality_jobs(dict(server={})), 2)
        self.assertEqual(env["OMP_NUM_THREADS"], "4")
        self.assertEqual(env["TIGAS_QUALITY_JOBS"], "4")


if __name__ == "__main__":
    unittest.main()
