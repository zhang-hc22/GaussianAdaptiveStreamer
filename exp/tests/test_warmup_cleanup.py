"""Warm-up payload cleanup and its QUIC and WebRTC entry points."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quality_store as store
import run_all as ra


class WarmupCleanup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.directory = self.out / 'warmup/webrtc/train'
        self.create_run(self.directory)

    def create_run(self, directory):
        (directory / 'frames').mkdir(parents=True, exist_ok=True)
        (directory / 'run.json').write_text(json.dumps(dict(status='ok', frames=36,
            first_frame_ms=12, webrtc=dict(times=dict(dc_open=100)))))
        (directory / 'frames.ndjson').write_text('{"i":0,"dec":1}\n')
        (directory / 'sender.log').write_text('keep diagnostic and controller data\n')
        (directory / 'frames/000000.jpg').write_bytes(b'keep image')
        for i in (0, 12, 24):
            (directory / f'frames/{i:06d}.yuv').write_bytes(b'warm-up frame')

    def payload(self):
        return sorted((self.directory / 'frames').glob('*.yuv'))

    def test_cleanup_is_checkpointed_and_preserves_every_non_yuv_file(self):
        protected = {p: p.read_bytes() for p in self.directory.rglob('*') if p.is_file() and p.suffix != '.yuv'}
        unlink = Path.unlink
        def checked_unlink(path, *args, **kwargs):
            if path.suffix == '.yuv':
                proof = json.loads((self.directory / 'warmup_cleanup.json').read_text())
                self.assertEqual(proof['run_sha256'], store.sha(self.directory / 'run.json'))
                self.assertFalse(proof['yuv_removed'])
            return unlink(path, *args, **kwargs)
        with mock.patch.object(Path, 'unlink', checked_unlink):
            result = store.remove_warmup_yuv(self.out, self.directory)
        self.assertEqual(result['deleted_files'], 3)
        self.assertFalse(self.payload())
        self.assertEqual(protected, {p: p.read_bytes() for p in protected})
        marker = json.loads((self.directory / 'warmup_cleanup.json').read_text())
        self.assertTrue(marker['yuv_removed'])
        self.assertEqual(marker['state'], 'warmup_complete')
        self.assertFalse((self.directory / 'quality_complete.json').exists())
        self.assertEqual(store.remove_warmup_yuv(self.out, self.directory)['deleted_files'], 0)

    def test_measured_directory_and_failed_warmup_are_rejected(self):
        measured = self.out / 'runs/train'
        self.create_run(measured)
        with self.assertRaises(store.QualityError):
            store.remove_warmup_yuv(self.out, measured)
        self.assertEqual(len(list((measured / 'frames').glob('*.yuv'))), 3)
        (self.directory / 'run.json').write_text('{"status":"failed"}')
        with self.assertRaises(store.QualityError):
            store.remove_warmup_yuv(self.out, self.directory)
        self.assertEqual(len(self.payload()), 3)

    def test_symlinked_warmup_parent_and_frames_are_rejected(self):
        alias = self.out / 'warmup/alias'
        alias.symlink_to(self.directory.parent, target_is_directory=True)
        with self.assertRaises(store.QualityError):
            store.remove_warmup_yuv(self.out, alias / 'train')
        frames = self.directory / 'frames'
        frames.rename(self.directory / 'original-frames')
        frames.symlink_to(self.directory / 'original-frames', target_is_directory=True)
        with self.assertRaises(store.QualityError):
            store.remove_warmup_yuv(self.out, self.directory)
        self.assertEqual(len(self.payload()), 3)

    def test_unexpected_file_or_payload_symlink_prevents_any_deletion(self):
        extra = self.directory / 'frames/reference.yuv'
        extra.write_bytes(b'preserve unexpected input')
        with self.assertRaises(store.QualityError):
            store.remove_warmup_yuv(self.out, self.directory)
        self.assertEqual(len(self.payload()), 4)
        extra.unlink()
        linked = self.directory / 'frames/000036.yuv'
        linked.symlink_to(self.directory / 'frames/000000.yuv')
        with self.assertRaises(store.QualityError):
            store.remove_warmup_yuv(self.out, self.directory)
        self.assertEqual(len(self.payload()), 4)

    def test_interrupted_cleanup_resumes_and_retains_original_manifest(self):
        unlink, count = Path.unlink, 0
        def interrupted(path, *args, **kwargs):
            nonlocal count
            if path.suffix == '.yuv':
                count += 1
                if count == 2:
                    raise KeyboardInterrupt()
            return unlink(path, *args, **kwargs)
        with mock.patch.object(Path, 'unlink', interrupted), self.assertRaises(KeyboardInterrupt):
            store.remove_warmup_yuv(self.out, self.directory)
        self.assertEqual(len(self.payload()), 2)
        self.assertEqual(store.remove_warmup_yuv(self.out, self.directory)['deleted_files'], 2)
        marker = json.loads((self.directory / 'warmup_cleanup.json').read_text())
        self.assertEqual(len(marker['yuv_files']), 3)
        self.assertTrue(marker['yuv_removed'])

    def test_payload_changed_after_checkpoint_is_retained(self):
        atomic = store.atomic_json
        def changed(path, record):
            atomic(path, record)
            self.payload()[0].write_bytes(b'new frame from another producer')
        with mock.patch.object(store, 'atomic_json', changed), self.assertRaises(store.QualityError):
            store.remove_warmup_yuv(self.out, self.directory)
        self.assertEqual(len(self.payload()), 3)
        with self.assertRaises(store.QualityError):
            store.remove_warmup_yuv(self.out, self.directory)

    def simulate_finished_warmup(self, cmd, **kwargs):
        self.create_run(Path(cmd[cmd.index('--out') + 1]))
        return 0, ''

    def test_webrtc_entry_point_cleans_before_returning(self):
        runs = [dict(scene='train', mode='webrtc', trajectory='fake.json')]
        with mock.patch.object(ra, 'call', self.simulate_finished_warmup), mock.patch.object(ra, 'log'):
            ra.warm_up_webrtc(str(self.out), runs, '1280x720', SimpleNamespace(socket='unused'))
        self.assertFalse(self.payload())

    def test_quic_entry_point_cleans_before_returning(self):
        runs = [dict(scene='train', mode='player', trajectory='fake.json')]
        with mock.patch.object(ra, 'call', self.simulate_finished_warmup), mock.patch.object(ra, 'log'):
            ra.warm_up(str(self.out), runs, '1280x720')
        self.assertFalse(list((self.out / 'warmup/train/frames').glob('*.yuv')))
        self.assertTrue((self.out / 'warmup/train/warmup_cleanup.json').exists())


if __name__ == '__main__':
    unittest.main()
