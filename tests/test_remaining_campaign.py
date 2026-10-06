"""Focused tests for the resumable publication campaign; never start training."""
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from experiments.core import ROOT

spec = importlib.util.spec_from_file_location(
    'remaining_campaign', str(ROOT / 'scripts/training/_remaining_campaign.py'))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class RemainingCampaign(unittest.TestCase):
    def test_classify_selects_highest_valid_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'publication_fixture'
            run.mkdir()
            manifest = dict(network='grid', controller='ia2c', seed=101,
                stage='continue', method='random_group', pilot=False,
                monitor_every=50, monitor_rollouts=3,
                parents={'controller': {'hash': 'parent-hash'}})
            manifest['manifest_hash'] = campaign.digest(manifest)
            (run / 'manifest.json').write_text(json.dumps(manifest))
            for step in (13200, 39600, 26400):
                (run / ('checkpoint_{:09d}'.format(step))).mkdir()
            with patch.object(campaign, 'task_root', return_value=root), \
                 patch.object(campaign, 'inspect_checkpoint', return_value={
                     'hash': 'parent-hash', 'signatures': {'controller': 'signature'}}), \
                 patch.object(campaign, 'valid_resume',
                              side_effect=lambda path, *_: int(Path(path).name.split('_')[-1])):
                state = campaign.classify(('grid', 'ia2c', 101), 'random_group', '/parent', None)
            self.assertEqual(state['status'], 'resume')
            self.assertEqual(state['step'], 39600)

    def test_command_includes_exact_resume_and_wce(self):
        job = {'method': 'online_wce', 'key': ('monaco', 'ppo', 101),
               'parent': '/parent', 'wce': '/wce',
               'state': {'status': 'resume', 'checkpoint': '/resume', 'step': 13200}}
        command = campaign.command_for(job, '/gate', True)
        self.assertEqual(command[command.index('--resume') + 1], '/resume')
        self.assertEqual(command[command.index('--parent') + 1], '/parent')
        self.assertEqual(command[command.index('--wce') + 1], '/wce')
        self.assertIn('--dry-run', command)
        self.assertIn('--no-visualization', command)

    def test_cleanup_only_removes_old_non_tenth_raw_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'output_coevolution/revised'
            run = root / 'grid/ia2c/seed_101/random_group/publication_fixture'
            runtime = run / 'runtime'
            runtime.mkdir(parents=True)
            names = ['episode_0001.npz', 'episode_0010.npz', 'episode_metrics.jsonl']
            for name in names:
                (run / name).write_text('x')
            (runtime / 'trips_1.xml').write_text('x')
            (runtime / 'trips_10.xml').write_text('x')
            old = time.time() - 7200
            for path in run.rglob('*'):
                if path.is_file():
                    os.utime(str(path), (old, old))
            log = io.StringIO()
            with patch.object(campaign, 'ROOT', Path(directory)):
                campaign.cleanup_once(log)
            self.assertFalse((run / 'episode_0001.npz').exists())
            self.assertFalse((runtime / 'trips_1.xml').exists())
            self.assertTrue((run / 'episode_0010.npz').exists())
            self.assertTrue((runtime / 'trips_10.xml').exists())
            self.assertTrue((run / 'episode_metrics.jsonl').exists())

    def test_dry_run_does_not_generate_gate_or_cleanup(self):
        jobs = [{'method': 'random_group', 'key': ('grid', 'ia2c', 101),
                 'parent': '/parent', 'wce': None, 'state': {'status': 'fresh', 'step': 0}}]
        completed = __import__('subprocess').CompletedProcess([], 0, '', '')
        with patch.object(campaign, 'resolve_jobs', return_value=jobs), \
             patch.object(campaign, 'resource_preflight'), \
             patch.object(campaign, 'matching_gate', return_value=(None, [])), \
             patch.object(campaign.subprocess, 'run', return_value=completed), \
             patch.object(campaign, 'generate_gate') as generate, \
             patch.object(campaign, 'cleanup_once') as cleanup, \
             patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(campaign.main(['--dry-run']), 0)
            generate.assert_not_called()
            cleanup.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
