"""Matrix launch verification using synthetic JSON bundles; never train or start SUMO."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.core import ROOT, digest, file_hash

spec = importlib.util.spec_from_file_location('training_matrix', str(ROOT / 'scripts/training/_matrix.py'))
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class FollowupLaunchers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        clean = {k: v for k, v in os.environ.items() if not k.startswith('CBWCE_')}
        clean.update(CBWCE_PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE='1')
        env = patch.dict(os.environ, clean, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def bundle(self, key, stage, pilot, parent_hash=None):
        n, c, s = key
        run = self.path / n / c / str(s) / stage
        checkpoint = run / 'checkpoint_fixture'
        checkpoint.mkdir(parents=True)
        origin = dict(network=n, controller=c, seed=s, stage=stage, pilot=pilot)
        origin['manifest_hash'] = digest(origin)
        (run / 'manifest.json').write_text(json.dumps(origin))
        # This sentinel exercises hashing only. It is not a loadable training checkpoint.
        (checkpoint / 'fixture.txt').write_text('synthetic test fixture')
        signatures = {'controller': 'fixture'}
        if stage == 'wce':
            signatures['wce'] = 'fixture'
        bundle = dict(version=1, signatures=signatures,
                      files={'fixture.txt': file_hash(checkpoint / 'fixture.txt')},
                      parents={'controller': {'hash': parent_hash}} if stage == 'wce' else {})
        bundle['hash'] = digest(bundle)
        (checkpoint / 'manifest.json').write_text(json.dumps(bundle))
        result = dict(status='complete', checkpoint=str(checkpoint),
                      manifest_hash=origin['manifest_hash'], learning_steps=160 if pilot else 1000000,
                      stage_simulation_steps=(2640 if pilot else 660000) if stage == 'wce' else
                      (160 if pilot else 1000000))
        (run / 'result.json').write_text(json.dumps(result))
        return str(run / 'result.json'), bundle['hash']

    def mapping(self, seeds=(9001,), pilot=True):
        rows = []
        for n in ('grid', 'monaco'):
            for c in ('ia2c', 'ma2c', 'iqll', 'ppo'):
                for s in seeds:
                    key = n, c, s
                    parent, h = self.bundle(key, 'parent', pilot)
                    wce, _ = self.bundle(key, 'wce', pilot, h)
                    rows.append(dict(network=n, controller=c, seed=s, parent_result=parent, wce_result=wce))
        path = self.path / 'selections.json'
        path.write_text(json.dumps(dict(version=1, runs=rows)))
        return path, rows

    def test_real_shell_previews(self):
        path, rows = self.mapping()
        scripts = ['09_all_wce.sh', '10_all_baseline.sh', '11_all_random_group.sh',
                   '12_all_domain_randomization.sh', '13_all_fixed_wce.sh', '14_all_online_wce.sh']
        for script in scripts:
            result = subprocess.run(['bash', str(ROOT / 'scripts/training' / script),
                                     '--mode', 'pilot', '--seed', '9001', '--checkpoints', str(path),
                                     '--visualization', '--dry-run'], cwd='/tmp',
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count('COMMAND:'), 8)
            self.assertNotIn('START /', result.stdout)
            for row in rows:
                self.assertIn(str(Path(row['parent_result']).parent / 'checkpoint_fixture'), result.stdout)

    def test_all_methods_publication_order_and_stop(self):
        seeds = (101,)
        path, rows = self.mapping(seeds, pilot=False)
        args = matrix.parser().parse_args(['--stage', 'all_continuations', '--mode', 'publication',
                                           '--seed', '101', '--checkpoints', str(path), '--gate', 'selected gate.json'])
        with patch.object(matrix.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')) as preview:
            with patch.object(matrix, 'launch_jobs', return_value=0) as launch, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(matrix.run(args), 0)
                self.assertEqual(preview.call_count, 40)
                self.assertEqual(launch.call_count, 10)
                commands = [cmd for call in launch.call_args_list for cmd in call[0][0]]
                self.assertEqual(len(commands), 40)
                for index, cmd in enumerate(commands):
                    method = matrix.METHODS[index // 8]
                    row = rows[index % 8]
                    self.assertTrue(cmd[1].endswith(matrix.SCRIPTS[method]))
                    self.assertEqual(cmd[cmd.index('--seed') + 1], str(row['seed']))
                    self.assertEqual(cmd[cmd.index('--network') + 1], row['network'])
                    self.assertEqual(cmd[cmd.index('--controller') + 1], row['controller'])
                    self.assertEqual(cmd[cmd.index('--parent') + 1],
                                     str(Path(row['parent_result']).parent / 'checkpoint_fixture'))
                    self.assertEqual('--wce' in cmd, method in ('fixed_wce', 'online_wce'))
                    self.assertNotIn('--steps', cmd)
                    self.assertNotIn('--episodes', cmd)
            with patch.object(matrix, 'launch_jobs', side_effect=[0, 7]) as launch, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(matrix.run(args), 7)
                self.assertEqual(launch.call_count, 2)

    def test_eight_workers_combine_grid_and_monaco(self):
        path, _ = self.mapping((101,), pilot=False)
        args = matrix.parser().parse_args(['--stage', 'random_group', '--mode', 'publication',
            '--seed', '101', '--workers', '8', '--checkpoints', str(path), '--gate', 'gate.json'])
        with patch.object(matrix.subprocess, 'run',
                          return_value=subprocess.CompletedProcess([], 0, '', '')) as preview, \
             patch.object(matrix, 'launch_jobs', return_value=0) as launch, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(matrix.run(args), 0)
            self.assertEqual(preview.call_count, 8)
            self.assertEqual(launch.call_count, 1)
            batch = launch.call_args[0][0]
            self.assertEqual(len(batch), 8)
            self.assertEqual({cmd[cmd.index('--network') + 1] for cmd in batch}, {'grid', 'monaco'})

    def test_interrupt_waits_for_owned_job_cleanup(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            old = signal.getsignal(sig)
            with patch.object(matrix.subprocess, 'Popen') as factory, \
                    patch.object(matrix.os, 'killpg') as kill, patch.object(matrix.time, 'sleep') as sleep:
                child = factory.return_value
                child.pid = 12345
                child.poll.side_effect = [None, None, 0]

                def interrupted(_):
                    signal.getsignal(sig)(sig, None)

                sleep.side_effect = interrupted
                self.assertEqual(matrix.launch_job(['harmless-fixture']), 128 + sig)
                kill.assert_called_once_with(12345, signal.SIGINT)
                self.assertTrue(factory.call_args[1]['start_new_session'])
                self.assertEqual(signal.getsignal(sig), old)

    def test_invalid_last_selection_prevents_all_launches(self):
        path, rows = self.mapping()
        args = matrix.parser().parse_args(['--stage', 'online_wce', '--checkpoints', str(path), '--dry-run'])
        last = Path(rows[-1]['wce_result'])
        good = last.read_text()
        for change in [dict(status='failed'), dict(manifest_hash='wrong')]:
            value = json.loads(good)
            value.update(change)
            last.write_text(json.dumps(value))
            with patch.object(matrix.subprocess, 'run') as child:
                with self.assertRaises(ValueError):
                    matrix.run(args)
                child.assert_not_called()
        last.write_text(good)
        bundle_path = last.parent / 'checkpoint_fixture/manifest.json'
        bundle = json.loads(bundle_path.read_text())
        bundle['parents']['controller']['hash'] = 'wrong parent'
        bundle.pop('hash')
        bundle['hash'] = digest(bundle)
        bundle_path.write_text(json.dumps(bundle))
        with self.assertRaisesRegex(ValueError, 'different parent'):
            matrix.run(args)
        path.write_text(json.dumps(dict(version=1, runs=rows + [rows[0]])))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            matrix.run(args)

    def test_template_and_inherited_resume(self):
        path = self.path / 'nested/selection.json'
        flags = ['--stage', 'wce', '--mode', 'publication', '--seed', '101', '--write-template', str(path)]
        with contextlib.redirect_stdout(io.StringIO()):
            matrix.run(matrix.parser().parse_args(flags))
        self.assertEqual(len(json.loads(path.read_text())['runs']), 8)
        with self.assertRaises(FileExistsError):
            matrix.run(matrix.parser().parse_args(flags))
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            matrix.run(matrix.parser().parse_args(flags + ['--dry-run']))
        self.assertEqual(path.read_bytes(), before)
        with patch.dict(os.environ, {'CBWCE_RESUME': '/wrong/shared/checkpoint'}):
            with self.assertRaisesRegex(ValueError, 'Unset CBWCE_RESUME'):
                matrix.run(matrix.parser().parse_args(['--stage', 'wce', '--checkpoints', str(path)]))


if __name__ == '__main__':
    unittest.main()
