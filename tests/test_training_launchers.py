"""Validate shell command generation without launching training or SUMO."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from experiments.core import ROOT
from experiments.runner import parser
from experiments.protocol import settings, output_root

SCRIPTS = [
    ('01_parent.sh', 'parent', 'baseline'),
    ('02_wce.sh', 'wce', 'baseline'),
    ('03_baseline.sh', 'continue', 'baseline'),
    ('04_random_group.sh', 'continue', 'random_group'),
    ('05_domain_randomization.sh', 'continue', 'domain_randomization'),
    ('06_fixed_wce.sh', 'continue', 'fixed_wce'),
    ('07_online_wce.sh', 'continue', 'online_wce'),
]


class TrainingLaunchers(unittest.TestCase):
    def invoke(self, script, flags, extra=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith('CBWCE_')}
        env.update(CBWCE_PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE='1')
        env.update(extra or {})
        return subprocess.run(['bash', str(ROOT / 'scripts/training' / script)] + flags,
                              cwd='/tmp', env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True)

    def arguments(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        command = next(line[len('COMMAND: '):] for line in result.stdout.splitlines()
                       if line.startswith('COMMAND: '))
        tokens = shlex.split(command)
        self.assertEqual(tokens[1:4], ['-u', 'main.py', 'experiment'])
        return parser().parse_args(['--stage', tokens[4]] + tokens[5:])

    def test_bash_syntax(self):
        for path in (ROOT / 'scripts/training').glob('*.sh'):
            result = subprocess.run(['bash', '-n', str(path)])
            self.assertEqual(result.returncode, 0, str(path))

    def test_stage_matrix_and_defaults(self):
        for network in settings()['networks']:
            for controller in settings()['controllers']:
                for mode in ('pilot', 'publication'):
                    for script, stage, method in SCRIPTS:
                        with self.subTest(network=network, controller=controller, mode=mode, script=script):
                            args = self.arguments(self.invoke(script, [
                                '--dry-run', '--network', network, '--controller', controller,
                                '--mode', mode, '--parent', '/selected/common parent/checkpoint',
                                '--wce', '/selected/common wce/checkpoint', '--gate', '/selected/gate.json']))
                            self.assertEqual((args.stage, args.method), (stage, method))
                            self.assertEqual(args.seed, 9001 if mode == 'pilot' else 101)
                            self.assertEqual(args.pilot, mode == 'pilot')
                            self.assertEqual(args.visualization, mode == 'pilot')
                            self.assertFalse(Path(args.output).exists())
                            self.assertIn(output_root(stage, network, method), Path(args.output).parents)
                            self.assertEqual(args.parent, None if stage == 'parent' else '/selected/common parent/checkpoint')
                            self.assertEqual(args.wce, '/selected/common wce/checkpoint' if method in ('fixed_wce', 'online_wce') else None)
                            if mode == 'pilot':
                                self.assertEqual(args.checkpoint_every, 1)
                                self.assertEqual(args.episodes, 2 if stage == 'wce' else None)
                                self.assertEqual(args.steps, None if stage == 'wce' else 160 if stage == 'parent' else 2640)
                            else:
                                self.assertIsNone(args.steps)
                                self.assertIsNone(args.episodes)
                                self.assertEqual(args.gate, '/selected/gate.json')

    def test_overrides_resume_and_literal_paths(self):
        literal = '/selected/checkpoint $(not-a-shell-command); with spaces'
        args = self.arguments(self.invoke('07_online_wce.sh', [
            '--dry-run', '--parent', literal, '--wce', '/selected/wce',
            '--resume', literal, '--steps', '3960', '--no-visualization', '--seed', '9002'],
            {'CBWCE_SEED': '9001', 'CBWCE_VISUALIZATION': 'on'}))
        self.assertEqual(args.parent, literal)
        self.assertEqual(args.resume, literal)
        self.assertEqual(args.steps, 3960)
        self.assertEqual(args.seed, 9002)
        self.assertFalse(args.visualization)

    def test_invalid_selections_stop_without_launch(self):
        cases = [
            ('01_parent.sh', ['--network', 'bad']),
            ('01_parent.sh', ['--mode', 'bad']),
            ('01_parent.sh', ['--controller', 'bad']),
            ('01_parent.sh', ['--mode', 'pilot', '--seed', '101']),
            ('01_parent.sh', ['--seed', '4294967296']),
            ('01_parent.sh', ['--mode', 'publication']),
            ('01_parent.sh', ['--mode', 'publication', '--seed', '202', '--gate', 'gate']),
            ('01_parent.sh', ['--mode', 'publication', '--gate', 'gate', '--steps', '160']),
            ('01_parent.sh', ['--steps', '0']),
            ('02_wce.sh', []),
            ('02_wce.sh', ['--parent', 'parent', '--episodes', '0']),
            ('03_baseline.sh', ['--parent', 'parent', '--steps', '160']),
            ('06_fixed_wce.sh', ['--parent', 'parent']),
            ('01_parent.sh', ['--seed']),
            ('01_parent.sh', ['--arbitrary-shell', 'false']),
        ]
        for script, flags in cases:
            with self.subTest(script=script, flags=flags):
                result = self.invoke(script, ['--dry-run'] + flags)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('START /', result.stdout)

    def test_parallel_parent_matrix(self):
        for mode, seeds in [('pilot', [9001]), ('publication', [101])]:
            flags = ['--dry-run', '--mode', mode, '--gate', '/selected/gate.json']
            result = self.invoke('08_all_parents.sh', flags)
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [shlex.split(line[len('COMMAND: '):])
                        for line in result.stdout.splitlines() if line.startswith('COMMAND: ')]
            expected = [(n, c, s) for n in ('grid', 'monaco')
                        for c in ('ia2c', 'ma2c', 'iqll', 'ppo') for s in seeds]
            observed = []
            for tokens in commands:
                args = parser().parse_args(['--stage', tokens[4]] + tokens[5:])
                self.assertEqual(args.stage, 'parent')
                self.assertIsNone(args.resume)
                self.assertFalse(Path(args.output).exists())
                observed.append((args.network, args.controller, args.seed))
            self.assertCountEqual(observed, expected)
            self.assertEqual(result.stdout.count('PARALLEL BATCH / 并行批次'), 2)
            self.assertEqual(result.stdout.count('workers=4'), 2)
        for flags, env in [(['--all-seeds'], {}),
                           (['--mode', 'publication', '--all-seeds', '--seed', '101'], {}),
                           ([], {'CBWCE_RESUME': '/selected/checkpoint'}),
                           (['--network', 'grid'], {})]:
            result = self.invoke('08_all_parents.sh', ['--dry-run'] + flags, env)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('COMMAND:', result.stdout)

    def test_parallel_parent_rejects_invalid_workers(self):
        for workers in ('0', '9'):
            result = self.invoke('08_all_parents.sh', ['--dry-run', '--workers', workers])
            self.assertEqual(result.returncode, 2)
            self.assertNotIn('COMMAND:', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
