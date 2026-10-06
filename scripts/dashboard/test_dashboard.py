"""Dashboard-only tests, deliberately outside gate-hashed tests/."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock

from dashboard import Collector, Lines, State


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v) + '\n' for v in values))


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.collector = Collector(self.root, self.root / 'selection.json')

    def test_incremental_partial_truncated_and_missing(self):
        path = self.root / 'progress.jsonl'
        reader = Lines()
        self.assertEqual(reader.get(path), [])
        path.write_text('{"n":1}\n{"n":')
        self.assertEqual(reader.get(path), [{'n': 1}])
        with path.open('a') as f:
            f.write('2}\n')
        self.assertEqual(reader.get(path), [{'n': 1}, {'n': 2}])
        self.assertEqual(reader.get(path), [{'n': 1}, {'n': 2}])
        path.write_text('{"n":3}\n')
        self.assertEqual(reader.get(path), [{'n': 3}])

    def test_monitor_reward_missing_failed_and_sd(self):
        folder = self.root / 'round_000050'
        obj = dict(status='complete', stage_simulation_steps=66000, episode=50,
                   learning_steps=1066000, metrics={'mean_total_queue': 90}, rollouts=[{}, {}, {}])
        put(folder / 'summary.json', obj)
        self.assertIsNone(self.collector.monitor(folder)['raw'])
        for i in range(1, 4):
            rows(folder / ('rollout_%02d' % i) / 'rollout.controls.jsonl',
                 [dict(time=t, raw_rewards=[-i, -i-2], learner_rewards=[-i/10, -(i+2)/10]) for t in range(5, 601, 5)])
        point = self.collector.monitor(folder)
        self.assertEqual(point['raw'], -3)
        self.assertEqual(point['raw_sd'], 1)
        self.assertAlmostEqual(point['learner'], -.3)
        failed = self.root / 'failed'
        put(failed / 'summary.json', dict(obj, status='failed'))
        self.assertIsNone(self.collector.monitor(failed))

    def test_resume_cutoff_dedup_and_partial_episode(self):
        old, new = self.root / 'old', self.root / 'new'
        old.mkdir(); new.mkdir()
        manifests = {old: dict(stage='continue'), new: dict(stage='continue', resume=str(old / 'checkpoint_000001320'))}
        rows(old / 'progress.jsonl', [dict(simulation_steps=s, episode=s//1320, goal=3960) for s in [1320, 2640]])
        rows(new / 'progress.jsonl', [dict(simulation_steps=1440, episode=2, goal=3960)])
        rows(old / 'episode_metrics.jsonl', [dict(stage_simulation_steps=1320, episode=1, complete_episode=True),
                                            dict(stage_simulation_steps=2640, episode=2, complete_episode=True)])
        rows(new / 'episode_metrics.jsonl', [dict(stage_simulation_steps=1440, episode=2, complete_episode=False)])
        branch = self.collector.branch(new, manifests, 3960, time.time())
        self.assertEqual(branch['step'], 1440)
        self.assertEqual(branch['completed_episodes'], 1)
        self.assertEqual(branch['partial_episodes'], [2])
        self.assertEqual(branch['process'], 'unknown')
        self.assertEqual(len(branch['training']), 2)
        # An unrelated branch has its own series, no implicit concatenation.
        separate = self.collector.branch(old, manifests, 3960, time.time())
        self.assertEqual(separate['step'], 2640)

    def test_complete_requires_result_budget(self):
        run = self.root / 'run'; run.mkdir()
        put(run / 'result.json', dict(status='complete', stage_simulation_steps=1320))
        self.assertNotEqual(self.collector.branch(run, {run: {'stage': 'continue'}}, 2640, time.time())['status'], 'complete')
        put(run / 'result.json', dict(status='complete', stage_simulation_steps=2640))
        self.assertEqual(self.collector.branch(run, {run: {'stage': 'continue'}}, 2640, time.time())['status'], 'complete')

    def test_wall_time_counts_attempts_not_cumulative_rows(self):
        old, new = self.root / 'old', self.root / 'new'
        old.mkdir(); new.mkdir()
        rows(old / 'progress.jsonl', [dict(simulation_steps=1320, wall_seconds=10), dict(simulation_steps=2640, wall_seconds=20)])
        put(old / 'result.json', dict(status='failed', wall_seconds=25))
        rows(new / 'progress.jsonl', [dict(simulation_steps=1440, wall_seconds=3), dict(simulation_steps=1560, wall_seconds=7)])
        manifests = {old: dict(stage='continue'), new: dict(stage='continue', resume=str(old / 'checkpoint_000001320'))}
        b = self.collector.branch(new, manifests, 3960, time.time())
        self.assertEqual(b['wall_seconds'], 32)  # 25 + 7, includes discarded work before restart
        self.assertTrue(b['wall_lower_bound'])
        self.assertEqual(len(b['wall_attempts']), 2)
        put(new / 'result.json', dict(status='complete', stage_simulation_steps=3960, wall_seconds=50))
        b = self.collector.branch(new, manifests, 3960, time.time())
        self.assertEqual(b['wall_seconds'], 75)
        self.assertFalse(b['wall_lower_bound'])

    def test_cleaned_ancestor_does_not_hide_completed_run(self):
        run = self.root / 'run'; run.mkdir()
        put(run / 'result.json', dict(status='complete', stage_simulation_steps=2640))
        rows(run / 'episode_metrics.jsonl', [dict(stage_simulation_steps=2640, episode=2, complete_episode=True)])
        m = {run: dict(stage='continue', resume=str(self.root / 'deleted' / 'checkpoint_000001320'))}
        b = self.collector.branch(run, m, 2640, time.time())
        self.assertEqual(b['status'], 'complete')
        self.assertEqual(b['completed_episodes'], 2)
        self.assertTrue(b['warnings'])
        self.assertTrue(b['wall_lower_bound'])

    def test_identity_parent_binding_and_seed(self):
        item = dict(network='grid', controller='ppo', seed=101)
        m = dict(item, stage='continue', method='fixed_wce', parents={'controller': {'hash': 'p'}, 'wce': {'hash': 'w'}})
        self.assertTrue(self.collector.identity(m, item, 'continue', 'fixed_wce', 'p', 'w'))
        self.assertFalse(self.collector.identity(m, item, 'continue', 'fixed_wce', 'wrong', 'w'))
        self.assertFalse(self.collector.identity(dict(m, pilot=True), item, 'continue', 'fixed_wce', 'p', 'w'))

    def test_refresh_failure_preserves_snapshot_and_schedule(self):
        collector = Mock()
        collector.scan.return_value = {'tasks': [], 'updated_at': 1}
        state = State(collector, self.root / 'cache.json', 1800)
        state.refresh()
        self.assertGreater(state.next_refresh, time.time() + 1790)
        collector.scan.side_effect = RuntimeError('test failure')
        state.refresh()
        self.assertEqual(state.data['updated_at'], 1)
        self.assertIn('test failure', state.error['message'])
        self.assertEqual(json.loads((self.root / 'cache.json').read_text())['updated_at'], 1)

    def test_background_loop_and_manual_event(self):
        collector = Mock()
        collector.scan.return_value = {'tasks': []}
        state = State(collector, self.root / 'cache.json', 1800)
        state.refresh()
        done = threading.Event()
        original = state.refresh
        def once():
            original()
            done.set()
        state.refresh = once
        threading.Thread(target=state.loop, daemon=True).start()
        state.event.set()
        self.assertTrue(done.wait(2))
        self.assertEqual(collector.scan.call_count, 2)

    def test_scheduled_refresh_without_browser(self):
        collector = Mock()
        collector.scan.return_value = {'tasks': []}
        state = State(collector, self.root / 'cache.json', 1800)
        state.refresh()
        state.next_refresh = time.time() + .03
        done = threading.Event()
        original = state.refresh
        def once():
            original()
            done.set()
        state.refresh = once
        threading.Thread(target=state.loop, daemon=True).start()
        self.assertTrue(done.wait(2))
        self.assertEqual(collector.scan.call_count, 2)


if __name__ == '__main__':
    unittest.main()
