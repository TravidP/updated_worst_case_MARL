"""Scheduler simulations only: no SUMO, training, cleanup or gate writes."""
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('campaign', str(Path(__file__).with_name('_remaining_campaign.py')))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class QueueTests(unittest.TestCase):
    def simulate(self, fail=False):
        jobs = [dict(method='random_group' if i < 8 else 'domain_randomization',
                     key=('grid', str(i), 101), parent='/parent', wce=None,
                     state={'status': 'fresh', 'step': 0}) for i in range(16)]
        done, starts, events, counts = set(), [], [], {}
        control = c.RuntimeControl()
        peak = [0]
        class Child:
            def __init__(self, name):
                self.name = name
                self.stdout = io.StringIO('')
                self.left = 20 if name == c.label(jobs[0]) else 1
                self.finished = False
            def poll(self):
                self.left -= 1
                if self.left > 0:
                    return None
                if not self.finished:
                    events.append(('finish', self.name))
                    self.finished = True
                if fail and self.name == c.label(jobs[1]):
                    return 1
                done.add(self.name)
                return 0
        def popen(cmd, **kwargs):
            name = cmd[0]
            peak[0] = max(peak[0], len(control.children)+1)
            starts.append(cmd)
            counts[name] = counts.get(name, 0)+1
            events.append(('start', name))
            return Child(name)
        def classify(key, method, parent, wce):
            name = '/'.join((method, key[0], key[1]))
            if name in done:
                return {'status': 'complete', 'step': c.GOAL}
            return {'status': 'resume', 'step': counts.get(name, 0)*13200,
                    'checkpoint': '/checkpoint_' + str(counts.get(name, 0))}
        with patch.object(c, 'classify', side_effect=classify), patch.object(c, 'command_for',
             side_effect=lambda j,g: [c.label(j), j['state']['checkpoint']]), \
             patch.object(c.subprocess, 'Popen', side_effect=popen), patch.object(c.time, 'sleep'), \
             patch('sys.stdout', new=io.StringIO()):
            result = c.run_commands(jobs, '/gate', 8, control, retry=True)
        self.assertEqual(peak[0], 8)
        self.assertLess(events.index(('start', c.label(jobs[8]))), events.index(('finish', c.label(jobs[0]))))
        return jobs, starts, counts, result

    def test_cross_method_refill_without_waiting_for_slow_task(self):
        jobs, starts, counts, result = self.simulate()
        self.assertEqual(len(starts), 16)
        self.assertTrue(all(v == 0 for v in result.values()))

    def test_failure_retries_twice_from_latest_checkpoint(self):
        jobs, starts, counts, result = self.simulate(True)
        name = c.label(jobs[1])
        self.assertEqual(counts[name], 3)
        self.assertEqual([cmd[1] for cmd in starts if cmd[0] == name],
                         ['/checkpoint_0', '/checkpoint_1', '/checkpoint_2'])
        self.assertEqual(result[name], 1)
        self.assertEqual(sum(v == 0 for v in result.values()), 15)

    def test_complete_jobs_are_skipped_and_stop_does_not_launch(self):
        control = c.RuntimeControl()
        with patch.object(c.subprocess, 'Popen') as launch:
            self.assertEqual(c.run_commands([{'state': {'status': 'complete'}}], '/gate', 8, control, True), {})
            control.stop.set()
            self.assertEqual(c.run_commands([{'state': {'status': 'fresh'}}], '/gate', 8, control, True), {})
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
