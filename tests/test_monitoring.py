"""Reward time alignment and native TensorBoard record integrity."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from experiments.core import QueueMetric
from experiments.telemetry import Telemetry
from tf_compat import tf


def read_scalars(path):
    values = {}
    for event_file in Path(path).glob('events.out.tfevents.*'):
        for event in tf.train.summary_iterator(str(event_file)):
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    values.setdefault(value.tag, []).append((event.step, value.simple_value))
    return values


def audit_run(path):
    path = Path(path)
    expected = []
    for file in sorted(path.glob('episode_*.controls.jsonl')):
        for line in file.read_text().splitlines():
            row = json.loads(line)
            if row.get('learning'):
                expected.append((row['learning_steps'], float(np.mean(row['learner_rewards']))))
    actual = read_scalars(path/'tensorboard').get('train/reward_by_learning_step', [])
    assert len(actual) == len(expected), (len(actual),len(expected))
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-5)
    for folder in path.glob('monitoring/round_*'):
        summary = json.loads((folder/'summary.json').read_text())
        if summary.get('reused_from'):
            continue
        for child in folder.glob('rollout_*'):
            rows = [json.loads(line) for line in (child/'rollout.jsonl').read_text().splitlines()]
            q = np.load(str(child/'rollout.npz'))
            w = np.load(str(child/'waiting.npz'))
            assert len(rows)==600 and q['queue'].shape[0]==600
            np.testing.assert_array_equal(q['time'],np.arange(1,601))
            np.testing.assert_array_equal(q['lanes'],w['lanes'])
            np.testing.assert_allclose(q['queue'].sum(axis=1),[r['queue'] for r in rows])
            np.testing.assert_allclose(w['current_wait'].sum(axis=1),[r['current_wait_sum_vehicle_seconds'] for r in rows])
            assert (child/'timeseries.csv').read_text().count('\n')==601
        assert (folder/'queue_waiting.png').is_file()
    return True


class Monitoring(unittest.TestCase):
    def test_endpoint_and_unchanged_wce(self):
        m = QueueMetric({'a':['a'], 'b':['b']},{'a':['b'],'b':['a']})
        q=np.array([[0,0],[1,0],[2,0],[3,0],[12,3.]])
        np.testing.assert_equal(m.rewards(q,'ia2c'),[-15,-15])
        np.testing.assert_allclose(m.rewards(q,'ma2c'),[-14.7,-13.8])
        self.assertAlmostEqual(m.wce(np.tile([12,3],(600,1))),15.)

    def test_native_raw_reward_and_partial_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            t=Telemetry(directory)
            for step,value in [(1000001,-15),(1000002,-30)]:
                t.scalars({'train/reward_by_learning_step':value},step)
            rows=[dict(queue=15,pending=1,active=20,inserted=0,completed=0,teleports=0,collisions=0) for _ in range(10)]
            env=SimpleNamespace(rows=rows,controller_rows=[dict(learner_rewards=[-15,-15])]*2,scheduled=21)
            ctrl=SimpleNamespace(learning_steps=1000002,backward_calls=1)
            t.episode(env,ctrl,'continue',1,2,.2,0);t.close()
            scalars=read_scalars(Path(directory)/'tensorboard')
            self.assertEqual(scalars['train/reward_by_learning_step'],[(1000001,-15),(1000002,-30)])
            self.assertIn('train/partial_episode/mean_total_queue',scalars)
            self.assertNotIn('train/episode/mean_total_queue',scalars)


if __name__=='__main__':unittest.main()
