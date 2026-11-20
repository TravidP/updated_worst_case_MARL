import gzip
import json
import tempfile
import unittest
from pathlib import Path

from experiments.curves import read_curves, point
from experiments.metrics import RewardMean
from experiments.storage import append_record


class CurveTests(unittest.TestCase):
    def test_mean_immediate_reward_uses_all_samples_after_clipping(self):
        meter=RewardMean()
        meter.add([-300.,-10.],[-2.,-.1])
        meter.add([-10.,0.],[-.1,0.])
        stats=meter.summary()
        self.assertEqual(stats['samples'],4)
        self.assertEqual(stats['raw'],-80.)
        self.assertAlmostEqual(stats['learner'],-.55)
        with self.assertRaises(ValueError):meter.add([1,2],[1])

    def test_validation_reward_does_not_replace_queue_selection(self):
        record=dict(episode=100,score=3.,rollouts=[
            dict(rewards={'controller':dict(raw=-5.,learner=-.05,samples=3)}),
            dict(rewards={'controller':dict(raw=-1.,learner=-.01,samples=1)})])
        p=point(record,'validation')
        self.assertEqual(p['score'],3.)
        self.assertEqual(p['rewards']['controller']['raw'],-4.)
        self.assertAlmostEqual(p['rewards']['controller']['learner'],-.04)

    def test_incremental_gzip_tail_and_rollback_before_first_new_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);log=d/'episodes.jsonl.gz'
            append_record(d/'attempts.jsonl',dict(attempt='a',resume_episode=0))
            append_record(log,dict(episode=1,attempt='a',mean_queue=4))
            self.assertEqual(read_curves(d)['training'][0]['queue'],4)
            member=gzip.compress((json.dumps(dict(episode=2,attempt='a',mean_queue=6))+'\n').encode())
            with log.open('ab') as f:f.write(member[:1])
            self.assertEqual(len(read_curves(d)['training']),1)
            with log.open('ab') as f:f.write(member[1:12])
            self.assertEqual(len(read_curves(d)['training']),1)
            with log.open('ab') as f:f.write(member[12:])
            self.assertEqual(len(read_curves(d)['training']),2)
            append_record(d/'attempts.jsonl',dict(attempt='b',resume_episode=1))
            self.assertEqual(len(read_curves(d)['training']),1)
            append_record(log,dict(episode=2,attempt='b',mean_queue=8))
            self.assertEqual([r['queue'] for r in read_curves(d)['training']],[4,8])
