import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.protocol import settings
from experiments.selection import Selection
from experiments.metrics import EpisodeMetrics
from experiments.validation import load_manifest
from experiments.release import package,archive,inventory
from experiments.core import QueueMetric

class ProtocolTests(unittest.TestCase):
    def test_plateau_and_meaningful_improvements(self):
        p=dict(settings(),stopping_rule='patience_v1',patience=5);p['min_episodes']['parent']=0
        s=Selection('parent',p);s.add(0,100,'0');s.add(100,90,'1')
        for e in range(200,700,100):s.add(e,90,'x'+str(e))
        self.assertTrue(s.candidate())
        self.assertEqual(s.terminal(),'early_stopped')
        self.assertEqual(s.diagnostics()['no_improvement_evals'],5)
        restored=Selection('parent',p,s.state());self.assertEqual(restored.state(),s.state())
        s.add(700,89.95,'tiny');self.assertEqual(s.anchor_episode,100)
        self.assertEqual(s.best['checkpoint'],'tiny')
        s.add(800,89,'big');self.assertEqual(s.anchor_episode,800)
        self.assertEqual(s.best['score'],89)
        with self.assertRaises(ValueError):s.add(800,88,'duplicate')

    def test_continuation_best_through_stopping(self):
        for improves in (True,False):
            selection=Selection('continue',dict(settings(),stopping_rule='patience_v1',patience=5))
            selection.add(0,100,'parent')
            for episode in range(100,5100,100):
                score=100-min(episode,1200)/100 if improves else 100
                selection.add(episode,score,str(episode))
                if selection.terminal():break
            self.assertEqual(episode,1700 if improves else 1000)
            self.assertEqual(selection.best['checkpoint'],'1200' if improves else 'parent')

    def test_directions_no_learning_and_continuation(self):
        p=dict(settings(),stopping_rule='patience_v1',patience=5);p['min_episodes']={k:0 for k in p['min_episodes']}
        for stage in ['parent','wce','continue']:
            s=Selection(stage,p)
            for e in range(0,700,100):s.add(e,10,'x'+str(e))
            self.assertTrue(s.candidate())
            self.assertEqual(s.terminal(),'early_stopped' if stage=='continue' else 'no_learning')
        s=Selection('wce',p);s.add(0,10,'a');s.add(100,12,'b');s.add(200,11,'c')
        self.assertEqual(s.best['checkpoint'],'b')

    def test_patience_ignores_noise_and_restores_counter(self):
        p=dict(settings(),stopping_rule='patience_v1',patience=5);p['min_episodes']['parent']=500
        s=Selection('parent',p)
        s.add(0,100,'initial');s.add(100,90,'best')
        for e,score in [(200,110),(300,95),(400,130),(500,120)]:s.add(e,score,str(e))
        self.assertIsNone(s.terminal())
        restored=Selection('parent',p,s.state())
        restored.add(600,140,'last')
        self.assertEqual(restored.terminal(),'early_stopped')
        self.assertEqual(restored.best['checkpoint'],'best')
        self.assertGreater(restored.diagnostics()['current_best_gap'],0)
        restored.add(700,89.95,'tiny')
        self.assertEqual(restored.no_improvement_evals,6)
        restored.add(800,89.,'meaningful')
        self.assertEqual(restored.no_improvement_evals,0)
        self.assertIsNone(restored.terminal())

    def test_minimum_budget_and_evaluation_count(self):
        p=dict(settings(),stopping_rule='patience_v1',patience=5);s=Selection('parent',p)
        for e in range(0,601,100):s.add(e,100,str(e))
        self.assertIsNone(s.terminal())  # Patience elapsed, but below minimum 1000.
        s.add(1000,100,'minimum');self.assertEqual(s.terminal(),'no_learning')
        p['min_episodes']['parent']=0
        s=Selection('parent',p);s.add(0,100,'initial');s.add(1000,90,'better')
        s.add(2000,90,'late')
        self.assertFalse(s.candidate())  # One observation, not ten inferred intervals.
        p['max_episodes']['parent']=2000
        self.assertEqual(s.terminal(),'max_budget_reached')
        self.assertEqual(s.best['checkpoint'],'better')

    def test_wce_patience_direction_and_old_state_rejection(self):
        p=dict(settings(),stopping_rule='patience_v1',patience=5);p['min_episodes']['wce']=0
        s=Selection('wce',p);s.add(0,10,'initial');s.add(100,12,'best')
        for e,score in [(200,11),(300,8),(400,10),(500,7),(600,9)]:s.add(e,score,str(e))
        self.assertEqual(s.terminal(),'early_stopped')
        self.assertEqual(s.best['checkpoint'],'best')
        old=s.state();del old['no_improvement_evals']
        with self.assertRaisesRegex(ValueError,'incompatible'):
            Selection('wce',p,old)

    def test_full_horizon_and_first_patience_best_survive_resume(self):
        for stage in ('parent','wce','continue'):
            protocol=settings()
            protocol['min_episodes'][stage]=0
            protocol['max_episodes'][stage]=1200
            selection=Selection(stage,protocol)
            for episode in range(0,1201,100):
                score=10 if episode<1100 else (20 if stage=='wce' else 5)
                selection.add(episode,score,str(episode))
                if episode==1000:
                    selection=Selection(stage,protocol,selection.state())
                self.assertEqual(selection.terminal(),'max_budget_reached' if episode==1200 else None)
            self.assertEqual(selection.best['episode'],1100)
            if stage=='continue':
                self.assertIsNone(selection.early_stop)
            else:
                self.assertEqual(selection.early_stop,dict(trigger_episode=1000,episode=0,score=10,checkpoint='0'))
        selection=Selection('parent',protocol)
        for episode in range(0,1201,100):
            selection.add(episode,100-episode/100,str(episode))
        self.assertIsNone(selection.early_stop)

    def test_streamed_statistics_and_bounds(self):
        m=EpisodeMetrics(['a','b'],audit=True)
        for t in range(1,6601):m.add(t,[t%3,t%7],active=3,speed_sum=9,pending=2,completed=int(t%7==0))
        a=np.asarray(m.trace);r=m.summary(6600,True)
        self.assertAlmostEqual(r['mean_queue'],a.sum(1).mean())
        self.assertAlmostEqual(r['queue_variance'],a.sum(1).var())
        np.testing.assert_allclose(r['lane_windows'][1],a[1200:2400].mean(0))
        self.assertEqual(len(m.recent),600)
        with self.assertRaises(ValueError):m.summary(6601)
        with self.assertRaises(ValueError):m.add(6602,[0,0])
        compact=EpisodeMetrics(['a']);self.assertIsNone(compact.trace)

    def test_reward_reads_neighborhood_coefficient(self):
        q=QueueMetric({'a':['x'],'b':['y']},{'a':['b'],'b':['a']},.5)
        np.testing.assert_array_equal(q.rewards([[4,2]],'ma2c'),[-5,-4])

    def test_manifest_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'m.json';path.write_text(json.dumps(dict(version='8.1',network='grid',split='validation',frozen=True,scenarios=[])))
            with self.assertRaises(ValueError):load_manifest(path,'grid',True)
            with self.assertRaises(ValueError):load_manifest(None,'grid',True)

    def test_release_and_archive(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);src=d/'src';src.mkdir();(src/'input').write_text('example')
            package('inputs',src,d/'release.tar.gz')
            self.assertTrue((d/'release.tar.gz.manifest.json').exists())
            archive(src,d/'archive')
            self.assertEqual((d/'archive/input').read_text(),'example')
            self.assertTrue((src/'input').exists())
            (src/'link').symlink_to('/tmp')
            with self.assertRaises(ValueError):inventory(src)

if __name__=='__main__':unittest.main()
