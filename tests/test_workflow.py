"""Known-answer checks for integration, scenario generation and reporting."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.protocol import settings, scenario_root, output_path, allowed_output
from experiments.scenarios import definitions, block_rows
from experiments.demand import profiles
from experiments.reporting import statistics, seed_interval, summarize, lane_window


class Workflow(unittest.TestCase):
    def test_single_seed_study_counts(self):
        p=settings()
        self.assertEqual(p['training_seeds'],[101])
        parents=len(p['networks'])*len(p['controllers'])*len(p['training_seeds'])
        self.assertEqual(parents,8)
        self.assertEqual(p['counts'],dict(parents=8,offline_wce=8,continuations=40,evaluation_rollouts=9200))
        self.assertEqual(parents*len(p['methods'])*23*len(p['arrival_seeds']),9200)

    def test_compatibility(self):
        from revision.runner import run as old
        from experiments.runner import run
        from revision.core import QueueMetric as a
        from experiments.core import QueueMetric as b
        self.assertIs(old,run);self.assertIs(a,b)

    def test_scenario_counts_and_rates(self):
        self.assertEqual(settings()['version'], 7)
        for network,total in [('grid',3000),('monaco',2383.3333)]:
            groups=profiles(network)
            for split,count in [('seen',11),('validation',6),('test',12)]:
                defs=definitions(network,split);self.assertEqual(len(defs),count)
                for d in defs:
                    cursor=0
                    for b in d['blocks']:
                        self.assertEqual(b['start'],cursor);cursor+=b['duration']
                        rows=block_rows(groups,d,b)
                        self.assertAlmostEqual(sum(r[2] for r in rows),total*b.get('multiplier',1),places=5)
                    self.assertEqual(cursor,3600)
            self.assertEqual([s['generation_seed'] for s in definitions(network,'test')],list(range(41001,41013)))

    def test_v7_temporal_scenarios_are_frozen_random_seen_profiles(self):
        for network in settings()['networks']:
            seen = {group['name'] for group in profiles(network)}
            first = [s for s in definitions(network, 'test') if s['family'] == 'temporal']
            second = [s for s in definitions(network, 'test') if s['family'] == 'temporal']
            self.assertEqual(first, second)
            self.assertEqual([s['id'] for s in first], ['switch_300', 'switch_900', 'switch_1200'])
            self.assertEqual([len(s['blocks']) for s in first], [12, 4, 3])
            for scenario in first:
                self.assertEqual(scenario['protocol_version'], 7)
                self.assertEqual(scenario['temporal_policy'], 'seeded_random_seen_profiles')
                selected = [block['profile'] for block in scenario['blocks']]
                self.assertTrue(set(selected) <= seen)
                self.assertTrue(all(a != b for a, b in zip(selected, selected[1:])))
            self.assertIn('protocol_v7', str(scenario_root(network, 'test')))

    def test_statistics_and_missing_cells(self):
        s=seed_interval([3]);self.assertEqual(s['mean'],3);self.assertIsNone(s['ci95']);self.assertIsNone(s['sd']);self.assertTrue(s['complete'])
        self.assertFalse(seed_interval([])['complete'])
        self.assertIsNone(seed_interval([1,2])['ci95'])
        rows=[]
        for method,offset in [('baseline',2),('online_wce',0)]:
            for seed in settings()['training_seeds']:
                for scenario in range(12):
                    for i,arrival in enumerate(settings()['arrival_seeds']):
                        rows.append(dict(network='grid',controller='ia2c',method=method,seed=seed,split='test',
                                         scenario=str(scenario),family='fixture',pilot=False,arrival_seed=arrival,mean_queue=seed/101+offset))
        _,seeds,agg,paired=summarize(rows)
        result=next(r for r in paired if r['network']=='grid' and r['controller']=='ia2c' and r['comparator']=='baseline' and r['split']=='test')
        self.assertEqual(result['mean'],-2);self.assertIsNone(result['ci95']);self.assertIsNone(result['sd']);self.assertTrue(result['complete'])
        self.assertFalse(next(r for r in agg if r['method']=='fixed_wce')['complete'])

    def test_heatmap_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'lanes.npz';q=np.zeros((3600,2));q[1200:2400]=[12,3]
            np.savez(str(p),time=np.arange(1,3601),lanes=['a','b'],queue=q)
            lanes,value,total=lane_window(p)
            self.assertEqual(lanes,['a','b']);np.testing.assert_array_equal(value,[12,3]);self.assertEqual(total,15)

    def test_output_and_launch_validation(self):
        from experiments.dashboard import preflight
        self.assertTrue(allowed_output(output_path('continue','monaco','ppo',9001,'online_wce',True)))
        self.assertFalse(allowed_output(Path('/tmp/arbitrary')))
        good=dict(stage='parent',network='grid',controller='iqll',seed=9001,pilot=True,steps=160)
        self.assertIn('experiment',preflight(good)['argv'])
        for bad in [dict(good,command='rm'),dict(good,seed=101),dict(good,steps=-1),dict(good,pilot=False,seed=101),dict(good,pilot=False,seed=202),dict(good,network='fake')]:
            with self.assertRaises((ValueError,SystemExit)):preflight(bad)

    def test_checkpoint_origin_identity(self):
        from experiments.checkpoint import checkpoint_identity
        from experiments.core import digest, write_json
        with tempfile.TemporaryDirectory() as tmp:
            origin=Path(tmp);checkpoint=origin/'checkpoint';checkpoint.mkdir()
            m=dict(network='grid',controller='iqll',seed=9001,stage='continue',method='online_wce')
            m['manifest_hash']=digest(m);write_json(origin/'manifest.json',m)
            self.assertEqual(checkpoint_identity(checkpoint,'grid','iqll',9001)['method'],'online_wce')
            with self.assertRaises(ValueError):checkpoint_identity(checkpoint,'grid','iqll',9002)
            with self.assertRaises(ValueError):checkpoint_identity(checkpoint,'monaco','iqll',9001)

    def test_finished_timer_and_duplicate_job(self):
        from experiments.dashboard import Jobs
        from types import SimpleNamespace
        from unittest.mock import patch
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'console.log').write_text('')
            job=Jobs.__new__(Jobs);job.lock=threading.Lock()
            job.current=dict(id='fixture',path=str(p),output=str(p/'output'),started=100.,status='running')
            job.process=SimpleNamespace(poll=lambda:0)
            with patch('experiments.dashboard.time.time',return_value=110.):
                self.assertEqual(job.status()['elapsed_seconds'],10.)
            with patch('experiments.dashboard.time.time',return_value=210.):
                self.assertEqual(job.status()['elapsed_seconds'],10.)
            job.process=SimpleNamespace(poll=lambda:None)
            with self.assertRaisesRegex(ValueError,'Another job is active'):job.start({})

    def test_evaluation_progress_is_not_a_training_curve(self):
        from experiments.reporting import report
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);evaluation=base/'evaluation';evaluation.mkdir()
            (evaluation/'manifest.json').write_text(json.dumps(dict(stage='evaluate',network='grid',controller='iqll',method='baseline')))
            (evaluation/'progress.jsonl').write_text(json.dumps(dict(learning_steps=2800,mean_queue=15))+'\n')
            report([evaluation],base/'report')
            data=json.loads((base/'report/dashboard.json').read_text())
            self.assertEqual(data['curves'],[])


if __name__=='__main__':unittest.main(verbosity=2)
