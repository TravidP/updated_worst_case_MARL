import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.metrics import EpisodeMetrics, METRICS_SCHEMA
from experiments.scenarios import test_manifest, check_test_manifest, block_rows
from experiments.demand import profiles
from experiments.tripinfo import summarize_trips, attach_trip_metrics
from experiments.reporting import report, sample_sd, bootstrap_mean_ci
from experiments.evaluation import checked_row, read_evaluation, evaluate_run
from experiments.storage import atomic_json, atomic_gzip_json
from experiments.core import digest, file_hash


class EvaluationTests(unittest.TestCase):
    def test_scenes_conserve_rates_and_switching(self):
        for network in ('grid','monaco'):
            m=test_manifest(network);check_test_manifest(m,network);groups=profiles(network)
            self.assertEqual(len(m['scenarios']),23)
            total=sum(r[2] for r in groups[0]['rows'])
            for s in m['scenarios']:
                self.assertEqual(sum(b['duration'] for b in s['blocks']),3600)
                for b in s['blocks']:
                    self.assertAlmostEqual(sum(r[2] for r in block_rows(groups,s,b)),total*b.get('multiplier',1),places=6)
                if s['family']=='temporal':
                    self.assertTrue(all(a['profile']!=b['profile'] for a,b in zip(s['blocks'],s['blocks'][1:])))
                if s['family']=='redistribution':
                    original={r[:2] for g in groups if g['name']=='Uniform' for r in g['rows'] if r[2]>0}
                    self.assertEqual({r[:2] for r in block_rows(groups,s,s['blocks'][0])},original)
            for field in ('profile_names','profiles_hash','generator_hash','test_schema'):
                bad=copy.deepcopy(m);bad.pop(field)
                with self.assertRaises(ValueError):check_test_manifest(bad,network)
            bad=copy.deepcopy(m);bad['scenarios'].pop()
            with self.assertRaises(ValueError):check_test_manifest(bad,network)

    def test_exact_windows_tail_and_bounded_detail(self):
        m=EpisodeMetrics(['a']);m.enable_detail(3600)
        for t in range(1,3601):m.add(t,[t],active=2,speed_sum=6,inserted=int(t==1),completed=int(t==3600))
        r=m.summary(3600,True)
        self.assertEqual(r['window_counts'],[1200]*3)
        self.assertEqual(r['window_mean_queue'],[600.5,1800.5,3000.5])
        self.assertEqual(r['last_600_s_slope'],1)
        self.assertAlmostEqual(r['p95_queue'],np.percentile(np.arange(1,3601),95))
        self.assertEqual(r['completion_rate'],1)
        self.assertEqual(r['mean_speed'],3)
        self.assertEqual(len(m.recent),600);self.assertIsNone(m.trace)
        self.assertEqual(m.detail_totals.nbytes,28800)

    def test_tripinfo_complete_excluded_empty_and_bad(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trips.xml'
            path.write_text('<tripinfos><tripinfo arrival="50" duration="40" waitingTime="10" timeLoss="12" departDelay="2"/>'
                            '<tripinfo arrival="-1"/><tripinfo arrival="40" vaporized="true"/></tripinfos>')
            r=summarize_trips(path);self.assertEqual(r['count'],1);self.assertEqual(r['excluded'],2)
            self.assertEqual(r['means'],dict(travel_time=40,waiting_time=10,time_loss=12,departure_delay=2))
            path.write_text('<tripinfos/>');r=summarize_trips(path)
            self.assertEqual(r['count'],0);self.assertIsNone(r['means']['travel_time'])
            path.write_text('<tripinfos><tripinfo arrival="3" duration="nan"/></tripinfos>')
            with self.assertRaises(ValueError):summarize_trips(path)

    def test_teleport_removals_are_arrivals_but_not_completions(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trips.xml'
            path.write_text('<tripinfos><tripinfo arrival="50" duration="40" waitingTime="10" timeLoss="12" departDelay="2"/>'
                '<tripinfo arrival="40" vaporized="teleport"/><tripinfo arrival="-1"/></tripinfos>')
            trips=summarize_trips(path)
            self.assertEqual((trips['count'],trips['removed'],trips['unfinished'],trips['arrived_records']),(1,1,1,2))
            row=dict(completed=2,inserted=4)
            attach_trip_metrics(row,trips)
            self.assertEqual((row['arrived'],row['completed'],row['removed'],row['completion_rate']),(2,1,1,.25))
            self.assertEqual(row['trips']['means']['travel_time'],40)
            with self.assertRaisesRegex(ValueError,'arrival records'):
                attach_trip_metrics(dict(completed=3,inserted=4),trips)

    def test_shard_integrity_and_resume(self):
        row=dict(scenario='seen_Uniform',realization=0,metrics_schema=METRICS_SCHEMA,sample_count=3600,horizon=3600,mean_queue=2)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'row.json.gz';atomic_gzip_json(path,dict(identity='model',row=row,hash=digest(row)))
            self.assertEqual(checked_row(path,'model','seen_Uniform',0),row)
            for identity,scenario,index in [('other','seen_Uniform',0),('model','peak_1.5',0),('model','seen_Uniform',1)]:
                with self.assertRaises(ValueError):checked_row(path,identity,scenario,index)
            row['mean_queue']=3;atomic_gzip_json(path,dict(identity='model',row=row,hash='damaged'))
            with self.assertRaises(ValueError):checked_row(path,'model','seen_Uniform',0)

    def test_partial_evaluation_resumes_only_missing_rollouts_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);run=base/'model';run.mkdir();checkpoint=run/'checkpoint';checkpoint.mkdir()
            atomic_json(run/'manifest.json',dict(network='grid',controller='dql',seed=9001,
                method='baseline',pilot=True,validation_hash='validation'))
            atomic_json(run/'result.json',dict(status='max_budget_reached',best_episode=0,completed_episodes=5000))
            manifest=test_manifest('grid');calls=[];fail={'once':True}
            def worker(request):
                calls.append(list(request['indices']))
                for i in request['indices']:
                    if i==1 and fail['once']:
                        fail['once']=False
                        raise RuntimeError('simulated worker interruption')
                    row=dict(scenario='mixture_1',realization=i,metrics_schema=METRICS_SCHEMA,
                             sample_count=3600,horizon=3600,mean_queue=2+i)
                    atomic_gzip_json(Path(request['rows_directory'])/('mixture_1_%02d.json.gz'%i),
                        dict(identity=request['row_identity'],row=row,hash=digest(row)))
            with patch('experiments.evaluation.Checkpoints') as store, \
                 patch('experiments.evaluation.inspect_checkpoint',return_value=dict(hash='checkpoint',
                    signatures=dict(controller=dict(config={'ENV_CONFIG':{}})))), \
                 patch('experiments.evaluation.load_manifest',return_value=(manifest,digest(manifest))), \
                 patch('experiments.evaluation.runtime_identity',return_value={}), \
                 patch('experiments.evaluation.fingerprint',return_value='source'), \
                 patch('experiments.evaluation.root',side_effect=lambda kind:base/kind), \
                 patch('experiments.evaluation.evaluate',side_effect=worker):
                store.return_value.role.return_value=checkpoint
                options=dict(pilot=True,rollouts=2,scenarios=['mixture_1'])
                output=base/'evaluation'
                with self.assertRaisesRegex(RuntimeError,'interruption'):
                    evaluate_run(run,'test.json',output,**options)
                first=output/'rows/mixture_1_00.json.gz';before=first.read_bytes()
                self.assertEqual(json.loads((output/'result.json').read_text())['status'],'failed')
                self.assertEqual(len(list((output/'rows').glob('*.gz'))),1)
                evaluate_run(run,'test.json',output,resume=True,**options)
                self.assertEqual(calls,[[0,1],[1]])
                self.assertEqual(first.read_bytes(),before)
                self.assertEqual(len(read_evaluation(output)[1]),2)
                other=base/'cached_best'
                for invalid in ('best_at_common_budget',):
                    with self.assertRaisesRegex(ValueError,'Unknown evaluation selection'):
                        evaluate_run(run,'test.json',other,selection=invalid,**options)
                evaluate_run(run,'test.json',other,**options)
                self.assertEqual(calls,[[0,1],[1]])
                metadata,rows=read_evaluation(other)
                self.assertEqual(len(rows),2)
                self.assertEqual((metadata['best_episode'],metadata['completed_episodes']),(0,5000))
                store.return_value.role.assert_called_with('best_inference')
                final=base/'cached_final'
                evaluate_run(run,'test.json',final,selection='final',**options)
                metadata,_=read_evaluation(final)
                self.assertEqual(metadata['selection'],'final')
                self.assertEqual(metadata['selected_episode'],5000)
                store.return_value.role.assert_called_with('final_inference')
                first.write_bytes(b'corrupted')
                with self.assertRaises((OSError,ValueError)):
                    evaluate_run(run,'test.json',output,resume=True,**options)
                self.assertEqual(calls,[[0,1],[1]])

    def test_report_single_seed_and_missing_training_seed(self):
        scenes=test_manifest('grid')['scenarios'];fixtures={}
        methods=['baseline','random_group','domain_randomization','fixed_wce','online_wce']
        for j,method in enumerate(methods):
            m=dict(network='grid',controller='dql',method=method,selection='best',seed=101,pilot=False,rollouts=10,
                   checkpoint_hash=method,test_hash='test',validation_hash='validation',source='source',runtime={},metrics_schema=METRICS_SCHEMA)
            rows=[]
            for s in scenes:
                for i in range(10):
                    q=10-j+i/10
                    rows.append(dict(scenario=s['id'],realization=i,mean_queue=q,integrated_queue=q*3600,
                        horizon=3600,sample_count=3600,minutes=[[k*60,1,1] for k in range(1,61)],window_counts=[1200]*3,
                        demand_hash='traffic'+s['id']+str(i),sumo_seed=i,arrival_seed=i,policy_seed=i,
                        trips=dict(count=1,sums=dict(travel_time=10,waiting_time=2,time_loss=3,departure_delay=1))))
            fixtures[method]=(m,rows)
        with tempfile.TemporaryDirectory() as d,patch('experiments.reporting.read_evaluation',side_effect=lambda p:fixtures[p]):
            result=report(methods,Path(d)/'one',networks=['grid'],controllers=['dql'])
            self.assertEqual(result['n_training_seeds'],1)
            summary=json.loads((Path(d)/'one/summary.json').read_text())
            self.assertTrue(all(row['training_seed_sd'] is None for row in summary['comparisons']))
            self.assertTrue(all(row['method_minus_comparator']<0 for row in summary['paired_training_seeds']))
            with self.assertRaisesRegex(ValueError,'Incomplete requested'):
                report(methods,Path(d)/'three',[101,202,303],['grid'],['dql'],['best'])
            for method in methods:
                manifest,rows=fixtures[method]
                fixtures[method+'_final']=(dict(manifest,selection='final'),rows)
            dual=report(list(fixtures),Path(d)/'dual',[101],['grid'],['dql'],['best','final'])
            self.assertEqual(dual['model_cells'],10)
            fixtures['online_wce'][1][0]['policy_seed']=99
            with self.assertRaisesRegex(ValueError,'Unpaired'):
                report(methods,Path(d)/'bad',networks=['grid'],controllers=['dql'],selections=['best'])
        self.assertIsNone(sample_sd([1]));self.assertEqual(bootstrap_mean_ci([-2]*10,'x'),[-2,-2])
