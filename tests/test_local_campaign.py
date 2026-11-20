import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.campaign import (task_matrix, prepare, load_campaign, resources_fit,
    resource_for, process_stat, alive, run_queue, GIB)
from experiments.protocol import settings, training_seeds, scientific_protocol
from experiments.scenarios import test_manifest
from experiments.validation import validation_manifest
from experiments.storage import atomic_json
from experiments.runtime import check_baseline, verify_runtime, verify_gate
from experiments.runner import parser, preflight


class CampaignTests(unittest.TestCase):
    def test_single_and_extended_matrix(self):
        tests={n:test_manifest(n) for n in ('grid','monaco')}
        rows=task_matrix([101],'/tmp/test-runs',tests)
        self.assertEqual(sum(t.get('stage')=='parent' for t in rows),6)
        self.assertEqual(sum(t.get('stage')=='wce' for t in rows),6)
        self.assertEqual(sum(t.get('stage')=='continue' for t in rows),30)
        self.assertEqual(sum(t['kind']=='evaluate' for t in rows),1380)
        self.assertEqual(sum(t['kind']=='evaluate' and t['selection']=='best' for t in rows)*10,6900)
        ids={t['id'] for t in rows};self.assertEqual(len(ids),len(rows))
        for t in rows:self.assertTrue(set(t['dependencies'])<=ids)
        more=task_matrix([101,202,303],'/tmp/test-runs',tests)
        self.assertEqual(len([t for t in more if t['id'] not in ids and t['kind']=='train']),84)
        self.assertEqual(training_seeds(),[101]);self.assertNotIn('training_seeds',scientific_protocol())
        for seeds in ([],[101,101],[9001],[True],[-1]):
            with self.assertRaises(ValueError):training_seeds(seeds)

    def test_campaign_freeze_and_extension(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);vals={}
            for n in ('grid','monaco'):
                vals[n]=d/(n+'.json');atomic_json(vals[n],validation_manifest(n,8001,True))
            summary=prepare(d/'one',vals,d/'gate.json',run_root=d/'runs')
            self.assertEqual((summary['evaluation_tasks'],summary['evaluation_rollouts']),(1380,13800))
            c=load_campaign(d/'one');self.assertEqual(c['training_seeds'],[101])
            prepare(d/'three',vals,d/'gate.json',[101,202,303],d/'runs',d/'one')
            self.assertEqual(load_campaign(d/'three')['training_seeds'],[101,202,303])
            with self.assertRaises(ValueError):prepare(d/'bad',vals,d/'gate.json',[202],d/'runs',d/'one')
            c['training_seeds']=[202];atomic_json(d/'one/campaign.json',c)
            with self.assertRaisesRegex(ValueError,'Invalid campaign'):load_campaign(d/'one')

    def test_defaults_and_unlimited_segments(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'validation.json';atomic_json(path,validation_manifest('grid',9001))
            args=parser().parse_args(['--stage','parent','--network','grid','--controller','dql','--pilot',
                                     '--validation',str(path),'--segment-seconds','0'])
            p,_,_=preflight(args);self.assertEqual(args.seed,9001);self.assertNotIn('training_seeds',p)
            args.pilot=False;args.seed=None
            with self.assertRaises(ValueError):preflight(args)
            self.assertEqual(args.seed,101)

    def test_resources_include_whole_tree_and_full_replay(self):
        estimate=dict(peak_rss_bytes=2*GIB,cpu_cores=2,checkpoint_headroom_bytes=GIB,coverage=['training','validation','checkpoint'])
        self.assertTrue(resources_fit([],estimate,8*GIB,4,8*GIB,20*GIB,10*GIB))
        self.assertFalse(resources_fit([estimate],estimate,4*GIB,4,8*GIB,20*GIB,10*GIB))
        self.assertFalse(resources_fit([estimate],estimate,8*GIB,3,8*GIB,20*GIB,10*GIB))
        self.assertFalse(resources_fit([estimate],None,8*GIB,8,8*GIB,20*GIB,10*GIB))
        task=dict(resource_class='grid/dql/parent',kind='train',stage='parent',controller='dql')
        self.assertIsNone(resource_for(task,{task['resource_class']:estimate}))
        estimate['coverage'].append('replay_full')
        self.assertIsNotNone(resource_for(task,{task['resource_class']:estimate}))
        stat=process_stat(os.getpid());self.assertTrue(alive(os.getpid(),stat['start']))
        self.assertFalse(alive(os.getpid(),'wrong-start'))

    def test_durable_queue_with_real_subprocess_receipt(self):
        import sys
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);out=d/'result';queue=d/'queue';queue.mkdir()
            task=dict(id='fixture',kind='evaluate',output=str(out),dependencies=[],resource_class='test',controller='dql')
            c=dict(hash='fixture',source='fixture',gate='unused',run_root=str(d/'runs'),tasks=[task],
                   data_root=str(d/'data'),cache_root=str(d/'cache'),assets_root=str(d/'assets'))
            atomic_json(queue/'state.json',dict(campaign_hash='fixture',tasks={'fixture':dict(status='pending',attempts=0,no_progress=0)}))
            code='from pathlib import Path; import json; p=Path('+repr(str(out))+'); p.mkdir(); (p/"result.json").write_text(json.dumps({"status":"complete"}))'
            def completed(t,c):return 'complete' if (out/'local_receipt.json').exists() else None
            with patch('experiments.campaign.load_campaign',return_value=c),patch('experiments.runtime.verify_gate'),\
                 patch('experiments.campaign.task_command',return_value=[sys.executable,'-c',code]),\
                 patch('experiments.campaign.completed_task',side_effect=completed),patch('experiments.campaign.time.sleep',side_effect=lambda _:__import__('select').select([],[],[],.03)):
                result=run_queue(queue,disk_reserve_gib=0)
                self.assertEqual(result['counts'],{'complete':1})
                again=run_queue(queue,disk_reserve_gib=0)
                self.assertEqual(again['counts'],{'complete':1})
                self.assertEqual(json.loads((queue/'state.json').read_text())['tasks']['fixture']['attempts'],1)


class RuntimeTests(unittest.TestCase):
    def test_locked_baseline_and_mismatch(self):
        from experiments.protocol import ROOT
        from experiments.core import digest
        packages=dict(line.split('==') for line in (ROOT/'packaging/requirements.in').read_text().splitlines() if '==' in line)
        actual=dict(python='3.10.12',sumo='Eclipse SUMO sumo Version 1.26.0',traci='1.26.0',sumolib='1.26.0',
                    numpy='1.26.4',tensorflow='2.15.1',packages=packages)
        check_baseline(actual)
        bad=copy.deepcopy(actual);bad['sumo']='Eclipse SUMO sumo v1_26_0+0455'
        with self.assertRaises(ValueError):check_baseline(bad)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'runtime.json';body=dict(schema='runtime_lock_v1',profile='local',runtime=actual)
            atomic_json(path,dict(body,hash=digest(body)))
            with patch('experiments.runtime.identity',return_value=actual):self.assertEqual(verify_runtime(path),actual)
            with patch('experiments.runtime.identity',return_value=dict(actual,python='3.10.13')):
                with self.assertRaises(ValueError):verify_runtime(path)
            gate=Path(d)/'gate.json';atomic_json(gate,dict(status='passed',fingerprint='code',runtime=actual,cases=['grid_dql']))
            with patch('experiments.runtime.verify_runtime',return_value=actual),patch('experiments.runner.fingerprint',return_value='code'):
                with self.assertRaises(ValueError):verify_gate(gate)


class QueueRecoveryTests(unittest.TestCase):
    def fixture(self,directory,tasks):
        d=Path(directory);queue=d/'queue';queue.mkdir()
        c=dict(hash='fixture',source='fixture',gate='unused',run_root=str(d/'runs'),tasks=tasks,
               data_root=str(d/'data'),cache_root=str(d/'cache'),assets_root=str(d/'assets'))
        atomic_json(queue/'state.json',dict(campaign_hash='fixture',tasks={t['id']:dict(status='pending',attempts=0,no_progress=0) for t in tasks}))
        return queue,c

    def test_adopts_running_supervisor_without_duplicate(self):
        import subprocess,sys
        from experiments.core import digest
        from experiments.protocol import ROOT
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);out=d/'result'
            task=dict(id='fixture',kind='evaluate',output=str(out),dependencies=[],resource_class='test',controller='dql')
            queue,c=self.fixture(d,[task]);request=d/'request.json';receipt=d/'receipt.json'
            code='import time,json; from pathlib import Path; time.sleep(.2); p=Path('+repr(str(out))+'); p.mkdir(); (p/"result.json").write_text(json.dumps({"status":"complete"}))'
            spec=dict(argv=[sys.executable,'-c',code],cwd=str(ROOT),environment={},log=str(d/'child.log'),receipt=str(receipt))
            spec['request_hash']=digest(spec);atomic_json(request,spec)
            child=subprocess.Popen([sys.executable,'-m','experiments.local_worker',str(request)],cwd=ROOT)
            try:
                row=dict(status='running',attempts=1,no_progress=0,pid=child.pid,start=process_stat(child.pid)['start'],
                         receipt=str(receipt),request_hash=spec['request_hash'],log=spec['log'])
                atomic_json(queue/'state.json',dict(campaign_hash='fixture',tasks={'fixture':row}))
                with patch('experiments.campaign.load_campaign',return_value=c),patch('experiments.runtime.verify_gate'),\
                     patch('experiments.campaign.task_command',side_effect=AssertionError('Duplicate launch')),\
                     patch('experiments.campaign.completed_task',return_value='complete'),\
                     patch('experiments.campaign.time.sleep',side_effect=lambda _:__import__('select').select([],[],[],.03)):
                    self.assertEqual(run_queue(queue,disk_reserve_gib=0)['counts'],{'complete':1})
                self.assertEqual(json.loads((queue/'state.json').read_text())['tasks']['fixture']['attempts'],1)
            finally:child.wait(timeout=10)

    def test_queue_lock_and_failure_blocks_dependants(self):
        import fcntl,sys
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            tasks=[dict(id=str(i),kind='evaluate',output=str(d/str(i)),dependencies=[] if i==0 else ['0'],resource_class='test',controller='dql') for i in range(2)]
            queue,c=self.fixture(d,tasks);Path(c['run_root']).mkdir()
            with patch('experiments.campaign.load_campaign',return_value=c),patch('experiments.runtime.verify_gate'),\
                 patch('experiments.campaign.task_command',return_value=[sys.executable,'-c','raise SystemExit(2)']),\
                 patch('experiments.campaign.completed_task',return_value=None),\
                 patch('experiments.campaign.time.sleep',side_effect=lambda _:__import__('select').select([],[],[],.03)):
                with (Path(c['run_root'])/'.local_scheduler.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    with self.assertRaises(BlockingIOError):run_queue(queue,disk_reserve_gib=0)
                self.assertEqual(run_queue(queue,disk_reserve_gib=0)['counts'],{'failed':1,'blocked':1})

    def test_stops_after_three_normal_segments_without_progress(self):
        import sys
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);out=d/'result'
            task=dict(id='fixture',kind='train',stage='parent',output=str(out),dependencies=[],resource_class='test',controller='dql')
            queue,c=self.fixture(d,[task])
            code='from pathlib import Path; import json; p=Path('+repr(str(out))+'); p.mkdir(exist_ok=True); (p/"manifest.json").write_text("{}"); (p/"result.json").write_text(json.dumps({"status":"interrupted","completed_episodes":0}))'
            with patch('experiments.campaign.load_campaign',return_value=c),patch('experiments.runtime.verify_gate'),\
                 patch('experiments.campaign.task_command',return_value=[sys.executable,'-c',code]),\
                 patch('experiments.campaign.completed_task',return_value=None),\
                 patch('experiments.checkpoint.Checkpoints',return_value=SimpleNamespace(role=lambda _:d/'checkpoint')),\
                 patch('experiments.checkpoint.inspect_checkpoint'),\
                 patch('experiments.campaign.time.sleep',side_effect=lambda _:__import__('select').select([],[],[],.03)):
                self.assertEqual(run_queue(queue,disk_reserve_gib=0)['counts'],{'failed':1})
            self.assertEqual(json.loads((queue/'state.json').read_text())['tasks']['fixture']['attempts'],3)

    def test_tuning_campaign_runs_24_explicit_development_configs(self):
        from experiments.campaign import prepare_tuning,task_command
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);vals={}
            for n in ('grid','monaco'):
                vals[n]=d/(n+'.json');atomic_json(vals[n],validation_manifest(n,9001))
            prepare_tuning(d/'tuning',vals,d/'runs');c=load_campaign(d/'tuning')
            self.assertEqual(c['purpose'],'development');self.assertEqual(len(c['tasks']),24)
            argv=task_command(c['tasks'][0],c,d/'tuning/campaign.json')
            self.assertIn('--pilot',argv);self.assertNotIn('--gate',argv);self.assertNotIn(None,argv)
            args=parser().parse_args(['--stage']+argv[2:3]+argv[3:])
            p,_,_=preflight(args)
            self.assertEqual(p['max_episodes']['parent'],1000);self.assertEqual(args.seed,9001)
