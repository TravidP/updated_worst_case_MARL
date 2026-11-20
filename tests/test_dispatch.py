import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.admission import GIB, observe
from experiments.campaign import run_queue, task_matrix
from experiments.dashboard import change
from experiments.dispatch import permitted, read_json, task_number
from experiments.storage import atomic_json
from experiments.workflow import initialize, freeze, run_phase, step, seed_parent_measurements


class DispatchTests(unittest.TestCase):
    def simulate(self,n,kind):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as mocks:
            d=Path(tmp);attempts=[];pids={};frames=[]
            tasks=[dict(id=str(i),kind='evaluate',output=str(d/f'output{i}'),
                        resource_class='normal',controller='dql',dependencies=[]) for i in range(n)]
            cold=kind in ('cold_tuning','cold_formal','backfill')
            if cold:
                atomic_json(d/'control.json',dict(auto=True))
                if kind=='cold_tuning':
                    for i,t in enumerate(tasks):
                        t.update(kind='train',stage='parent',candidate_rank=i%4,config_hash=str(i))
                if kind=='backfill':
                    tasks[1]['resource_class']='wide'
            if kind=='oversize':tasks[0]['resource_class']='large'
            c=dict(hash='id',source='src',purpose='development',tasks=tasks,run_root=str(d/'run'),
                   data_root=tmp,cache_root=tmp,assets_root=tmp)
            if kind=='cold_formal':c.update(purpose='formal',gate='fixture')
            atomic_json(d/'state.json',dict(campaign_hash='id',tasks={
                t['id']:dict(status='pending',attempts=0,no_progress=0,ready_since=0) for t in tasks}))
            profile=dict(peak_rss_bytes=GIB,cpu_cores=1,checkpoint_headroom_bytes=GIB,coverage=['evaluation'])
            atomic_json(d/'resources',dict(source='src',runtime={},classes={'normal':profile,'large':dict(profile,cpu_cores=17)}))
            if kind=='backfill':
                # Older wide job cannot coexist with job 0. Job 2 still fits.
                atomic_json(d/'resources',dict(source='src',runtime={},classes={
                    'wide@default':dict(profile,peak_rss_bytes=61*GIB)}))
            def popen(cmd,**kwargs):
                req=read_json(cmd[-1]);attempts.append(req);pid=len(attempts)+80;pids[pid]=req
                return SimpleNamespace(pid=pid,wait=lambda:0)
            def end(req):
                t=tasks[int(req['argv'][0])];out=Path(t['output']);out.mkdir()
                atomic_json(out/'result.json',dict(status='complete'))
                atomic_json(req['receipt'],dict(exit_code=0,request_hash=req['request_hash'],ended=1))
            def tick(_):
                state=read_json(d/'state.json');rows=state['tasks'].values()
                running=[r for r in rows if r['status']=='running'];waiting=[r for r in rows if r.get('wait')=='waiting_resources']
                frames.append((len(running),len(waiting)))
                self.assertLess(len(frames),12,'Queue failed to make progress')
                for r in (running[:1] if len(frames)==1 else running):end(pids[r['pid']])
            def memory():
                available=GIB if kind=='ram' and not frames else 100*GIB
                return 100*GIB,available
            def disk(*args):
                return SimpleNamespace(free=(GIB if kind=='disk' and not frames else 200*GIB))
            def replace(name,**kwargs):mocks.enter_context(patch('experiments.campaign.'+name,**kwargs))
            replace('load_campaign',return_value=c);mocks.enter_context(patch('experiments.runtime.identity',return_value={}))
            replace('completed_task',side_effect=lambda t,c:'complete' if (Path(t['output'])/'local_receipt.json').exists() else None)
            replace('cpu_capacity',return_value=dict(physical=20,logical=20,effective=20,default_budget=16))
            replace('memory_info',side_effect=memory);replace('shutil.disk_usage',side_effect=disk)
            replace('subprocess.Popen',side_effect=popen);replace('process_stat',return_value=dict(start='1',state='R'))
            replace('process_tree',return_value=dict(rss_bytes=1,cpu_ticks=1,processes=1))
            replace('task_command',side_effect=lambda t,*args:[t['id']]);replace('time.sleep',side_effect=tick)
            mocks.enter_context(patch('experiments.runtime.verify_gate'))
            state=run_queue(d,max_parallel=24,workflow=d if cold else None,
                            resources=None if kind.startswith('cold_') else d/'resources')
            self.assertEqual(len(attempts),len({r['argv'][0] for r in attempts}))
            return state,frames,read_json(d/'state.json')

    def test_pool_backfills_one_free_slot(self):
        result,frames,_=self.simulate(24,'cpu')
        self.assertEqual(frames[:2],[(16,8),(16,7)])
        self.assertEqual(result['counts'],{'complete':24})

    def test_workflow_concurrent_cold_start_and_refill(self):
        for kind,slots in [('cold_tuning',12),('cold_formal',16)]:
            with self.subTest(kind=kind):
                result,frames,_=self.simulate(24,kind)
                self.assertEqual(frames[:2],[(slots,24-slots),(slots,23-slots)])
                self.assertEqual(result['counts'],{'complete':24})

    def test_old_waiting_job_allows_smaller_job_to_backfill(self):
        result,frames,_=self.simulate(3,'backfill')
        self.assertEqual(frames[0],(2,1))
        self.assertEqual(result['counts'],{'complete':3})

    def test_zero_workers_retry_temporary_ram_and_disk(self):
        for pressure in ('ram','disk'):
            with self.subTest(pressure=pressure):
                result,frames,_=self.simulate(1,pressure)
                self.assertEqual(frames[:2],[(0,1),(1,0)])
                self.assertEqual(result['counts'],{'complete':1})

    def test_oversized_job_leaves_other_lane_available(self):
        result,_,state=self.simulate(2,'oversize')
        self.assertEqual(result['counts'],{'pending':1,'complete':1})
        self.assertEqual(state['tasks']['0']['wait'],'needs_attention')

    def test_task_numbers_and_independent_release(self):
        tests={n:dict(scenarios=[dict(id='scene')]) for n in ('grid','monaco')}
        jobs=task_matrix([101],'/tmp/unused',tests)
        ids={t['id']:task_number(t) for t in jobs if t['kind']=='train'}
        self.assertEqual(set(ids.values()),{f'{i:02d}' for i in range(1,43)})
        def ready(done):
            return {ids[t['id']] for t in jobs if t['id'] in ids and t['dependencies']
                    and ids[t['id']] not in done and all(ids[x] in done for x in t['dependencies'])}
        self.assertEqual(ready({'01'}),{'07','13','19','25'})
        self.assertEqual(ready({'01','07'}),{'13','19','25','31','37'})

    def test_manual_release_pause_resume_and_auto(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'workflow';initialize(d,[71001,71002])
            job=read_json(d/'tuning/campaign.json')['tasks'][0]['id']
            control=change(d,'run',[job],None)
            self.assertTrue(permitted(control,'tuning',job));self.assertFalse(permitted(control,'tuning','other'))
            self.assertFalse(permitted(change(d,'pause',[job],None),'tuning',job))
            self.assertTrue(permitted(change(d,'resume',[job],None),'tuning',job))
            self.assertTrue(permitted(change(d,'auto',[],None),'verify'))
            self.assertFalse(permitted(change(d,'pause',[],None),'verify'))
            self.assertTrue(permitted(change(d,'resume',[],None),'verify'))

    def test_freeze_six_winners_and_reject_a_different_ini(self):
        from experiments.core import file_hash
        from experiments.protocol import config_path
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'workflow';initialize(d,[71001,71002])
            c=read_json(d/'tuning/campaign.json');original={}
            def select(_,output):
                output.mkdir();rows=[]
                for t in c['tasks']:
                    if t['candidate_rank']!=3:continue
                    original_ini=config_path(t['network'],t['controller'])
                    original[original_ini]=file_hash(original_ini)
                    (output/original_ini.name).write_bytes(Path(t['config']).read_bytes())
                    rows.append(dict(network=t['network'],controller=t['controller'],config=original_ini.name,sha256=t['config_hash']))
                atomic_json(output/'selection.json',dict(selected=rows,campaign_hash=c['hash']))
            with patch('experiments.campaign.select_tuning',side_effect=select),patch('experiments.tuning_cleanup.prune_tuning'):
                freeze(d)
            selected=read_json(d/'selection/selection.json')['selected']
            self.assertEqual(len(selected),6)
            self.assertTrue(all(file_hash(f)==original[f] for f in original))
            for row in selected:self.assertEqual(file_hash(d/'frozen/config/revised'/row['config']),row['sha256'])
            old=selected[0];(d/'frozen/config/revised'/old['config']).write_text('changed')
            with self.assertRaisesRegex(ValueError,'differs from selected'):
                run_phase(d,'verify',d)
            with self.assertRaisesRegex(ValueError,'prerequisite'):
                step(d,'formal')

    def test_calibration_coverage_and_exact_configuration_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);output=d/'job';cp=output/'checkpoints/cp'
            atomic_json(output/'references.json',dict(roles=dict(latest='cp')))
            atomic_json(cp/'manifest.json',dict(kind='full',files={'controller/replay/index.json':'fixture'}))
            atomic_json(cp/'controller/replay/index.json',dict(size=128,capacity=128))
            t=dict(kind='train',output=str(output));r=dict(status='running',peak_rss_bytes=2*GIB,rss_bytes=GIB,cpu_cores=1.)
            learned={};key='grid/dql/parent@winner'
            for phase in ('training','validation'):observe(t,r,learned,key,dict(phase=phase))
            self.assertEqual(set(learned[key]['coverage']),{'training','validation','checkpoint','replay_full'})
            learned['grid/dql/parent@loser']=dict(learned[key],peak_rss_bytes=10*GIB)
            atomic_json(d/'workflow.json',dict(source='development'))
            atomic_json(d/'tuning/measurements.json',dict(source='development',runtime={'host':'one'},classes=learned))
            atomic_json(d/'selection/selection.json',dict(selected=[dict(network='grid',controller='dql',sha256='winner')]))
            with patch('experiments.runtime.identity',return_value={'host':'one'}):
                seed_parent_measurements(d,dict(source='formal'))
            result=read_json(d/'formal/measurements.json')
            self.assertEqual(result['source'],'formal')
            self.assertEqual(result['classes'],{'grid/dql/parent@default':learned[key]})


if __name__=='__main__':unittest.main()
