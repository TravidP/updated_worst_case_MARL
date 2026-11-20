"""Frozen local task DAG and resource-bounded, restartable CPU process queue."""
import argparse
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import sys
import time
from pathlib import Path
from experiments.core import digest, file_hash
from experiments.protocol import ROOT, root, settings, scientific_protocol, training_seeds
from experiments.storage import atomic_json, append_record
from experiments.admission import admission, cpu_capacity, memory_info, observe, workflow_reservation, TUNING_SLOTS
from experiments.dispatch import read_json, stage, permitted, task_number

GIB=1024**3


def load_campaign(path):
    from experiments.runner import fingerprint
    path=Path(path)
    if path.is_dir():path=path/'campaign.json'
    c=json.loads(path.read_text());body={k:v for k,v in c.items() if k!='hash'}
    if c.get('schema')!='local_campaign_v1' or c.get('hash')!=digest(body):raise ValueError('Invalid campaign manifest')
    development=c.get('purpose')=='development'
    if development:
        if c['training_seeds']!=[settings()['pilot_seed']]:raise ValueError('Development requires the pilot seed')
    else:training_seeds(c['training_seeds'])
    if c['source']!=fingerprint() or c['protocol']!=scientific_protocol():raise ValueError('Campaign source/protocol is stale')
    from experiments.validation import load_manifest
    for network,entry in c['validation'].items():
        if load_manifest(entry['path'],network,not development)[1]!=entry['hash']:raise ValueError('Campaign validation changed')
    for network,entry in c['tests'].items():
        if load_manifest(entry['path'],network,True,'test')[1]!=entry['hash']:raise ValueError('Campaign test changed')
    return c


def prepare_tuning(output,validation,run_root=None):
    from experiments.runner import fingerprint
    from experiments.validation import load_manifest
    from experiments.tuning import generate
    out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    vals={}
    for n in settings()['networks']:
        m,h=load_manifest(validation[n],n)
        if m['frozen']:raise ValueError('Tuning requires an unfrozen development manifest, separate from formal selection')
        vals[n]=dict(path=str(Path(validation[n]).resolve()),hash=h)
    run_root=Path(run_root or root('run')).resolve();out.mkdir(parents=True)
    generate(out/'configs');candidates=json.loads((out/'configs/candidates.json').read_text());jobs=[]
    for row in candidates:
        tid='%s/%s/candidate_%d'%(row['network'],row['controller'],row['tie_rank'])
        path=out/'configs'/row['config']
        jobs.append(dict(id=tid,kind='train',stage='parent',method='baseline',network=row['network'],
            controller=row['controller'],seed=row['seed'],output=str(run_root/'development'/tid),dependencies=[],
            resource_class=row['network']+'/'+row['controller']+'/parent',config=str(path),config_hash=file_hash(path),
            candidate_rank=row['tie_rank']))
    c=dict(schema='local_campaign_v1',purpose='development',source=fingerprint(),protocol=scientific_protocol(),
        training_seeds=[settings()['pilot_seed']],validation=vals,tests={},gate=None,run_root=str(run_root),
        data_root=str(root('data')),cache_root=str(root('cache')),assets_root=str(root('assets')),tasks=jobs,
        extends=None,evaluation_selections=[])
    atomic_json(out/'campaign.json',dict(c,hash=digest(c)))
    atomic_json(out/'state.json',dict(campaign_hash=digest(c),tasks={j['id']:dict(status='pending',attempts=0,no_progress=0) for j in jobs}))
    return dict(campaign=str(out/'campaign.json'),development_tasks=len(jobs),episodes_per_candidate=1000)


def tuning_winners(c):
    from experiments.checkpoint import Checkpoints, inspect_checkpoint
    from experiments.storage import read_records
    if c.get('purpose')!='development':raise ValueError('Expected a development campaign')
    expected={(n,a,r) for n in settings()['networks'] for a in settings()['controllers'] for r in range(4)}
    actual=[(t['network'],t['controller'],t['candidate_rank']) for t in c['tasks']]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Expected four candidates for each map and controller')
    choices={}
    for task in c['tasks']:
        path=Path(task['output'])
        if completed_task(task,c)!='complete':raise ValueError('All 24 tuning candidates must finish')
        result=json.loads((path/'result.json').read_text())
        if result['completed_episodes']!=1000:raise ValueError('Each development candidate requires 1000 episodes')
        history=read_records(path/'validation.jsonl.gz')
        # A checkpoint hash in the validation record identifies the selected candidate.
        h=inspect_checkpoint(Checkpoints(path).role('best_inference'))['hash']
        score=min(r['score'] for r in history if r['checkpoint_hash']==h)
        key=(task['network'],task['controller']);choices.setdefault(key,[]).append((score,task['candidate_rank'],task))
    selected=[]
    for key,items in sorted(choices.items()):
        score,rank,task=min(items,key=lambda item:item[:2])
        source=Path(task['config'])
        if file_hash(source)!=task['config_hash']:raise ValueError('Tuning config changed')
        name='config_%s_%s.ini'%(key[1],'large' if key[0]=='grid' else 'real')
        selected.append(dict(network=key[0],controller=key[1],score=score,tie_rank=rank,config=name,sha256=task['config_hash']))
    return selected


def select_tuning(directory,output):
    c=load_campaign(directory)
    selected=tuning_winners(c)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    sources={(t['network'],t['controller'],t['candidate_rank']):t['config'] for t in c['tasks']}
    for row in selected:
        source=sources[row['network'],row['controller'],row['tie_rank']]
        shutil.copyfile(source,out/row['config'])
        if file_hash(out/row['config'])!=row['sha256']:raise ValueError('Winner export mismatch')
    atomic_json(out/'selection.json',dict(campaign_hash=c['hash'],selected=selected,
        next_step='Version these six configs in config/revised and rerun verification before preparing the formal campaign.'))
    return dict(selected=len(selected),output=str(out))


def task_matrix(seeds, run_root, tests, include_evaluation=True):
    """Pure task generation: 42 training jobs and 1380 scenario jobs for one seed."""
    p=settings();jobs=[];run_root=Path(run_root)
    for n in p['networks']:
        for a in p['controllers']:
            for s in training_seeds(seeds):
                prefix='%s/%s/seed_%d'%(n,a,s);base=run_root/prefix
                for stage,method in [('parent','baseline'),('wce','baseline')]+[('continue',m) for m in p['methods']]:
                    label=method if stage=='continue' else stage
                    deps=[] if stage=='parent' else [prefix+'/parent']
                    if stage=='continue' and method in ('fixed_wce','online_wce'):deps.append(prefix+'/wce')
                    jobs.append(dict(id=prefix+'/'+label,kind='train',stage=stage,method=method,network=n,
                        controller=a,seed=s,output=str(base/label),dependencies=deps,resource_class=n+'/'+a+'/'+label))
                if include_evaluation:
                    for m in p['methods']:
                        for selection in ('best','final'):
                            for scene in tests[n]['scenarios']:
                                jobs.append(dict(id=prefix+'/'+m+'/'+selection+'/'+scene['id'],kind='evaluate',
                                    network=n,controller=a,seed=s,method=m,selection=selection,scenario=scene['id'],
                                    run=str(base/m),output=str(base/'evaluations'/m/selection/scene['id']),
                                    dependencies=[prefix+'/'+m],resource_class=n+'/'+a+'/evaluate'))
    return jobs


def prepare(output, validation, gate, seeds=None, run_root=None, extend=None, include_evaluation=True):
    from experiments.runner import fingerprint
    from experiments.scenarios import test_manifest
    from experiments.validation import load_manifest
    out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    seeds=training_seeds(seeds);run_root=Path(run_root or root('run')).resolve()
    vals={n:dict(path=str(Path(validation[n]).resolve()),hash=load_manifest(validation[n],n,True)[1])
          for n in settings()['networks']}
    previous=load_campaign(extend) if extend else None
    if previous and (previous['validation']!=vals or not set(previous['training_seeds'])<=set(seeds)
                     or previous['run_root']!=str(run_root)):
        raise ValueError('Extension must retain prior seeds, validation identity and run root')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        staging=Path(temp)/'campaign';staging.mkdir()
        tests={n:test_manifest(n) for n in settings()['networks']};test_entries={}
        for n,m in tests.items():
            path=out/(n+'_test.json');atomic_json(staging/path.name,m)
            test_entries[n]=dict(path=str(path),hash=digest(m))
        jobs=task_matrix(seeds,run_root,tests,include_evaluation)
        c=dict(schema='local_campaign_v1',purpose='formal',source=fingerprint(),protocol=scientific_protocol(),
            training_seeds=seeds,validation=vals,tests=test_entries,gate=str(Path(gate).resolve()),
            run_root=str(run_root),data_root=str(root('data')),cache_root=str(root('cache')),assets_root=str(root('assets')),
            tasks=jobs,extends=str(Path(extend).resolve()) if extend else None,
            evaluation_selections=['best','final'] if include_evaluation else [])
        atomic_json(staging/'campaign.json',dict(c,hash=digest(c)))
        atomic_json(staging/'state.json',dict(campaign_hash=digest(c),tasks={j['id']:dict(status='pending',attempts=0,no_progress=0) for j in jobs}))
        staging.rename(out)
    evaluation_tasks=sum(j['kind']=='evaluate' for j in jobs)
    return dict(campaign=str(out/'campaign.json'),training_tasks=sum(j['kind']=='train' for j in jobs),
                evaluation_tasks=evaluation_tasks,evaluation_rollouts=evaluation_tasks*10)


def process_stat(pid):
    try:
        # comm may contain spaces and parentheses; fields after its final ')' are stable.
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),state=fields[0],ppid=int(fields[1]),start=fields[19],
                    cpu_ticks=int(fields[11])+int(fields[12]),rss=int(fields[21])*os.sysconf('SC_PAGE_SIZE'))
    except (OSError,ValueError,IndexError):return None


def alive(pid,start):
    stat=process_stat(pid)
    return bool(stat and stat['start']==start and stat['state']!='Z')


def process_tree(pid):
    rows={}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            stat=process_stat(int(path.name))
            if stat:rows[stat['pid']]=stat
    found={pid};previous=set()
    while found!=previous:
        previous=set(found);found|={p for p,s in rows.items() if s['ppid'] in found}
    return dict(rss_bytes=sum(rows[p]['rss'] for p in found if p in rows),
                cpu_ticks=sum(rows[p]['cpu_ticks'] for p in found if p in rows),processes=len(found))


def physical_cpu_budget():
    return cpu_capacity()['default_budget']


def resources_fit(active, candidate, ram_budget, cpu_budget, available, free_disk, disk_reserve):
    return admission(active,candidate,ram_budget,cpu_budget,available,free_disk,disk_reserve)[0]=='ready'


def resource_for(task, estimates):
    entry=estimates.get(task['resource_class'])
    if entry is None:return None
    for k in ('peak_rss_bytes','cpu_cores','checkpoint_headroom_bytes'):
        if k not in entry or not math.isfinite(entry[k]) or entry[k]<=0:raise ValueError('Invalid resource estimate '+k)
    required={'training','validation','checkpoint'} if task['kind']=='train' else {'evaluation'}
    if task['controller']=='dql' and task['kind']=='train' and task['stage']!='wce':required.add('replay_full')
    if not required<=set(entry.get('coverage',[])):return None
    return entry


def task_command(task,campaign,campaign_path,segment_seconds=0):
    from experiments.checkpoint import Checkpoints,inspect_checkpoint
    base=[sys.executable,str(ROOT/'main.py')]
    if task['kind']=='evaluate':
        argv=[sys.executable,str(ROOT/'scripts/evaluate_suite.py'),'--run',task['run'],
              '--manifest',campaign['tests'][task['network']]['path'],'--selection',task['selection'],
              '--scenario',task['scenario'],'--output',task['output']]
    else:
        argv=base+[task['stage'],'--network',task['network'],'--controller',task['controller'],
            '--seed',str(task['seed']),'--method',task['method'],'--output',task['output'],
            '--validation',campaign['validation'][task['network']]['path'],'--gate',campaign['gate'],
            '--campaign',str(campaign_path),'--segment-seconds',str(segment_seconds)]
        prefix=Path(task['output']).parent
        for flag,label in ([('--parent','parent')] if task['stage']!='parent' else [])+(
                [('--wce','wce')] if task['stage']=='continue' and task['method'] in ('fixed_wce','online_wce') else []):
            cp=Checkpoints(prefix/label).role('best');inspect_checkpoint(cp);argv += [flag,str(cp)]
        if campaign.get('purpose')=='development':
            # Development has a fixed 1000-episode budget and no publication gate.
            index=argv.index('--gate');del argv[index:index+2]
            argv += ['--pilot','--config',task['config'],'--max-episodes','1000']
    if Path(task['output']).exists():argv+=['--resume']
    return argv


def completed_task(task,campaign):
    """Adopt only a verified local receipt and matching immutable run identity."""
    from experiments.checkpoint import Checkpoints,inspect_checkpoint
    path=Path(task['output'])
    if not (path/'local_receipt.json').exists():return None
    receipt=json.loads((path/'local_receipt.json').read_text())
    if receipt.get('exit_code')!=0 or receipt.get('source')!=campaign['source']:return None
    if not (path/'result.json').exists() or not (path/'manifest.json').exists():return None
    result=json.loads((path/'result.json').read_text());m=json.loads((path/'manifest.json').read_text())
    for key in ('network','controller','seed','method'):
        if m[key]!=task[key]:raise ValueError('Existing task identity mismatch: '+task['id'])
    development=campaign.get('purpose')=='development'
    if bool(m.get('pilot'))!=development or m['source']!=campaign['source'] or m['validation_hash']!=campaign['validation'][task['network']]['hash']:
        raise ValueError('Existing task provenance mismatch')
    if task['kind']=='evaluate':
        from experiments.evaluation import read_evaluation
        if result['status']=='complete':
            if m['selection']!=task['selection'] or m['scenarios']!=[task['scenario']] or m['test_hash']!=campaign['tests'][task['network']]['hash']:
                raise ValueError('Existing evaluation identity mismatch')
            read_evaluation(path);return 'complete'
    elif result['status'] in ('early_stopped','no_learning','max_budget_reached'):
        expected=__import__('copy').deepcopy(campaign['protocol'])
        if development:expected['max_episodes']['parent']=1000
        if m['protocol']!=expected or m['stage']!=task['stage']:raise ValueError('Existing protocol mismatch')
        if development and (m.get('config_hash')!=task['config_hash'] or file_hash(task['config'])!=task['config_hash']):
            raise ValueError('Development configuration mismatch')
        inspect_checkpoint(Checkpoints(path).role('best'))
        return 'complete' if development or (result['status']=='max_budget_reached' and result['completed_episodes']==expected['max_episodes'][task['stage']]) else 'review_required'
    return None


def run_queue(directory,max_parallel=1,resources=None,ram_gib=None,cpu_cores=None,disk_reserve_gib=10,segment_seconds=0,workflow=None):
    from experiments.runtime import verify_gate,identity as runtime_identity
    from experiments.checkpoint import Checkpoints,inspect_checkpoint
    directory=Path(directory).resolve();cp=directory/'campaign.json';c=load_campaign(cp)
    if c.get('purpose')!='development':verify_gate(c['gate'])
    if max_parallel<1 or segment_seconds<0:raise ValueError('Invalid scheduler limits')
    estimates=json.loads(Path(resources).read_text()) if resources else {}
    if estimates:
        if estimates.get('source')!=c['source'] or estimates.get('runtime')!=runtime_identity():
            raise ValueError('Resource measurements must match this source and runtime')
        estimates=estimates['classes']
    run_root=Path(c['run_root']);run_root.mkdir(parents=True,exist_ok=True)
    total,_=memory_info();ram=min(ram_gib*GIB if ram_gib else total*.8,total*.8)
    capacity=cpu_capacity()
    cpu=min(cpu_cores,capacity['effective']) if cpu_cores is not None else capacity['default_budget']
    if workflow and c.get('purpose')=='development':
        max_parallel=min(max_parallel,TUNING_SLOTS)
        cpu=min(cpu,TUNING_SLOTS)
    if ram<=0 or cpu<=0 or disk_reserve_gib<0:raise ValueError('Invalid resource budget')
    tasks={t['id']:t for t in c['tasks']};stop={'requested':False,'signals':0};children={}
    learned_path=directory/'measurements.json'
    runtime_id=runtime_identity() if workflow or learned_path.exists() else None
    recorded=read_json(learned_path,{})
    if recorded and (recorded['source']!=c['source'] or recorded['runtime']!=runtime_id):
        raise ValueError('Stale measured resource identity')
    learned=recorded.get('classes',{})
    keys={tid:t['resource_class']+'@'+t.get('config_hash','default') for tid,t in tasks.items()}
    def estimate(tid):
        task=tasks[tid];key=keys[tid]
        if workflow:
            return workflow_reservation(learned.get(key), estimates.get(key))
        return resource_for(task,estimates)
    def save():
        atomic_json(directory/'state.json',state)
    def interrupt(sig,frame):
        stop['requested']=True;stop['signals']+=1
        for row in state['tasks'].values():
            if row['status']=='running' and alive(row['pid'],row['start']):
                if row.get('request'):Path(row['receipt']).with_name('pause').touch(exist_ok=True)
                else:os.kill(row['pid'],sig)
    handlers={sig:signal.signal(sig,interrupt) for sig in (signal.SIGTERM,signal.SIGINT)}
    environment={k:('1' if workflow else os.environ.get(k,'1')) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                                                  'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS')}
    environment.update(TF_ENABLE_ONEDNN_OPTS=os.environ.get('TF_ENABLE_ONEDNN_OPTS','0'),CUDA_VISIBLE_DEVICES='-1',
        CBWCE_RUN_ROOT=c['run_root'],CBWCE_DATA_ROOT=c['data_root'],CBWCE_CACHE_ROOT=c['cache_root'],CBWCE_ASSETS_ROOT=c['assets_root'])
    def supervise(tid,row):
        request=Path(row['request'])
        owner=read_json(request.with_suffix('.owner.json'))
        if owner:
            if owner['request_hash']!=row['request_hash']:raise ValueError('Worker owner identity mismatch')
            row.update(status='running',pid=owner['pid'],start=owner['start'])
        else:
            process=subprocess.Popen([sys.executable,'-m','experiments.local_worker',str(request)],cwd=ROOT,start_new_session=True)
            children[tid]=process
            stat=process_stat(process.pid)
            row.update(status='running',pid=process.pid,start=stat['start'] if stat else '')
    def adopt(task):
        try:return completed_task(task,c)
        except (OSError,KeyError,ValueError) as exc:
            state['tasks'][task['id']]['error']=str(exc)
            return 'failed'
    try:
        with (run_root/'.local_scheduler.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=read_json(directory/'state.json')
            if state['campaign_hash']!=c['hash']:raise ValueError('Queue state identity mismatch')
            for tid,row in state['tasks'].items():
                if row['status']=='starting':supervise(tid,row)
                if row['status']=='pending':
                    adopted=adopt(tasks[tid])
                    if adopted:row['status']=adopted
            save()
            while True:
                control=read_json(Path(workflow)/'control.json',{}) if workflow else {'auto':True}
                now=time.time()
                for tid,row in state['tasks'].items():
                    if row['status']=='failed' and control.get('retry',{}).get(tid,0)>row.get('retry_seen',0):
                        row.update(status='pending',no_progress=0,retry_seen=control['retry'][tid])
                    if row['status']=='blocked' and all(state['tasks'][d]['status'] not in ('blocked','failed','review_required') for d in tasks[tid]['dependencies']):
                        row['status']='pending'
                for tid,row in state['tasks'].items():
                    if row['status']!='running':continue
                    task=tasks[tid];receipt=Path(row['receipt'])
                    if row.get('request'):
                        owner=read_json(Path(row['request']).with_suffix('.owner.json'))
                        if owner and owner['request_hash']==row['request_hash']:
                            row.update(pid=owner['pid'],start=owner['start'])
                    if workflow and (control.get('paused') or tid in control.get('held',[])):
                        Path(row['receipt']).with_name('pause').touch(exist_ok=True)
                    row['reason']='Pausing at a recovery boundary' if receipt.with_name('pause').exists() else ''
                    if receipt.exists():
                        done=json.loads(receipt.read_text())
                        if done.get('request_hash')!=row['request_hash']:
                            row.update(status='failed',error='Worker receipt identity mismatch');save();continue
                        if tid in children:children.pop(tid).wait()
                        row['exit_code']=done['exit_code'];row['ended']=done['ended']
                        result_path=Path(task['output'])/'result.json'
                        result=json.loads(result_path.read_text()) if result_path.exists() else {}
                        if done['exit_code']!=0:
                            row.update(status='failed',error=result.get('error') or done.get('error') or f"Worker exited with code {done['exit_code']}")
                        else:
                            atomic_json(Path(task['output'])/'local_receipt.json',dict(done,source=c['source']))
                            if result.get('status')=='interrupted':
                                if task['kind']=='evaluate':
                                    row.update(status='pending',progress=result.get('rollouts',0))
                                    save()
                                    continue
                                latest=Checkpoints(task['output']).role('latest')
                                if not latest:row['status']='failed';row['error']='Interrupted without a recovery checkpoint'
                                else:
                                    try:inspect_checkpoint(latest)
                                    except (OSError,ValueError,KeyError) as exc:
                                        row.update(status='failed',error=str(exc));save();continue
                                    progress=result.get('completed_episodes',0)
                                    row['no_progress']=row.get('no_progress',0)+1 if progress<=row.get('progress',0) and not receipt.with_name('pause').exists() else 0
                                    row['progress']=progress
                                    row['status']='failed' if row['no_progress']>=3 else 'pending'
                            else:row['status']=adopt(task) or 'failed'
                        save()
                    elif not alive(row['pid'],row['start']):
                        row['status']='failed';row['error']='Worker disappeared without an exit receipt';save()
                    else:
                        sample=process_tree(row['pid']);elapsed=time.monotonic()-row.get('sample_clock',time.monotonic())
                        sample['cpu_cores']=max(0.,(sample['cpu_ticks']-row.get('cpu_ticks',sample['cpu_ticks']))/os.sysconf('SC_CLK_TCK')/max(elapsed,.001))
                        row['cpu_ticks']=sample['cpu_ticks'];row['sample_clock']=time.monotonic()
                        row.update(cpu_cores=sample['cpu_cores'],rss_bytes=sample['rss_bytes'])
                        row['peak_rss_bytes']=max(row.get('peak_rss_bytes',0),sample['rss_bytes'])
                        append_record(directory/'resources.jsonl.gz',dict(time=time.time(),task=tid,resource_class=task['resource_class'],**sample))
                    if workflow:
                        progress=read_json(Path(task['output'])/'progress.json',{})
                        observe(task,row,learned,keys[tid],progress)
                active=[tid for tid,r in state['tasks'].items() if r['status']=='running']
                if stop['requested'] and not active:break
                ready=[]
                for tid,row in state['tasks'].items():
                    if row['status']!='pending':continue
                    task=tasks[tid]
                    deps=[d for d in task['dependencies'] if state['tasks'][d]['status']!='complete']
                    if any(state['tasks'][d]['status'] in ('failed','review_required','blocked') for d in deps):
                        row.update(status='blocked',reason='Upstream requires attention: '+', '.join(deps))
                    elif deps:row.update(wait='dependencies',reason='Waiting for '+', '.join(deps))
                    elif not permitted(control,stage(task),tid):
                        row.update(wait='paused' if control.get('paused') or tid in control.get('held',[]) else 'manual',
                                   reason='Paused' if control.get('paused') or tid in control.get('held',[]) else 'Awaiting manual release')
                        row.pop('ready_since',None)
                    else:
                        row.setdefault('ready_since',now)
                        ready.append(tid)
                ready.sort(key=lambda tid:(tasks[tid]['kind']=='evaluate',state['tasks'][tid]['ready_since'],
                                           tasks[tid].get('candidate_rank',0),task_number(tasks[tid]),tid))
                _,available=memory_info()
                disk=shutil.disk_usage(run_root)
                free=disk.free
                pending_growth=sum(max(0,1.3*profile['peak_rss_bytes']-state['tasks'][tid].get('rss_bytes',0))
                                   for tid in active if (profile:=estimate(tid)))
                usable_ram=max(0,available-total*.2-pending_growth)
                reserve=disk_reserve_gib*GIB
                state['pool']=dict(**capacity,cpu_budget=cpu,ram_budget=ram,available_ram=available,free_disk=free,
                                   max_parallel=max_parallel,updated=now)
                for tid in ready:
                    row=state['tasks'][tid];task=tasks[tid]
                    if stop['requested']:break
                    candidate=estimate(tid)
                    disk_needed=reserve+(candidate['checkpoint_headroom_bytes'] if candidate else 0)
                    if disk_needed>getattr(disk,'total',float('inf')):
                        disposition,reason='needs_attention',f'Job and reserve require {disk_needed/GIB:.1f} GiB; filesystem holds {disk.total/GIB:.1f} GiB'
                    else:
                        disposition,reason=admission([estimate(x) for x in active],candidate,ram,cpu,usable_ram,free,reserve)
                    if disposition=='ready' and len(active)>=max_parallel:
                        disposition='waiting_resources'
                        reason=f'Worker limit reached ({max_parallel})'
                    row.update(wait=disposition,reason=reason)
                    if disposition!='ready':
                        continue
                    if Path(task['output']).exists() and not (Path(task['output'])/'manifest.json').exists():
                        row['status']='failed';row['error']='Existing output without a run manifest';save();continue
                    try:argv=task_command(task,c,cp,segment_seconds)
                    except (OSError,ValueError,KeyError) as exc:
                        row.update(status='failed',error=str(exc));save();continue
                    attempt=directory/'attempts'/(digest(tid)[:16]+'_%03d'%(row['attempts']+1));attempt.mkdir(parents=True,exist_ok=True)
                    spec=dict(argv=argv,cwd=str(ROOT),environment=dict(environment,CBWCE_PAUSE_FILE=str(attempt/'pause')),log=str(attempt/'process.log'),receipt=str(attempt/'receipt.json'))
                    spec['request_hash']=digest(spec);request=attempt/'request.json';atomic_json(request,spec)
                    row.update(status='starting',attempts=row['attempts']+1,request=str(request),
                               request_hash=spec['request_hash'],receipt=spec['receipt'],log=spec['log'],started=time.time(),wait='',reason='')
                    row.pop('ready_since',None);row.pop('ended',None);row.pop('error',None)
                    save()
                    supervise(tid,row);active.append(tid);save()
                    if candidate:usable_ram-=1.3*candidate['peak_rss_bytes']
                if workflow and learned:
                    atomic_json(learned_path,dict(source=c['source'],runtime=runtime_id,classes=learned))
                pending=[r for r in state['tasks'].values() if r['status']=='pending']
                state['note']='Waiting for resources' if not active and any(r.get('wait')=='waiting_resources' for r in pending) else ''
                save()
                if not active and not pending:break
                if not active and not workflow and not any(r.get('wait') in ('waiting_resources','dependencies') for r in pending):break
                time.sleep(5)
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
    return status(directory)


def status(directory):
    state=json.loads((Path(directory)/'state.json').read_text());counts={}
    for r in state['tasks'].values():counts[r['status']]=counts.get(r['status'],0)+1
    waiting={tid:row.get('reason','') for tid,row in state['tasks'].items()
             if row.get('wait') in ('waiting_resources','needs_attention') and row['status']=='pending'}
    return dict(counts=counts,note=state.get('note'),waiting=waiting,pool=state.get('pool',{}),active=[dict(task=k,pid=r['pid'],log=r['log'])
                 for k,r in state['tasks'].items() if r['status']=='running'])


def export_resources(directory,output):
    """Export conservative envelopes from actual sampled trees and checkpoint evidence."""
    from experiments.storage import read_records
    from experiments.checkpoint import Checkpoints,inspect_checkpoint
    from experiments.runtime import identity
    c=load_campaign(directory);tasks={t['id']:t for t in c['tasks']};classes={};runtime=identity()
    for row in read_records(Path(directory)/'resources.jsonl.gz'):
        task=tasks[row['task']];path=Path(task['output'])
        if not (path/'manifest.json').exists():continue
        m=json.loads((path/'manifest.json').read_text())
        if m['source']!=c['source'] or m.get('runtime')!=runtime:raise ValueError('Measured run identity mismatch')
        entry=classes.setdefault(task['resource_class'],dict(peak_rss_bytes=0,cpu_cores=1.,checkpoint_headroom_bytes=GIB,coverage=set()))
        entry['peak_rss_bytes']=max(entry['peak_rss_bytes'],row['rss_bytes'])
        entry['cpu_cores']=max(entry['cpu_cores'],row.get('cpu_cores',0.))
        if task['kind']=='evaluate':
            if (path/'result.json').exists() and json.loads((path/'result.json').read_text())['status']=='complete':
                entry['coverage'].add('evaluation')
        else:
            if read_records(path/'episodes.jsonl.gz'):entry['coverage'].add('training')
            if read_records(path/'validation.jsonl.gz'):entry['coverage'].add('validation')
            latest=Checkpoints(path).role('latest')
            if latest:
                inspect_checkpoint(latest);entry['coverage'].add('checkpoint')
                size=sum(f.stat().st_size for f in latest.rglob('*') if f.is_file())
                entry['checkpoint_headroom_bytes']=max(entry['checkpoint_headroom_bytes'],2*size)
                replay=latest/'controller/replay/index.json'
                if replay.exists():
                    r=json.loads(replay.read_text())
                    if r['size']==r['capacity']:entry['coverage'].add('replay_full')
    for entry in classes.values():entry['coverage']=sorted(entry['coverage'])
    if Path(output).exists():raise FileExistsError(output)
    atomic_json(output,dict(source=c['source'],runtime=runtime,classes=classes,
        note='Observed 5-second samples, not a proof of worst-case memory. Admission adds 1.3x RAM; unmeasured classes run alone.'))
    return dict(classes=len(classes),output=str(output))


def main(argv=None):
    if argv and argv[0] in ('dashboard','init-workflow','workflow-step'):
        from experiments.workflow import cli
        return cli(argv)
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--output',required=True)
    prep.add_argument('--validation-grid',required=True);prep.add_argument('--validation-monaco',required=True)
    prep.add_argument('--gate',required=True);prep.add_argument('--training-seeds',nargs='+',type=int)
    prep.add_argument('--run-root');prep.add_argument('--extend');prep.add_argument('--training-only',action='store_true')
    tune=sub.add_parser('prepare-tuning');tune.add_argument('--output',required=True);tune.add_argument('--run-root')
    tune.add_argument('--validation-grid',required=True);tune.add_argument('--validation-monaco',required=True)
    select=sub.add_parser('select-tuning');select.add_argument('--campaign',required=True);select.add_argument('--output',required=True)
    prune=sub.add_parser('prune-tuning');prune.add_argument('--campaign',required=True);prune.add_argument('--selection',required=True)
    measure=sub.add_parser('export-resources');measure.add_argument('--campaign',required=True);measure.add_argument('--output',required=True)
    for name in ('run','resume'):
        run=sub.add_parser(name);run.add_argument('--campaign',required=True);run.add_argument('--max-parallel',type=int,default=1)
        run.add_argument('--workflow',type=Path);run.add_argument('--resources');run.add_argument('--ram-gib',type=float);run.add_argument('--cpu-cores',type=float)
        run.add_argument('--disk-reserve-gib',type=float,default=10);run.add_argument('--segment-seconds',type=int,default=0)
    show=sub.add_parser('status');show.add_argument('--campaign',required=True)
    a=p.parse_args(argv)
    if a.command=='prepare-tuning':result=prepare_tuning(a.output,dict(grid=a.validation_grid,monaco=a.validation_monaco),a.run_root)
    elif a.command=='select-tuning':result=select_tuning(a.campaign,a.output)
    elif a.command=='prune-tuning':
        from experiments.tuning_cleanup import prune_tuning
        result=prune_tuning(a.campaign,a.selection)
    elif a.command=='export-resources':result=export_resources(a.campaign,a.output)
    elif a.command=='prepare':result=prepare(a.output,dict(grid=a.validation_grid,monaco=a.validation_monaco),a.gate,
                                           a.training_seeds,a.run_root,a.extend,not a.training_only)
    elif a.command=='status':result=status(a.campaign)
    else:result=run_queue(a.campaign,a.max_parallel,a.resources,a.ram_gib,a.cpu_cores,a.disk_reserve_gib,a.segment_seconds,a.workflow)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
