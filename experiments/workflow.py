"""Local workflow lifecycle, immutable tuning handoff and restartable phase execution."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from experiments.admission import FORMAL_SLOTS, TUNING_SLOTS
from experiments.core import file_hash
from experiments.dispatch import TRAIN_STAGES, permitted, read_json
from experiments.protocol import ROOT, root
from experiments.storage import atomic_json

PHASES = ('inventory', 'tuning', 'freeze', 'verify', 'formal', 'report')


def initialize(directory, design_seeds, resources=None):
    from experiments.campaign import prepare_tuning
    from experiments.runner import fingerprint
    from experiments.runtime import lock_path
    from experiments.validation import validation_manifest
    d = Path(directory).resolve()
    if d == ROOT or ROOT in d.parents:
        raise ValueError('Choose a workflow directory outside the source tree')
    if any(not 0 <= s < 2**31 or s == 9001 for s in design_seeds):
        raise ValueError('Formal design seeds must be uint31 and distinct from development seed 9001')
    if resources:
        measured=read_json(resources)
        if not measured or not measured.get('classes') or any('@' not in k for k in measured['classes']):
            raise ValueError('Workflow resources must be a config-bound measurements.json file')
    d.mkdir(parents=True, exist_ok=False)
    run_root = d/'runs'
    vals = {}
    for n, seed in zip(('grid','monaco'), design_seeds):
        vals[n] = str(d/f'{n}_dev.json')
        atomic_json(vals[n], validation_manifest(n,9001))
        atomic_json(d/f'{n}_validation.json', validation_manifest(n,seed,True))
    prepare_tuning(d/'tuning', vals, run_root)
    spec = dict(source=fingerprint(), run_root=str(run_root), assets=str(root('assets')),
                cache=str(root('cache')), data=str(root('data')), runtime_lock=str(lock_path()),
                resources=str(Path(resources).resolve()) if resources else None)
    atomic_json(d/'workflow.json', spec)
    atomic_json(d/'control.json',dict(auto=False, paused=False, stages=[],selected=[],held=[],retry={}))
    return dict(workflow=str(d), next='Open main.py local dashboard --workflow '+str(d))


def command(d, code, args, log):
    spec=read_json(d/'workflow.json')
    env=dict(os.environ, CBWCE_RUN_ROOT=spec['run_root'], CBWCE_ASSETS_ROOT=spec['assets'],
             CBWCE_DATA_ROOT=spec['data'], CBWCE_CACHE_ROOT=spec['cache'], CBWCE_RUNTIME_LOCK=spec['runtime_lock'])
    with log.open('a') as stream:
        subprocess.run([sys.executable,*map(str,args)],cwd=code,env=env,
                       stdout=stream,stderr=subprocess.STDOUT,check=True)


def freeze(d):
    from experiments.campaign import select_tuning
    from experiments.tuning_cleanup import prune_tuning
    winners=d/'selection'
    if not winners.exists():
        with tempfile.TemporaryDirectory(dir=d) as temp:
            tmp=Path(temp)/'selection'
            select_tuning(d/'tuning',tmp)
            tmp.rename(winners)
    # Verify the exact source tree with the selected INIs at its normal config paths.
    target=d/'frozen'
    record=read_json(winners/'selection.json')
    selected=record['selected']
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=d) as temp:
            tmp=Path(temp)/'frozen';tmp.mkdir()
            allow=read_json(ROOT/'packaging/release_allowlist.json')
            for name in allow['files']:
                shutil.copy2(ROOT/name,tmp/name)
            for name in allow['directories']:
                shutil.copytree(ROOT/name,tmp/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            for row in selected:
                shutil.copyfile(winners/row['config'],tmp/'config/revised'/row['config'])
            tmp.rename(target)
    for row in selected:
        if file_hash(target/'config/revised'/row['config']) != row['sha256']:
            raise ValueError('Frozen winner configuration mismatch')
    if record.get('basis')!='operator_locked_partial_tuning':
        prune_tuning(d/'tuning',winners)


def seed_parent_measurements(d, campaign):
    """Carry only measured winning configurations across the isolated INI overlay."""
    from experiments.runtime import identity
    measured=read_json(d/'tuning/measurements.json',{})
    if not measured or (d/'formal/measurements.json').exists():
        return
    runtime=identity()
    if read_json(d/'selection/selection.json').get('basis')=='operator_locked_partial_tuning':
        return
    if measured['source']!=read_json(d/'workflow.json')['source'] or measured['runtime']!=runtime:
        return
    classes={}
    for row in read_json(d/'selection/selection.json')['selected']:
        resource=f"{row['network']}/{row['controller']}/parent"
        key=resource+'@'+row['sha256']
        if key in measured['classes']:
            classes[resource+'@default']=measured['classes'][key]
    atomic_json(d/'formal/measurements.json',dict(source=campaign['source'],runtime=runtime,classes=classes))


def ensure_formal(d):
    from experiments.campaign import load_campaign,prepare
    spec=read_json(d/'workflow.json')
    if not (d/'formal/campaign.json').exists():
        prepare(d/'formal',{n:d/f'{n}_validation.json' for n in ('grid','monaco')},
                d/'gate.json',run_root=spec['run_root'])
    seed_parent_measurements(d,load_campaign(d/'formal'))


def run_phase(d, phase, attempt):
    from experiments.campaign import run_queue
    from experiments.runner import fingerprint
    from experiments.runtime import verify_gate, verify_runtime
    spec=read_json(d/'workflow.json');code=d/'frozen';log=attempt/'process.log'
    if phase in ('inventory','tuning','freeze') and fingerprint()!=spec['source']:
        raise ValueError('Development source changed since workflow initialization')
    if phase in ('verify','formal','report'):
        for row in read_json(d/'selection/selection.json')['selected']:
            if file_hash(code/'config/revised'/row['config'])!=row['sha256']:
                raise ValueError('Frozen configuration differs from selected winner')
    if phase=='inventory':
        command(d,ROOT,['scripts/inspect_models.py','--output',d/'inventory.json'],log)
    elif phase=='freeze':
        freeze(d)
    elif phase=='verify':
        verify_runtime()
        output=attempt/'verification'
        command(d,code,['scripts/verify.py','--extended','--output',output],log)
        gate=read_json(output/'gate.json')
        verify_gate(output/'gate.json')
        atomic_json(d/'gate.json',gate)
        ensure_formal(d)
    elif phase in ('formal','tuning'):
        if phase=='formal':
            verify_gate(d/'gate.json')
            ensure_formal(d)
        slots=TUNING_SLOTS if phase=='tuning' else FORMAL_SLOTS
        summary=run_queue(d/phase,max_parallel=slots,workflow=d,resources=spec.get('resources') if phase=='tuning' else None)
        if set(summary['counts'])!={'complete'}:
            raise ValueError('Queue needs attention: '+str(summary['counts']))
    elif phase=='report':
        report=attempt/'report'
        command(d,code,['scripts/report.py','--campaign',d/'formal','--output',report],log)
        if not (d/'report').exists():
            report.rename(d/'report')


def step(d,phase):
    """The phase process survives a dashboard restart; the lock guards duplicate launches."""
    base=d/'phases'/phase
    base.mkdir(parents=True,exist_ok=True)
    with (base/'owner.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        prev=read_json(base/'state.json',{})
        if prev.get('status')=='complete':return
        for upstream in PHASES[:PHASES.index(phase)]:
            if read_json(d/'phases'/upstream/'state.json',{}).get('status')!='complete':
                raise ValueError('Incomplete workflow prerequisite: '+upstream)
        if not phase_allowed(read_json(d/'control.json'),phase):return
        attempt=Path(tempfile.mkdtemp(prefix='attempt-',dir=base))
        control=read_json(d/'control.json')
        state=dict(status='running',started=time.time(),attempt=str(attempt),
                   retry=control.get('retry',{}).get(phase,0))
        atomic_json(base/'state.json',state)
        try:
            run_phase(d,phase,attempt)
        except Exception as exc:
            atomic_json(base/'state.json',dict(state,status='needs_attention',reason=str(exc),ended=time.time()))
            raise
        atomic_json(base/'state.json',dict(state,status='complete',ended=time.time()))


def phase_allowed(control,phase):
    if phase in ('tuning','formal'):
        stages=('tuning',) if phase=='tuning' else (*TRAIN_STAGES,'evaluate')
        selected=control.get('selected',[])
        return not control['paused'] and (any(permitted(control,s) for s in stages)
               or any(('/candidate_' in tid)==(phase=='tuning') for tid in selected))
    if phase in ('inventory','freeze','verify') and permitted(control,'tuning'):
        return True
    return permitted(control,phase)


def advance(d):
    """Start at most one phase. Called by the dashboard's single owner."""
    control=read_json(d/'control.json')
    for phase in PHASES:
        base=d/'phases'/phase;base.mkdir(parents=True,exist_ok=True)
        state=read_json(base/'state.json',{})
        if state.get('status')=='complete':continue
        if not phase_allowed(control,phase):return
        if state.get('status')=='needs_attention' and control.get('retry',{}).get(phase,0)<=state['retry']:
            return
        with (base/'owner.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return
        spec=read_json(d/'workflow.json')
        code=d/'frozen' if phase in ('verify','formal','report') else ROOT
        env=dict(os.environ,CBWCE_RUN_ROOT=spec['run_root'],CBWCE_RUNTIME_LOCK=spec['runtime_lock'],
                 CBWCE_DATA_ROOT=spec['data'],CBWCE_ASSETS_ROOT=spec['assets'],CBWCE_CACHE_ROOT=spec['cache'])
        try:
            with (base/'service.log').open('a') as log:
                return subprocess.Popen([sys.executable,'main.py','local','workflow-step','--workflow',str(d),'--phase',phase],
                                        cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except OSError as exc:
            atomic_json(base/'state.json',dict(status='needs_attention',reason=str(exc),
                                               retry=control['retry'].get(phase,0)))
            return None


def cli(argv):
    p=argparse.ArgumentParser(prog='main.py local '+argv[0])
    p.add_argument('--workflow',required=True,type=Path)
    if argv[0]=='init-workflow':
        p.add_argument('--design-seed-grid',type=int,required=True);p.add_argument('--design-seed-monaco',type=int,required=True)
        p.add_argument('--resources')
        a=p.parse_args(argv[1:]);print(json.dumps(initialize(a.workflow,[a.design_seed_grid,a.design_seed_monaco],a.resources),indent=2))
    elif argv[0]=='dashboard':
        from experiments.dashboard import serve
        p.add_argument('--port',type=int,default=8765)
        a=p.parse_args(argv[1:]);serve(a.workflow.resolve(),a.port)
    else:
        p.add_argument('--phase',required=True,choices=PHASES)
        a=p.parse_args(argv[1:]);step(a.workflow.resolve(),a.phase)
