"""Loopback-only, single-job training workspace. No arbitrary shell execution."""
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import mimetypes
import shlex
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, unquote
from experiments.core import ROOT, write_json
from experiments.protocol import settings, output_path, output_root, allowed_output

SITE=ROOT/'docs/site/dist'
RECORDS=ROOT/'runs_eval/revised/jobs'


def readable(path):
    p=Path(path)
    if not p.is_absolute():p=ROOT/p
    p=p.resolve()
    roots=[ROOT/x for x in ('runs','output_adversary','output_adversary_monaco','output_coevolution',
                           'output_coevolution_real','runs_eval','output_result','revision/verification',
                           'data_traffic/revised','real_net_subnet/demand_groups/revised')]
    if not any(r==p or r in p.parents for r in roots):raise ValueError('Path outside experiment records')
    if not p.exists():raise ValueError('Path does not exist')
    return p


def inventory():
    from experiments.checkpoint import inspect_checkpoint
    checkpoints=[];gates=[];runs=[]
    roots=[output_root(s,n,m) for s in ('parent','wce','continue','evaluate') for n in ('grid','monaco') for m in ('baseline','online_wce')]
    roots=list(set(roots))+[ROOT/'revision/verification']
    for root in roots:
        if not root.exists():continue
        for path in root.rglob('manifest.json'):
            try:
                m=json.loads(path.read_text())
                if 'signatures' in m:
                    # Full binary integrity is checked at launch, not on each listing.
                    checkpoints.append({'path':str(path.parent.relative_to(ROOT)),'signatures':m['signatures'],'parents':m.get('parents',{})})
                elif 'stage' in m:
                    result=path.parent/'result.json'
                    runs.append({'path':str(path.parent.relative_to(ROOT)),'network':m['network'],'controller':m['controller'],
                                 'method':m['method'],'stage':m['stage'],'seed':m['seed'],'pilot':m.get('pilot',False),
                                 'status':json.loads(result.read_text())['status'] if result.exists() else 'unfinished'})
            except (ValueError,KeyError,OSError):continue
        for path in root.rglob('*gate*.json'):
            try:
                g=json.loads(path.read_text())
                if 'corrections' in g:gates.append(str(path.relative_to(ROOT)))
            except (ValueError,OSError):pass
    return {'checkpoints':checkpoints,'gates':gates,'runs':runs}


def preflight(spec):
    from experiments.cli import stage_args
    from experiments.runner import require_gate
    from experiments.checkpoint import inspect_checkpoint
    allowed={'stage','network','controller','method','seed','pilot','visualization','parent','wce','resume','gate','steps','episodes','suite','scenario','rollouts','artifact','checkpoint_every','monitor_every','monitor_rollouts'}
    if not isinstance(spec,dict) or set(spec)-allowed:raise ValueError('Unknown job fields')
    stage=spec.get('stage','parent');p=settings()
    if stage not in ('parent','wce','continue','evaluate'):raise ValueError('Unsupported launch stage')
    network=spec.get('network');controller=spec.get('controller');method=spec.get('method','baseline')
    if network not in p['networks'] or controller not in p['controllers'] or method not in p['methods']:raise ValueError('Invalid selection')
    pilot=spec.get('pilot',True)
    if type(pilot) is not bool:raise ValueError('pilot must be boolean')
    visualization=spec.get('visualization',False)
    from envs.experiment_env import validate_visualization
    validate_visualization(visualization)
    seed=int(spec.get('seed',9001 if pilot else 101))
    if pilot and seed in p['training_seeds']:raise ValueError('Pilot seed overlaps publication seeds')
    if not pilot and seed not in p['training_seeds']:raise ValueError('Invalid publication seed')
    if not pilot:require_gate(str(readable(spec.get('gate',''))))
    tokens=['--network',network,'--controller',controller,'--method',method,'--seed',str(seed)]
    if pilot:tokens+=['--pilot']
    tokens+=['--visualization' if visualization else '--no-visualization']
    checkpoints={}
    for field in ('parent','wce','resume','gate','artifact'):
        if spec.get(field):
            path=readable(spec[field]);tokens+=['--'+field,str(path)]
            if field in ('parent','wce','resume'):
                from experiments.checkpoint import checkpoint_identity
                checkpoint_identity(path,network,controller,seed)
                manifest=inspect_checkpoint(path);checkpoints[field]=manifest
                role='wce' if field=='wce' else 'controller'
                signature=manifest['signatures'].get(role,{})
                if role == 'controller' and signature.get('reward') != 'learner_boundary_scaled_v3':raise ValueError('Historical reward protocol: start a fresh parent')
                if role == 'wce' and signature.get('protocol_version') != 6:raise ValueError('Historical WCE protocol: retrain against the new parent')
                if signature.get('network')!=network or (role=='controller' and signature.get('family')!=controller):
                    raise ValueError('Checkpoint network/controller mismatch')
    if stage!='parent' and 'parent' not in checkpoints:raise ValueError('Select an explicit controller checkpoint')
    if 'parent' in checkpoints:
        import pickle
        with (readable(spec['parent'])/'controller/state.pkl').open('rb') as f:controller_state=pickle.load(f)
        if not pilot and controller_state['learning_steps'] != (2320000 if stage=='evaluate' else 1000000):
            raise ValueError('Checkpoint does not have the exact required learning budget')
    if stage=='parent' and any(x in checkpoints for x in ('parent','wce')):raise ValueError('Fresh parent must not load a parent/WCE')
    if stage=='continue' and method in ('fixed_wce','online_wce'):
        if 'wce' not in checkpoints:raise ValueError('Select pretrained WCE')
        if checkpoints['wce']['parents'].get('controller',{}).get('hash')!=checkpoints['parent']['hash']:raise ValueError('WCE parent mismatch')
    elif 'wce' in checkpoints:raise ValueError('This stage does not use a pretrained WCE')
    if 'resume' in checkpoints:
        import pickle
        with (readable(spec['resume'])/'runner.pkl').open('rb') as f:state=pickle.load(f)
        if (state['stage'],state['method'])!=(stage,method):raise ValueError('Resume stage/method mismatch')
        for role in ('controller','wce'):
            if role in checkpoints['resume']['parents']:
                supplied='parent' if role=='controller' else 'wce'
                if checkpoints['resume']['parents'][role]['hash']!=checkpoints.get(supplied,{}).get('hash'):
                    raise ValueError('Resume parent checkpoint mismatch')
    if stage in ('parent','continue') and pilot:tokens+=['--steps',str(int(spec.get('steps',160 if stage=='parent' else 2640)))]
    if stage=='wce' and pilot:tokens+=['--episodes',str(int(spec.get('episodes',2)))]
    if not pilot and (spec.get('steps') or spec.get('episodes')):raise ValueError('Publication budgets are fixed')
    tokens+=['--checkpoint-every',str(int(spec.get('checkpoint_every',1 if pilot else 10)))]
    if stage=='evaluate' and not spec.get('artifact'):
        tokens+=['--suite',spec.get('suite','all'),'--rollouts',str(int(spec.get('rollouts',10)))]
        if spec.get('scenario'):tokens+=['--scenario',spec['scenario']]
    tokens+=['--monitor-every',str(int(spec.get('monitor_every',50))),'--monitor-rollouts',str(int(spec.get('monitor_rollouts',3)))]
    args=stage_args(stage,tokens)
    if args.monitor_every < 1 or not 1 <= args.monitor_rollouts <= 3:raise ValueError('Invalid monitoring settings')
    if not pilot and (args.monitor_every,args.monitor_rollouts)!=(50,3):raise ValueError('Publication monitoring requires 50 episodes and three rollouts')
    if args.checkpoint_every<1:raise ValueError('Checkpoint interval must be positive')
    if args.steps is not None and (args.steps<=0 or (stage=='continue' and args.steps%1320)):raise ValueError('Invalid pilot step budget')
    if args.episodes is not None and args.episodes<=0:raise ValueError('Invalid episode budget')
    if args.rollouts<1 or args.rollouts>10 or (not pilot and args.rollouts!=10):raise ValueError('Invalid rollout count')
    if 'resume' in checkpoints:
        goal=(args.steps if stage in ('parent','continue') and pilot else 1320*args.episodes if stage=='wce' and pilot else p['parent_steps'] if stage=='parent' else p['offline_episodes']*1320 if stage=='wce' else p['continuation_steps'])
        if state['goal']!=goal:raise ValueError('Resume budget differs from original attempt')
    output=Path(args.output)
    if not allowed_output(output):raise ValueError('Output outside permitted experiment roots')
    tokens+=['--output',str(output)]
    argv=[sys.executable,str(ROOT/'main.py'),'experiment',stage]+tokens
    return {'argv':argv,'command':' '.join(shlex.quote(x) for x in argv),'output':str(output.relative_to(ROOT)),
            'spec':spec,'stage':stage,'pilot':pilot}


class Jobs:
    def __init__(self):
        self.lock=threading.Lock();self.process=None;self.current=None
        RECORDS.mkdir(parents=True,exist_ok=True)
        # A restarted server records recovery state; it never relaunches work.
        for path in RECORDS.glob('*/started.json'):
            terminal=path.parent/'finished.json';recovery=path.parent/'recovery.json'
            if not terminal.exists() and not recovery.exists():
                write_json(recovery,{'status':'unmanaged_after_restart','message':'Inspect the original process; no automatic restart.'})
    def start(self,spec):
        with self.lock:
            if self.process and self.process.poll() is None:raise ValueError('Another job is active')
            for started in RECORDS.glob('*/started.json'):
                if (started.parent/'finished.json').exists():continue
                previous=json.loads(started.read_text());pid=previous.get('pid')
                cmdline=Path('/proc')/str(pid)/'cmdline'
                if cmdline.exists() and str(ROOT/'main.py').encode() in cmdline.read_bytes():
                    raise ValueError('An earlier dashboard job is still active; stop it before starting another')
            plan=preflight(spec);job_id=Path(plan['output']).name
            path=RECORDS/job_id;path.mkdir(exist_ok=False)
            write_json(path/'request.json',plan)
            log=(path/'console.log').open('x')
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TF_CPP_MIN_LOG_LEVEL='2',PYTHONWARNINGS='ignore')
            self.process=subprocess.Popen(plan['argv'],cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close();self.current=dict(id=job_id,path=str(path),output=plan['output'],started=time.time(),pid=self.process.pid,status='running')
            write_json(path/'started.json',self.current)
            return self.status()
    def status(self):
        if not self.current:return {'status':'idle'}
        state=dict(self.current);state['elapsed_seconds']=time.time()-state['started']
        code=self.process.poll()
        if code is not None:
            state['status']='stopped' if self.current.get('stop_requested') else ('complete' if code==0 else 'failed')
            terminal=Path(state['path'])/'finished.json'
            if not terminal.exists():write_json(terminal,dict(state,exit_code=code))
            state['elapsed_seconds']=json.loads(terminal.read_text())['elapsed_seconds']
        log=Path(state['path'])/'console.log'
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-24000));state['log']=f.read().decode(errors='replace')
        output=ROOT/state['output'];progress=output/'progress.jsonl'
        if progress.exists():
            lines=progress.read_text().splitlines()
            if lines:
                try:state['progress']=json.loads(lines[-1])
                except ValueError:pass
        checkpoints=sorted(output.glob('checkpoint_*/manifest.json'))
        state['checkpoint']=str(checkpoints[-1].parent.relative_to(ROOT)) if checkpoints else None
        state.pop('path',None)
        return state
    def stop(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.current['stop_requested']=True
                # Signal the Python owner; its finally block closes TraCI/SUMO.
                self.process.send_signal(signal.SIGTERM)
            return self.status()


def serve(port=8765):
    jobs=Jobs();token=secrets.token_urlsafe(32)
    class Server(ThreadingMixIn,HTTPServer):daemon_threads=True
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,value,mime='application/json'):
            data=json.dumps(value,allow_nan=False).encode() if mime=='application/json' else value
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        def local(self):
            hosts=['127.0.0.1:'+str(port),'localhost:'+str(port)]
            if self.headers.get('Host') not in hosts:raise ValueError('Invalid Host')
            origin=self.headers.get('Origin')
            if origin and origin not in ['http://'+h for h in hosts]:raise ValueError('Invalid Origin')
        def do_GET(self):
            try:
                self.local();route=urlparse(self.path).path
                if route=='/api/session':return self.send(200,{'token':token,'protocol':settings(),'mode':'local'})
                if route.startswith('/api/'):
                    if self.headers.get('X-Session-Token')!=token:raise ValueError('Invalid session')
                    if route=='/api/job':return self.send(200,jobs.status())
                    if route=='/api/runs':return self.send(200,inventory())
                    if route=='/api/check':
                        from experiments.cli import check
                        return self.send(200,check())
                    return self.send(404,{'error':'Unknown endpoint'})
                path=(SITE/unquote(route.lstrip('/') or 'index.html')).resolve()
                if SITE.resolve() not in path.parents or not path.is_file():return self.send(404,b'Not found','text/plain')
                return self.send(200,path.read_bytes(),mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
            except Exception as exc:self.send(400,{'error':str(exc)})
        def do_POST(self):
            try:
                self.local()
                if self.headers.get('X-Session-Token')!=token:raise ValueError('Invalid session')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<65536:raise ValueError('Invalid request size')
                body=json.loads(self.rfile.read(length).decode())
                route=urlparse(self.path).path
                if route=='/api/preflight':return self.send(200,preflight(body))
                if route=='/api/jobs':return self.send(201,jobs.start(body))
                if route=='/api/stop':return self.send(200,jobs.stop())
                if route=='/api/report':
                    from experiments.reporting import report
                    from experiments.protocol import run_id
                    roots=[readable(p) for p in body.get('inputs',[])]
                    if not roots:raise ValueError('Select explicit report inputs')
                    result=report(roots,output_root('report')/run_id(True))
                    result['data']=json.loads((Path(result['output'])/'dashboard.json').read_text())
                    return self.send(200,result)
                return self.send(404,{'error':'Unknown endpoint'})
            except Exception as exc:self.send(400,{'error':str(exc)})
    server=Server(('127.0.0.1',port),Handler)
    print('CB-WCE workspace: http://127.0.0.1:'+str(port),flush=True)
    try:server.serve_forever()
    finally:
        jobs.stop();server.server_close()
        if jobs.process and jobs.process.poll() is None:
            try:jobs.process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(jobs.process.pid,signal.SIGKILL)
