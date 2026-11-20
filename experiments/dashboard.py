"""Loopback-only English workflow console. POSTs change intent; the queue owns task state."""
import csv
import fcntl
import json
import mimetypes
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from experiments.admission import cpu_capacity, memory_info, TUNING_SLOTS, FORMAL_SLOTS
from experiments.curves import read_curves
from experiments.dispatch import STAGES, read_json, stage, task_number
from experiments.storage import atomic_json, read_gzip_json
from experiments.workflow import PHASES, advance

REPORTS=('seeds.csv','comparison.csv','paired_differences.csv','paired_training_seeds.csv','auxiliary_metrics.csv','summary.json')
WEB=Path(__file__).with_name('dashboard_web')


def catalog(d):
    tasks={}
    for phase in ('tuning','formal'):
        manifest=read_json(d/phase/'campaign.json',{})
        for t in manifest.get('tasks',[]):
            tasks[t['id']]=(phase,t)
    return tasks


def change(d,action,ids,group):
    tasks=catalog(d)
    if not isinstance(ids,list) or any(not isinstance(x,str) or x not in tasks for x in ids):
        raise ValueError('Unknown task selection')
    if action not in ('auto','stage','run','pause','resume','retry','retry-phase'):
        raise ValueError('Unknown action')
    with (d/'.control.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        c=read_json(d/'control.json')
        held=set(c['held']);selected=set(c['selected'])
        if action=='auto':
            c.update(auto=True,paused=False)
        elif action=='stage':
            if group not in STAGES:raise ValueError('Unknown workflow stage')
            c.update(auto=False,paused=False,stages=[group])
            selected.clear()
            held.difference_update(tid for tid,(_,t) in tasks.items() if stage(t)==group)
        elif action=='run':
            if not ids:raise ValueError('Select at least one task')
            c.update(auto=False,paused=False,stages=[])
            selected=set(ids);held.difference_update(ids)
        elif action=='pause':
            if ids:held.update(ids)
            else:c['paused']=True
        elif action=='resume':
            if ids:selected.update(ids);held.difference_update(ids)
            else:c['paused']=False
        elif action=='retry-phase':
            if group not in PHASES:raise ValueError('Unknown workflow phase')
            c['retry'][group]=time.time()
        elif action=='retry':
            if not ids:raise ValueError('Select failed tasks to retry')
            for tid in ids:
                phase,_=tasks[tid]
                row=read_json(d/phase/'state.json',{}).get('tasks',{}).get(tid,{})
                if row.get('status')!='failed':raise ValueError('Retry requires a failed task')
                c['retry'][tid]=c['retry'][phase]=time.time()
            selected.update(ids);held.difference_update(ids)
        c.update(selected=sorted(selected),held=sorted(held))
        atomic_json(d/'control.json',c)
    return c


def snapshot(d):
    now=time.time();tasks=[];pool=cpu_capacity()
    total,available=memory_info()
    formal=(d/'formal/campaign.json').is_file()
    slots=FORMAL_SLOTS if formal else TUNING_SLOTS
    pool.update(cpu_budget=min(pool['default_budget'],slots),available_ram=available,ram_budget=total*.8,max_parallel=slots)
    for name in ('tuning','formal'):
        campaign=read_json(d/name/'campaign.json',{})
        state=read_json(d/name/'state.json',{})
        if name==('formal' if formal else 'tuning'):
            pool.update(state.get('pool',{}))
        for t in campaign.get('tasks',[]):
            tid=t['id'];r=state.get('tasks',{}).get(tid,{})
            output=Path(t['output']);status=r.get('status','pending')
            progress=read_json(output/'progress.json',{}) if t['kind']=='train' else {}
            result=read_json(output/'result.json',{})
            quantity=progress.get('episode',r.get('progress',0))
            if t['kind']=='evaluate':
                quantity=result.get('rollouts',0)
                if status=='running' or result.get('status')=='interrupted':
                    quantity=len(list((output/'rows').glob('*.json.gz')))
            elif status not in ('pending','running'):
                quantity=result.get('completed_episodes',quantity)
            limit=(10 if t['kind']=='evaluate' else 1000 if name=='tuning' else campaign['protocol']['max_episodes'][t['stage']])
            display=(r.get('wait') or status) if status=='pending' else status
            if status=='running' and r.get('reason','').startswith('Pausing'):display='pausing'
            tasks.append(dict(id=tid,number=task_number(t),stage=stage(t),network=t['network'],
                algorithm=t['controller'],selection=t.get('selection'),method=t.get('scenario',t['method']),status=display,
                quantity=quantity,limit=limit,best=progress.get('best_score'),best_episode=result.get('best_episode',progress.get('best_episode')),
                latest=progress.get('validation_episode'),reason=r.get('reason') or r.get('error') or result.get('error',''),
                deps=t['dependencies'],cpu=r.get('cpu_cores',0) if status=='running' else 0,
                rss=r.get('rss_bytes',0) if status=='running' else 0,
                elapsed=(r.get('ended',now) if status!='running' else now)-r['started'] if 'started' in r else 0,
                updated=progress.get('updated',r.get('ended',r.get('started'))),score=result.get('score')))
    tasks.sort(key=lambda t:(0 if t['stage']=='tuning' else 1 if t['number'] else 2,t['number'],t['id']))
    pool['available_ram']=available
    return dict(workflow=d.name,tasks=tasks,pool=pool,now=now,control=read_json(d/'control.json'),
                phases=[dict(name=k,**read_json(d/'phases'/k/'state.json',{'status':'pending'})) for k in PHASES],
                reports=[r for r in REPORTS if (d/'report'/r).exists()])


def detail(d,tid):
    phase,t=catalog(d)[tid];out=Path(t['output'])
    state=read_json(d/phase/'state.json')['tasks'][tid]
    curves=read_curves(out) if t['kind']=='train' else dict(training=[],validation=[],live={})
    log=''
    if state.get('log'):
        with Path(state['log']).open('rb') as f:
            f.seek(max(0,f.seek(0,2)-4000));log=f.read().decode(errors='replace')
    rollouts=[]
    if t['kind']=='evaluate':
        for shard in sorted((out/'rows').glob('*.json.gz')):
            r=read_gzip_json(shard)['row']
            rollouts.append(dict(scenario=r['scenario'],realization=r['realization'],mean_queue=r['mean_queue']))
    return dict(history=curves['validation'],curves=curves,log=log,result=read_json(out/'result.json',{}),output=str(out),rollouts=rollouts)


def serve(d,port):
    if not (d/'workflow.json').exists():raise ValueError('Initialize the workflow first')
    token=secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def reply(self,code,body,kind='application/json'):
            if kind=='application/json':body=json.dumps(body,allow_nan=False).encode()
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def host_ok(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
        def do_GET(self):
            if not self.host_ok():return self.reply(403,{'error':'Loopback Host required'})
            u=urlparse(self.path)
            try:
                if u.path in ('/','/style.css','/app.js'):
                    file=WEB/('index.html' if u.path=='/' else u.path[1:])
                    return self.reply(200,file.read_bytes(),mimetypes.guess_type(file)[0]+'; charset=utf-8')
                if u.path=='/api/state':return self.reply(200,dict(snapshot(d),token=token))
                if u.path=='/api/task':return self.reply(200,detail(d,parse_qs(u.query)['id'][0]))
                if u.path=='/api/results':
                    with (d/'report/comparison.csv').open() as f:return self.reply(200,list(csv.DictReader(f)))
                if u.path.startswith('/reports/') and u.path[9:] in REPORTS:
                    file=d/'report'/u.path[9:]
                    return self.reply(200,file.read_bytes(),'text/plain' if file.suffix=='.json' else 'text/csv')
                self.reply(404,{'error':'Not found'})
            except (OSError,EOFError,ValueError,KeyError) as exc:self.reply(400,{'error':str(exc)})
        def do_POST(self):
            origin=self.headers.get('Origin','')
            if (self.path!='/api/control' or not self.host_ok() or
                    (origin and origin!='http://'+self.headers['Host']) or self.headers.get('X-Workflow-Token')!=token):
                return self.reply(403,{'error':'Invalid local control request'})
            try:
                n=int(self.headers.get('Content-Length',0))
                if not 0<n<=128000:raise ValueError('Invalid request length')
                req=json.loads(self.rfile.read(n))
                self.reply(200,change(d,req['action'],req.get('ids',[]),req.get('stage')))
            except (ValueError,KeyError,TypeError) as exc:self.reply(400,{'error':str(exc)})
        def log_message(self,*args):pass
    with (d/'.console.lock').open('a') as lock,ThreadingHTTPServer(('127.0.0.1',port),Handler) as server:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(f'http://127.0.0.1:{server.server_port}',flush=True)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        child=None
        try:
            while True:
                if child is None or child.poll() is not None:
                    child=advance(d)
                time.sleep(1)
        except KeyboardInterrupt:pass
        finally:server.shutdown()
