"""Explicit scenario manifests and independent frozen-model evaluation workers."""
import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from experiments.core import digest
from experiments.storage import atomic_json
from experiments.protocol import root

def seed(*parts):return int(digest(list(parts))[:8],16)%2147483646+1

def load_manifest(path,network,publication=False,split='validation'):
    if not path:raise ValueError('Explicit frozen validation manifest required')
    manifest=json.loads(Path(path).read_text())
    if manifest.get('version')!='8.1' or manifest.get('network')!=network or manifest.get('split')!=split:
        raise ValueError('Manifest version/network/split mismatch')
    if type(manifest.get('frozen')) is not bool:raise ValueError('frozen must be a JSON boolean')
    if publication and manifest['frozen'] is not True:raise ValueError('Publication requires frozen validation manifest')
    scenarios=manifest.get('scenarios',[])
    if not scenarios:raise ValueError('Validation scenario design has not been supplied')
    ids=[]
    for scene in scenarios:
        ids.append(scene['id']);cursor=0
        if scene.get('horizon')!=3600:raise ValueError('Validation/test horizon must be 3600 seconds')
        for b in scene['blocks']:
            if b['start']!=cursor or b['duration']<=0 or b['duration']%5:raise ValueError('Invalid scenario intervals')
            if 'weights' in b:
                w=np.asarray(b['weights'],dtype=float)
                if w.shape!=(11,) or not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1):raise ValueError('Invalid mixture weights')
            elif b.get('profile') not in ['N_to_S','S_to_N','W_to_E','E_to_W','NW_to_SE','SE_to_NW','SW_to_NE','NE_to_SW','Periphery_to_Center','Center_to_Periphery','Uniform']:
                raise ValueError('Unknown demand profile')
            for k in ('multiplier','sigma'):
                if k in b and (not np.isfinite(b[k]) or b[k]<0):raise ValueError('Invalid demand parameter')
            if 'sigma' in b and (type(scene.get('generation_seed')) is not int or not 0<=scene['generation_seed']<2**32):
                raise ValueError('Redistribution requires a uint32 generation_seed')
            cursor+=b['duration']
        if cursor!=3600:raise ValueError('Scenario must cover 3600 seconds')
    if len(set(ids))!=len(ids):raise ValueError('Duplicate scenario IDs')
    return manifest,digest(manifest)

def traffic(env,scene,manifest_hash,index):
    from experiments.demand import materialize
    from experiments.scenarios import block_rows
    key=dict(version='8.1',generator='materialize_v1',network=env.asset_hashes,
             profiles=digest(env.groups),scene=scene,manifest=manifest_hash,realization=index)
    folder=root('data')/env.network/'traffic';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(digest(key)+'.json.gz')
    if path.exists():
        with gzip.open(str(path),'rt') as f:data=json.load(f)
        if data['key']!=key or digest(data['payload'])!=data['hash']:raise ValueError('Corrupt traffic artifact')
    else:
        rng=np.random.RandomState(seed(manifest_hash,scene['id'],index,'arrival'))
        vehicles=[]
        for i,b in enumerate(scene['blocks']):
            vehicles.extend(materialize(block_rows(env.groups,scene,b),b['start'],b['duration'],rng,env.resolve_route,'b%d'%i))
        routes={};compact=[]
        for v in vehicles:
            edges=v['edges'];rid=digest(edges);routes[rid]=edges
            compact.append(dict(id=v['id'],depart=v['depart'],route=rid,speed_factor=v['speed_factor']))
        payload=dict(routes=routes,vehicles=compact);data=dict(key=key,payload=payload,hash=digest(payload))
        fd,tmp=tempfile.mkstemp(dir=str(folder),suffix='.partial');os.close(fd)
        try:
            with gzip.open(tmp,'wt',compresslevel=1) as f:json.dump(data,f,sort_keys=True)
            # Same key has deterministic content; atomic concurrent producers are equivalent.
            os.replace(tmp,str(path))
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    payload=data['payload']
    vehicles=[dict(id=v['id'],depart=v['depart'],edges=payload['routes'][v['route']],speed_factor=v['speed_factor']) for v in payload['vehicles']]
    return vehicles,data['hash']

def indexed_demand(rows,episode,block,random_seed,env):
    """Independent count/time/speed streams for each OD, not action-dependent draws."""
    vehicles=[]
    for origin,destination,rate in rows:
        edges=env.resolve_route(origin,destination)
        key=(random_seed,episode,block,origin,destination)
        count=np.random.RandomState(seed(*key,'count')).poisson(rate/6.)
        rng=np.random.RandomState(seed(*key,'time'))
        jitter=np.random.RandomState(seed(*key,'jitter'))
        times=np.clip(rng.uniform(block*600,(block+1)*600,count)+jitter.normal(0,2,count),block*600+.01,(block+1)*600-.01)
        speeds=np.random.RandomState(seed(*key,'speed'))
        for departure in sorted(times):
            speed=float(speeds.normal(1.,.1))
            while speed<=0:speed=float(speeds.normal(1.,.1))
            vehicles.append(dict(id='b%d_%d'%(block,len(vehicles)),depart=round(float(departure),2),edges=edges,speed_factor=speed))
    return sorted(vehicles,key=lambda v:(v['depart'],v['id']))

def worker(request):
    from agents.controller import Controller
    from agents.wce import WCE
    from envs.experiment_env import make_environment
    from experiments.checkpoint import load_checkpoint,inspect_checkpoint
    from experiments.demand import mixture
    rows=[]
    with tempfile.TemporaryDirectory(prefix='cbwce-validation-',dir=os.environ.get('TMPDIR')) as work:
        env=make_environment(request['network'],request['controller'],work,request['seed'],
                             config_path=request.get('config'),audit=request.get('audit',False))
        ctrl=wce=None
        try:
            env.reset_episode(1,evaluation=True)
            ctrl=Controller(env,request['controller'],request['seed'],training=False)
            is_wce=request.get('stage')=='wce'
            parent=request['parent'] if is_wce else request['checkpoint']
            load_checkpoint(parent,{'controller':ctrl},inference=True)
            if is_wce:
                wce=WCE(env,request['seed'],training=False);load_checkpoint(request['checkpoint'],{'wce':wce},inference=True)
                scenes=[dict(id='wce',horizon=6600)]
            else:scenes=request['manifest']['scenarios']
            for scene in scenes:
                horizon=request.get('pilot_horizon',scene['horizon'])
                block_seconds=request.get('pilot_block_seconds',600)
                for index in range(request.get('start',0),request['repeats']):
                    namespace=request['manifest_hash']
                    rs=seed(namespace,scene['id'],index,'sumo')
                    obs=env.reset_episode(rs,evaluation=True,horizon=horizon);ctrl.reset()
                    ctrl.rng.rng['policy']=np.random.RandomState(seed(namespace,scene['id'],index,'policy'))
                    if wce:wce.rng.rng['wce']=np.random.RandomState(seed(namespace,index,'wce'))
                    hashes=[]
                    if not is_wce:
                        vehicles,h=traffic(env,scene,namespace,index);env.inject(vehicles);hashes.append(h)
                    for step in range(horizon//5):
                        if is_wce and step%(block_seconds//5)==0:
                            weights,_,_=wce.act(env.wce_observation())
                            vehicles=indexed_demand(mixture(env.groups,weights),0,step//120,seed(namespace,index,'traffic'),env) if block_seconds==600 else __import__('experiments.demand',fromlist=['materialize']).materialize(mixture(env.groups,weights),step*5,block_seconds,np.random.RandomState(seed(namespace,index,step,'traffic')),env.resolve_route,'b%d'%step)
                            env.inject(vehicles);hashes.append(digest(vehicles))
                        decision=ctrl.act(obs,env,False);obs,reward,done=env.step(decision[0])
                        ctrl.previous_done=done
                    result=env.metrics.summary(horizon,detail=request.get('detail',False))
                    result.update(scenario=scene['id'],realization=index,demand_hash=digest(hashes),scheduled=env.scheduled,sumo_seed=rs)
                    rows.append(result)
                    if env.audit and request.get('audit_output'):
                        env.export_episode(Path(request['audit_output'])/(digest([scene['id'],index])[:16]))
            scores=[np.mean([r['mean_queue'] for r in rows if r['scenario']==s['id']]) for s in scenes]
            return dict(score=float(np.mean(scores)),rollouts=rows,checkpoint_hash=inspect_checkpoint(request['checkpoint'])['hash'])
        finally:
            if ctrl:ctrl.close()
            if wce:wce.close()
            env.terminate()

def evaluate(request):
    """Spawn, wait and reclaim inference graph memory between validation rounds."""
    with tempfile.TemporaryDirectory(prefix='cbwce-worker-',dir=os.environ.get('TMPDIR')) as d:
        inp=Path(d)/'request.json';out=Path(d)/'result.json';atomic_json(inp,request)
        command=[sys.executable,'-m','experiments.validation',str(inp),str(out)]
        subprocess.run(command,check=True)
        return json.loads(out.read_text())

if __name__=='__main__':
    atomic_json(sys.argv[2],worker(json.loads(Path(sys.argv[1]).read_text())))
