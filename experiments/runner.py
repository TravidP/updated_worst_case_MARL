"""Single validation-driven stage loop with bounded data and segmented recovery."""
import argparse
import copy
import fcntl
import json
import os
import resource
import signal
import tempfile
import time
from pathlib import Path
import numpy as np
from experiments.core import Streams, METHODS, digest, file_hash
from experiments.protocol import settings, scientific_protocol, training_seeds, root, ROOT
from experiments.storage import atomic_json, append_record
from experiments.selection import Selection
from experiments.checkpoint import Checkpoints, inspect_checkpoint, load_checkpoint, pin_external
from experiments.validation import load_manifest, evaluate, indexed_demand
from experiments.dispatch import pause_requested
from experiments.metrics import Diagnostics, RewardMean

def initialize_continuation(controller, checkpoint):
    """Seed a newly constructed learner with the selected parent's policy weights."""
    load_checkpoint(checkpoint, {'controller': controller}, inference=True)
    if controller.family == 'dql':
        controller.sess.run(controller.sync)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['parent','wce','continue'],required=True)
    p.add_argument('--network',choices=['grid','monaco'],required=True)
    p.add_argument('--controller',choices=['ma2c','dql','ppo'],required=True)
    p.add_argument('--seed',type=int);p.add_argument('--method',choices=METHODS,default='baseline')
    p.add_argument('--output');p.add_argument('--validation',required=True)
    p.add_argument('--parent');p.add_argument('--wce');p.add_argument('--resume',action='store_true')
    p.add_argument('--pilot',action='store_true');p.add_argument('--gate');p.add_argument('--config')
    p.add_argument('--max-episodes',type=int);p.add_argument('--validate-every',type=int)
    p.add_argument('--validation-repeats',type=int);p.add_argument('--pilot-block-seconds',type=int)
    p.add_argument('--pilot-stop-after',type=int,help='Correctness check: interrupt at this complete episode boundary')
    p.add_argument('--segment-seconds',type=int,default=23*3600)
    p.add_argument('--audit',action='store_true')
    p.add_argument('--campaign',help='Frozen local campaign; authorizes its explicit training seed list')
    return p

def fingerprint():
    files=[]
    for folder in ('agents','envs','experiments','config','assets','scripts'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.extend([ROOT/'main.py',ROOT/'tf_compat.py'])
    return digest({str(p.relative_to(ROOT)):file_hash(p) for p in sorted(files)})

def preflight(args):
    defaults=settings()
    if args.seed is None:args.seed=defaults['pilot_seed'] if args.pilot else training_seeds()[0]
    allowed=training_seeds()
    if getattr(args,'campaign',None):
        from experiments.campaign import load_campaign
        campaign=load_campaign(args.campaign)
        if bool(args.pilot)!=(campaign.get('purpose')=='development'):
            raise ValueError('Campaign pilot/publication mode mismatch')
        if not args.pilot:allowed=campaign['training_seeds']
        if campaign['validation'][args.network]['hash'] != load_manifest(args.validation,args.network,not args.pilot)[1]:
            raise ValueError('Campaign validation identity mismatch')
    p=scientific_protocol();manifest,mhash=load_manifest(args.validation,args.network,not args.pilot)
    if args.pilot:
        if args.pilot_stop_after is not None and args.pilot_stop_after<1:raise ValueError('Positive pilot stop episode required')
        if args.seed in allowed:raise ValueError('Pilot and publication training seeds must differ')
        if args.max_episodes is not None:
            if args.max_episodes<1:raise ValueError('Positive episode budget required')
            p['max_episodes'][args.stage]=args.max_episodes
        if args.validate_every is not None:
            if args.validate_every<1:raise ValueError('Positive validation interval required')
            p['validation_every']=args.validate_every
        if args.validation_repeats is not None:
            if args.validation_repeats<1:raise ValueError('Positive repeats required')
            p['validation_repeats']=p['wce_validation_repeats']=args.validation_repeats
        if args.pilot_block_seconds is not None:
            if args.pilot_block_seconds<5 or args.pilot_block_seconds%5:raise ValueError('Pilot block must be a positive multiple of five')
            p['demand_seconds']=args.pilot_block_seconds;p['training_seconds']=11*args.pilot_block_seconds
    else:
        if args.seed not in allowed:raise ValueError('Unprescribed publication seed')
        if any(getattr(args,k) is not None for k in ('max_episodes','validate_every','validation_repeats','pilot_block_seconds','pilot_stop_after','config')):
            raise ValueError('Publication protocol cannot be overridden')
        if not args.gate:raise ValueError('Publication requires a current verification gate')
        from experiments.runtime import verify_gate
        verify_gate(args.gate)
    if args.segment_seconds<0:raise ValueError('Segment time must be nonnegative; zero disables segmentation')
    if args.stage!='continue' and args.method!='baseline':raise ValueError('Method applies only to continuation')
    if args.stage!='parent' and not args.parent:raise ValueError('Explicit best parent required')
    if args.stage=='continue' and args.method in ('fixed_wce','online_wce') and not args.wce:
        raise ValueError('Explicit shared best WCE required')
    for path,kind in [(args.parent,'controller'),(args.wce,'wce')]:
        if not path:continue
        cp=inspect_checkpoint(path)
        if cp['kind']!='full' or kind not in cp['signatures']:raise ValueError('Stage branch requires full best state')
        origin=Path(path).resolve().parent.parent
        identity=json.loads((origin/'manifest.json').read_text())
        if identity['stage']!=('parent' if kind=='controller' else 'wce'):
            raise ValueError('Incorrect parent stage')
        if (identity['network'],identity['controller'],identity['seed'])!=(args.network,args.controller,args.seed):
            raise ValueError('Parent network/controller/seed mismatch')
        if identity['pilot']!=args.pilot:raise ValueError('Pilot/publication ancestry mismatch')
        if not args.pilot:
            if identity['source']!=fingerprint() or identity['protocol']!=p or identity['validation_hash']!=mhash:
                raise ValueError('Publication ancestry source/protocol mismatch')
            result=json.loads((origin/'result.json').read_text())
            if (result['status']!='max_budget_reached' or result['completed_episodes']!=p['max_episodes'][identity['stage']]) or Checkpoints(origin).role('best').resolve()!=Path(path).resolve():
                raise ValueError('Stage branch requires completed horizon and its validation best')
    if args.wce and inspect_checkpoint(args.wce)['parents'].get('controller',{}).get('hash')!=inspect_checkpoint(args.parent)['hash']:
        raise ValueError('WCE belongs to a different parent')
    return p,manifest,mhash

class Interrupted(Exception):pass

def run(args):
    protocol,manifest,mhash=preflight(args)
    if not args.output:
        if args.resume:raise ValueError('Resume requires an explicit output directory')
        args.output=str(root('run')/args.network/args.controller/('seed_%d'%args.seed)/args.stage/os.urandom(8).hex())
    run=Path(args.output).resolve()
    if args.resume:
        if not run.is_dir():raise ValueError('Resume requires the original run directory')
    else:run.mkdir(parents=True,exist_ok=False)
    with (run/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another process owns this run')
        try:return _run(args,run,protocol,manifest,mhash)
        except BaseException as exc:
            if not (run/'result.json').exists():atomic_json(run/'result.json',dict(status='failed',error=repr(exc)))
            raise

def _run(args,run,p,manifest,mhash):
    from agents.controller import Controller
    from agents.wce import WCE
    from envs.experiment_env import make_environment
    from experiments.demand import mixture,materialize
    parents={}
    for name,path in [('controller',args.parent),('wce',args.wce)]:
        if path:parents[name]=pin_external(path)
    from experiments.runtime import identity as runtime_identity
    identity=dict(runtime=runtime_identity(),protocol=p,network=args.network,controller=args.controller,seed=args.seed,stage=args.stage,
                  method=args.method,pilot=args.pilot,validation_hash=mhash,parents=parents,
                  source=fingerprint(),config_hash=file_hash(args.config) if args.config else None)
    if args.resume:
        if json.loads((run/'manifest.json').read_text())!=identity:raise ValueError('Resume identity mismatch')
        if (run/'result.json').exists() and json.loads((run/'result.json').read_text())['status'] not in ('interrupted','failed'):
            raise ValueError('Run is already terminal')
    else:atomic_json(run/'manifest.json',identity)
    store=Checkpoints(run);streams=Streams(args.seed);selection=Selection(args.stage,p)
    episode=0;initial_steps=0;stage_steps=0;best_full=None;max_episode_seconds=0.;max_checkpoint_seconds=0.;total_seconds=0.
    attempt=os.urandom(8).hex()
    ctrl=wce=None;started=time.monotonic();status='interrupted';stop={'requested':False,'force':False}
    def interrupt(signum,frame):
        stop['force']=stop['requested'];stop['requested']=True
    handlers={sig:signal.signal(sig,interrupt) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1)}
    with tempfile.TemporaryDirectory(prefix='cbwce-train-',dir=os.environ.get('TMPDIR')) as runtime:
        env=None
        try:
            env=make_environment(args.network,args.controller,runtime,args.seed,config_path=args.config,audit=args.audit)
            obs=env.reset_episode(1,horizon=p['training_seconds'])
            ctrl=Controller(env,args.controller,args.seed,training=args.stage!='wce')
            if args.stage=='wce':
                parent_state=load_checkpoint(args.parent,{'controller':ctrl},inference=True)
                ctrl.learning_steps=parent_state['initial_steps']+parent_state['stage_steps']
            elif args.stage=='continue' and not args.resume:
                initialize_continuation(ctrl,args.parent)
            initial_steps=ctrl.learning_steps
            use_wce=args.stage=='wce' or (args.stage=='continue' and args.method in ('fixed_wce','online_wce'))
            if use_wce:
                wce=WCE(env,args.seed,training=args.stage=='wce' or args.method=='online_wce')
                if args.wce:load_checkpoint(args.wce,{'wce':wce},inference=args.method=='fixed_wce')
            models={}
            if ctrl.training:models['controller']=ctrl
            if wce and wce.training:models['wce']=wce
            if args.resume:
                latest=store.role('latest')
                if not latest:raise ValueError('No complete recovery point')
                state=load_checkpoint(latest,models)
                if not ctrl.training:
                    if 'frozen_controller' not in state:raise ValueError('Missing frozen controller RNG state')
                    ctrl.restore_state(state['frozen_controller'])
                if wce and not wce.training:
                    if 'frozen_wce' not in state:raise ValueError('Missing frozen WCE RNG state')
                    wce.restore_state(state['frozen_wce'])
                streams.restore(state['streams']);selection=Selection(args.stage,p,state['selection'])
                episode=state['episode'];stage_steps=state['stage_steps'];initial_steps=state['initial_steps']
                best_full=state['best_full'];max_episode_seconds=state['max_episode_seconds'];total_seconds=state['total_seconds']
                max_checkpoint_seconds=state.get('max_checkpoint_seconds',0.)
                store.update('best',best_full)
                store.update('best_inference',selection.best['checkpoint'] if selection.best else None)
                status=selection.terminal() or 'interrupted'
            append_record(run/'attempts.jsonl',dict(attempt=attempt,resume_episode=episode,started_unix=time.time()))
            diag=Diagnostics()
            def state():
                return dict(episode=episode,stage_steps=stage_steps,initial_steps=initial_steps,
                            streams=streams.state(),selection=selection.state(),best_full=best_full,
                            max_episode_seconds=max_episode_seconds,max_checkpoint_seconds=max_checkpoint_seconds,
                            total_seconds=total_seconds+time.monotonic()-started,
                            frozen_controller=ctrl.state() if not ctrl.training else None,
                            frozen_wce=wce.state() if wce and not wce.training else None)
            def save(kind,label):
                nonlocal max_checkpoint_seconds
                checkpoint_started=time.monotonic()
                name='e%06d_%s_%s'%(episode,label,os.urandom(4).hex())
                store.publish(name,models,state(),kind,parents)
                max_checkpoint_seconds=max(max_checkpoint_seconds,time.monotonic()-checkpoint_started)
                return name
            def request(name,repeats,start=0):
                req=dict(stage=args.stage,network=args.network,controller=args.controller,seed=args.seed,
                         checkpoint=str(store.path/name),parent=str(Path(args.parent).resolve()) if args.parent else None,
                         config=str(Path(args.config).resolve()) if args.config else None,manifest=manifest,
                         manifest_hash=mhash,repeats=repeats,start=start)
                if args.pilot and args.pilot_block_seconds:
                    req['pilot_horizon']=p['training_seconds']
                    req['pilot_block_seconds']=p['demand_seconds']
                return evaluate(req)
            live_blocks=[];live_episode=None
            def progress(phase):
                atomic_json(run/'progress.json', dict(episode=episode, phase=phase,
                    best_episode=selection.best['episode'] if selection.best else None,
                    best_score=selection.best['score'] if selection.best else None,
                    validation_episode=selection.history[-1]['episode'] if selection.history else None,
                    updated=time.time(),live_episode=live_episode,block_queues=live_blocks))
            def validate():
                nonlocal best_full
                progress('validation')
                snapshot=save('inference','validation')
                repeats=p['wce_validation_repeats'] if args.stage=='wce' else p['validation_repeats']
                result=request(snapshot,repeats)
                improved=selection.add(episode,result['score'],snapshot)
                append_record(run/'validation.jsonl.gz',dict(attempt=attempt,episode=episode,kind='routine',stopping_diagnostics=selection.diagnostics(),**result))
                store.update(window=True,name=snapshot)
                if improved:
                    best_full=save('full','best');store.update('best',best_full)
                    store.update('best_inference',snapshot)
                if selection.early_stop and selection.early_stop['trigger_episode']==episode:
                    store.update('early_best',best_full)
                    store.update('early_best_inference',selection.best['checkpoint'])
                if episode==p['max_episodes'][args.stage]:
                    store.update('final_inference',snapshot)
                recovery=save('full','recovery');store.update(name=recovery,latest=True)
                store.collect()
                return selection.terminal()

            if not selection.history:status=validate() or 'interrupted'
            while episode<p['max_episodes'][args.stage] and status=='interrupted':
                if args.pilot_stop_after is not None and episode>=args.pilot_stop_after:break
                if stop['requested'] or pause_requested() or (args.segment_seconds and time.monotonic()-started+max(600.,2*max_episode_seconds+2*max_checkpoint_seconds)>=args.segment_seconds):break
                live_episode=episode+1;live_blocks=[]
                controller_reward=RewardMean()
                demand_reward=RewardMean()
                progress('training')
                episode_start=time.monotonic();obs=env.reset_episode(streams['sumo'].randint(1,2147483647),horizon=p['training_seconds']);ctrl.reset()
                for block in range(11):
                    wobs=logits=value=None
                    if use_wce:
                        wobs=env.wce_observation();weights,logits,value=wce.act(wobs)
                    elif args.stage=='parent' or args.method=='baseline':weights=np.eye(11)[block]
                    elif args.method=='random_group':weights=np.eye(11)[streams['selection'].randint(11)]
                    else:weights=streams['selection'].dirichlet(np.ones(11))
                    rates=mixture(env.groups,weights)
                    if use_wce and p['demand_seconds']==600:
                        vehicles=indexed_demand(rates,episode,block,args.seed,env)
                    else:vehicles=materialize(rates,block*p['demand_seconds'],p['demand_seconds'],streams['demand'],env.resolve_route,'b%d'%block)
                    env.inject(vehicles);before=env.metrics.total
                    for _ in range(p['demand_seconds']//5):
                        if stop['force']:raise Interrupted('Second signal during an episode; rolling back to durable checkpoint')
                        decision=ctrl.act(obs,env,ctrl.training);nxt,rewards,done=env.step(decision[0])
                        learner,_=ctrl.observe(obs,decision,rewards,nxt,done,ctrl.training)
                        if ctrl.training:controller_reward.add(rewards,learner)
                        obs=nxt;stage_steps+=1
                        diag.add(ctrl.diagnostics)
                    block_reward=(env.metrics.total-before)/p['demand_seconds']
                    if wce and wce.training:
                        learner,_=wce.observe(wobs,logits,value,block_reward)
                        demand_reward.add(block_reward,learner)
                        diag.add(wce.diagnostics)
                    live_blocks.append(dict(end_seconds=(block+1)*p['demand_seconds'],mean_queue=block_reward))
                    progress('training')
                    append_record(run/'blocks.jsonl.gz',dict(attempt=attempt,episode=episode+1,block=block,weights=weights.tolist(),
                        traffic_hash=digest(vehicles),scheduled=len(vehicles),mean_queue=block_reward,diagnostics=diag.flush()))
                ctrl.flush(obs,True);ctrl.assert_finite();episode+=1
                elapsed=time.monotonic()-episode_start;max_episode_seconds=max(max_episode_seconds,elapsed)
                summary=env.metrics.summary(p['training_seconds'])
                summary['rewards']={name:metric.summary() for name,metric in
                                    [('controller',controller_reward),('wce',demand_reward)] if metric.count}
                append_record(run/'episodes.jsonl.gz',dict(attempt=attempt,episode=episode,learning_steps=ctrl.learning_steps,stage_steps=stage_steps,
                    elapsed_seconds=elapsed,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**summary))
                if args.audit:
                    audit=run/'audit';audit.mkdir(exist_ok=True);env.export_episode(audit/('episode_%06d'%episode))
                if episode%p['validation_every']==0 or episode==p['max_episodes'][args.stage]:
                    status=validate() or 'interrupted'
                elif episode%p['checkpoint_every']==0:
                    recovery=save('full','recovery');store.update(name=recovery,latest=True);store.collect()
                print('%s/%s %s episode=%d score=%s'%(args.network,args.controller,args.stage,episode,selection.best['score']),flush=True)
            if not stop['force']:
                final=save('full','segment' if status=='interrupted' else 'final')
                store.update('latest' if status=='interrupted' else 'final',final,latest=True)
                store.collect()
        except Interrupted:
            status='interrupted'
        except BaseException as exc:
            if stop['requested']:status='interrupted'
            else:
                status='failed';atomic_json(run/'result.json',dict(status=status,error=repr(exc),completed_episodes=episode))
                raise
        finally:
            if ctrl:ctrl.close()
            if wce:wce.close()
            if env:env.terminate()
            for sig,handler in handlers.items():signal.signal(sig,handler)
        if selection.history:progress(status)
        roles=store.read()['roles']
        result=dict(status=status,completed_episodes=episode,early_stop=selection.early_stop,
                    early_best=roles.get('early_best') if selection.early_stop else None,
                    best_score=selection.best['score'] if selection.best else None,
                    final_validation=selection.history[-1] if status=='max_budget_reached' else None,
                    final_checkpoint=roles.get('final'),best=store.read()['roles'].get('best'),
                    best_episode=selection.best['episode'] if selection.best else None,
                    attempt=attempt,max_episode_seconds=max_episode_seconds,max_checkpoint_seconds=max_checkpoint_seconds,
                    total_seconds=total_seconds+time.monotonic()-started,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    output_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()),
                    output_files=sum(1 for p in run.rglob('*') if p.is_file()),pilot=args.pilot)
        atomic_json(run/'result.json',result);return result
