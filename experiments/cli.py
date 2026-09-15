"""User-facing commands for the integrated experiment workflow."""
import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from experiments.core import ROOT
from experiments.protocol import settings, output_path, output_root, run_id, config_path


def check(gate=None, checkpoint=None):
    import importlib.util
    import shutil
    from experiments.demand import profiles
    result={'python':sys.version.split()[0], 'executable':sys.executable,
            'sumo':shutil.which('sumo'), 'modules':{}, 'inputs':{}, 'gate':'not_selected'}
    result['sumo_gui']=shutil.which('sumo-gui')
    result['display']=os.environ.get('DISPLAY')
    for name in ('tensorflow','numpy','traci','matplotlib'):
        result['modules'][name]=importlib.util.find_spec(name) is not None
    for network in settings()['networks']:
        try:
            groups=profiles(network)
            result['inputs'][network]={'profiles':len(groups),'ready':len(groups)==11,
                                      'prepared':(ROOT/groups[0].get('prepared','missing')).exists()}
        except Exception as exc: result['inputs'][network]={'ready':False,'error':str(exc)}
    if gate:
        from experiments.runner import require_gate
        try: require_gate(gate); result['gate']='passed'
        except Exception as exc: result['gate']=str(exc)
    if checkpoint:
        from experiments.checkpoint import inspect_checkpoint
        try: result['checkpoint']=inspect_checkpoint(checkpoint)['signatures']
        except Exception as exc: result['checkpoint_error']=str(exc)
    result['ready']=bool(result['sumo']) and all(result['modules'].values()) and all(x['ready'] for x in result['inputs'].values())
    return result


def stage_args(stage, argv):
    from experiments.runner import parser
    p=parser()
    next(a for a in p._actions if a.dest=='output').required=False
    p.add_argument('--suite',choices=['seen','test','validation','all'])
    p.add_argument('--scenario')
    p.add_argument('--rollouts',type=int,default=10)
    args=p.parse_args(['--stage',stage]+argv)
    if stage=='evaluate' and args.parent:
        from experiments.checkpoint import checkpoint_identity
        origin=checkpoint_identity(args.parent,args.network,args.controller,args.seed)
        if origin.get('stage')=='continue':args.method=origin['method']
    if not args.output: args.output=str(output_path(stage,args.network,args.controller,args.seed,args.method,args.pilot))
    return args


def evaluate_suite(args):
    from experiments.runner import run
    from experiments.scenarios import definitions, artifact_for
    from envs.experiment_env import make_environment
    from experiments.core import digest, write_json
    import copy
    if args.rollouts < 1 or args.rollouts > 10: raise ValueError('rollouts must be 1..10')
    if not args.pilot and args.rollouts !=10: raise ValueError('Publication evaluation requires ten rollouts')
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    splits=('seen','test') if args.suite=='all' else (args.suite,)
    scenarios=[s for split in splits for s in definitions(args.network,split) if not args.scenario or s['id']==args.scenario]
    if not scenarios: raise ValueError('No matching scenario')
    env=make_environment(args.network,args.controller,output/'demand_runtime',9001)
    try:
        env.reset_episode(61001,evaluation=True,horizon=3600); env.prepare_routes()
        artifacts={(s['split'],s['id'],i):artifact_for(args.network,s,seed,env) for s in scenarios
                   for i,seed in enumerate(settings()['arrival_seeds'][:args.rollouts])}
    finally: env.terminate()
    write_json(output/'suite.json',dict(network=args.network,controller=args.controller,seed=args.seed,method=args.method,
               pilot=args.pilot,parent=args.parent,scenarios=scenarios,rollouts=args.rollouts))
    results=[]
    for scenario in scenarios:
        for i in range(args.rollouts):
            child=copy.deepcopy(args)
            child.artifact=str(artifacts[(scenario['split'],scenario['id'],i)])
            child.output=str(output/scenario['split']/scenario['id']/('rollout_%02d'%(i+1))/'attempt_001')
            child.sumo_seed=settings()['sumo_seeds'][i]
            child.policy_seed=int(digest(['evaluation-policy',args.seed,i])[:8],16)
            results.append(run(child))
    write_json(output/'suite_result.json',{'status':'complete','rollouts':results})
    return {'status':'complete','rollouts':len(results),'output':str(output)}


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h','--help'):
        print('CB-WCE: python main.py experiment {check,prepare,verify,dashboard,parent,wce,continue,demand,evaluate,report} [options]')
        return
    command=argv.pop(0)
    def interrupt(*_): raise KeyboardInterrupt('Stopped by user')
    signal.signal(signal.SIGTERM,interrupt)
    if command in ('parent','wce','continue','demand','evaluate'):
        from experiments.runner import run
        args=stage_args(command,argv)
        result=evaluate_suite(args) if command=='evaluate' and args.suite else run(args)
    elif command=='prepare':
        p=argparse.ArgumentParser();p.add_argument('--materialize',action='store_true');p.add_argument('--network',choices=['grid','monaco'],default='grid')
        args=p.parse_args(argv)
        from experiments.prepare import prepare
        result=prepare()
        if args.materialize:
            from experiments.scenarios import materialize_suite
            result=materialize_suite(args.network,'ia2c',output_root('demand')/'preparation'/run_id(True))
    elif command=='check':
        p=argparse.ArgumentParser();p.add_argument('--gate');p.add_argument('--checkpoint');args=p.parse_args(argv)
        result=check(args.gate,args.checkpoint)
    elif command=='verify':
        if '--output' not in argv: argv += ['--output',str(output_root('verify')/'verification'/run_id(True))]
        subprocess.check_call([sys.executable,'-m','experiments.verify']+argv,cwd=str(ROOT));return
    elif command=='dashboard':
        from experiments.dashboard import serve
        p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);args=p.parse_args(argv)
        serve(args.port);return
    elif command=='report':
        from experiments.reporting import report
        p=argparse.ArgumentParser();p.add_argument('--input',action='append');p.add_argument('--output');args=p.parse_args(argv)
        result=report([Path(x) for x in args.input] if args.input else [output_root('evaluate')],
                      Path(args.output) if args.output else output_root('report')/run_id(True))
    else: raise ValueError('Unknown experiment command: '+command)
    print(json.dumps(result,indent=2,default=str))


if __name__=='__main__': main()
