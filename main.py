#!/usr/bin/env python3
"""Standalone experiment CLI. Never imports the original workspace."""
import argparse
import json
import sys
from pathlib import Path

def main():
    args=sys.argv[1:]
    if args and args[0]=='experiment':args=args[1:]
    if args and args[0] in ('parent','wce','continue'):
        from experiments.runner import parser,run
        options=parser().parse_args(['--stage',args[0]]+args[1:])
        print(json.dumps(run(options),indent=2));return
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    check=sub.add_parser('check');check.add_argument('--network',choices=['grid','monaco'],default='grid')
    check.add_argument('--validation');check.add_argument('--publication',action='store_true')
    make=sub.add_parser('pilot-manifest');make.add_argument('--network',choices=['grid','monaco'],required=True);make.add_argument('--output',required=True)
    bench=sub.add_parser('test-manifest');bench.add_argument('--network',choices=['grid','monaco'],required=True);bench.add_argument('--output',required=True)
    ev=sub.add_parser('evaluate');ev.add_argument('--request',required=True);ev.add_argument('--output',required=True)
    package=sub.add_parser('package');package.add_argument('--kind',choices=['code','inputs','run'],required=True);package.add_argument('--source');package.add_argument('--output',required=True)
    ar=sub.add_parser('archive');ar.add_argument('--source',required=True);ar.add_argument('--destination',required=True)
    tune=sub.add_parser('tuning-configs');tune.add_argument('--output',required=True)
    opts=p.parse_args(args)
    from experiments.storage import atomic_json
    if opts.command=='check':
        from experiments.configuration import load_controller_config
        from experiments.protocol import root
        for family in ('ma2c','ppo','dql'):load_controller_config(opts.network,family)
        from experiments.demand import profiles
        groups=profiles(opts.network)
        if opts.validation:
            from experiments.validation import load_manifest
            load_manifest(opts.validation,opts.network,opts.publication)
        elif opts.publication:raise ValueError('Publication requires an explicit frozen validation manifest')
        if opts.publication:
            from experiments.runtime import verify_runtime
            verify_runtime()
        print(json.dumps(dict(status='passed',network=opts.network,profiles=len(groups),publication=opts.publication,
                             assets_root=str(root('assets'))),indent=2))
    elif opts.command in ('pilot-manifest','test-manifest'):
        if Path(opts.output).exists():raise FileExistsError(opts.output)
        if opts.command=='pilot-manifest':
            scenes=[dict(id='pilot_uniform',horizon=3600,generation_seed=9001,blocks=[dict(start=0,duration=3600,profile='Uniform')])]
            split='validation';frozen=False
        else:
            from experiments.scenarios import definitions
            scenes=definitions(opts.network,'seen')+definitions(opts.network,'test');split='test';frozen=True
        atomic_json(opts.output,dict(version='8.1',network=opts.network,split=split,frozen=frozen,scenarios=scenes))
    elif opts.command=='evaluate':
        from experiments.validation import evaluate
        if Path(opts.output).exists():raise FileExistsError(opts.output)
        atomic_json(opts.output,evaluate(json.loads(Path(opts.request).read_text())))
    elif opts.command=='package':
        from experiments.release import package
        print(package(opts.kind,opts.source,opts.output))
    elif opts.command=='archive':
        from experiments.release import archive
        print(archive(opts.source,opts.destination))
    elif opts.command=='tuning-configs':
        from experiments.tuning import generate
        print(generate(opts.output))

if __name__=='__main__':main()
