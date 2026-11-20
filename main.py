#!/usr/bin/env python3
"""Standalone experiment CLI. Never imports the original workspace."""
import argparse
import json
import sys
from pathlib import Path

def main():
    from experiments.runtime import configure_threads
    configure_threads()
    args=sys.argv[1:]
    if args and args[0]=='local':
        from experiments.campaign import main as local_main
        local_main(args[1:]);return
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
    val=sub.add_parser('validation-manifest');val.add_argument('--network',choices=['grid','monaco'],required=True);val.add_argument('--output',required=True)
    val.add_argument('--design-seed',type=int,required=True);val.add_argument('--frozen',action='store_true')
    bench=sub.add_parser('test-manifest');bench.add_argument('--network',choices=['grid','monaco'],required=True);bench.add_argument('--output',required=True)
    ev=sub.add_parser('evaluate');ev.add_argument('--request',required=True);ev.add_argument('--output',required=True)
    package=sub.add_parser('package');package.add_argument('--kind',choices=['code','inputs','run'],required=True);package.add_argument('--source');package.add_argument('--output',required=True)
    ar=sub.add_parser('archive');ar.add_argument('--source',required=True);ar.add_argument('--destination',required=True)
    tune=sub.add_parser('tuning-configs');tune.add_argument('--output',required=True)
    lock=sub.add_parser('runtime-lock');lock.add_argument('--output',required=True)
    opts=p.parse_args(args)
    from experiments.storage import atomic_json
    if opts.command=='runtime-lock':
        from experiments.runtime import create_local_lock
        print(create_local_lock(opts.output))
    elif opts.command=='check':
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
    elif opts.command in ('pilot-manifest','validation-manifest','test-manifest'):
        if Path(opts.output).exists():raise FileExistsError(opts.output)
        if opts.command in ('pilot-manifest','validation-manifest'):
            from experiments.validation import validation_manifest
            manifest=validation_manifest(opts.network,9001 if opts.command=='pilot-manifest' else opts.design_seed,
                                         False if opts.command=='pilot-manifest' else opts.frozen)
        else:
            from experiments.scenarios import test_manifest
            manifest=test_manifest(opts.network)
        atomic_json(opts.output,manifest)
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
