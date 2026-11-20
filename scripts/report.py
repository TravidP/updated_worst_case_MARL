#!/usr/bin/env python3
"""Report a complete, explicitly specified one- or multi-training-seed campaign."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.runtime import configure_threads
configure_threads()
from experiments.reporting import report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',action='append',default=[]);p.add_argument('--campaign');p.add_argument('--output',required=True)
    p.add_argument('--training-seeds',nargs='+',type=int);p.add_argument('--network',action='append',choices=['grid','monaco'])
    p.add_argument('--controller',action='append',choices=['ma2c','dql','ppo'])
    p.add_argument('--selection',action='append',choices=['best','final'])
    a=p.parse_args();inputs=a.input
    if a.campaign and not inputs:
        from experiments.campaign import load_campaign
        c=load_campaign(a.campaign)
        inputs=[t['output'] for t in c['tasks'] if t['kind']=='evaluate'
                and (not a.network or t['network'] in a.network)
                and (not a.controller or t['controller'] in a.controller)
                and (not a.selection or t['selection'] in a.selection)]
    if not inputs:p.error('Supply --input or a campaign containing evaluation tasks')
    print(json.dumps(report(inputs,a.output,a.training_seeds,a.network,a.controller,a.selection,a.campaign),indent=2))

if __name__=='__main__':main()
