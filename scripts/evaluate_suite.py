#!/usr/bin/env python3
"""Evaluate a frozen endpoint, saving and resuming each rollout independently."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.runtime import configure_threads
configure_threads()
from experiments.evaluation import evaluate_run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--selection',choices=['best','final'],default='best')
    p.add_argument('--pilot',action='store_true');p.add_argument('--rollouts',type=int,default=10)
    p.add_argument('--scenario',action='append');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    print(json.dumps(evaluate_run(a.run,a.manifest,a.output,a.selection,a.pilot,a.rollouts,a.scenario,a.resume),indent=2))

if __name__=='__main__':main()
