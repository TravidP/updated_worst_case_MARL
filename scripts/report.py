#!/usr/bin/env python3
"""Aggregate complete final-evaluation shards using independent training seeds."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.storage import read_records,atomic_json
from experiments.core import file_hash

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',action='append',required=True);p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
cells={};rows=[];pairing={};contracts={}
for entry in a.input:
    path=Path(entry);m=json.loads((path/'manifest.json').read_text());result=json.loads((path/'result.json').read_text())
    if m['pilot'] or result['status']!='complete':raise ValueError('Only complete publication evaluations may enter this report')
    if file_hash(path/'rollouts.jsonl.gz')!=result['rollouts_sha256']:raise ValueError('Damaged evaluation shard')
    contract=(m['test_hash'],m['validation_hash'])
    if m['network'] in contracts and contracts[m['network']]!=contract:raise ValueError('Incompatible test/validation contracts')
    contracts[m['network']]=contract
    data=read_records(path/'rollouts.jsonl.gz');seen=set();group=defaultdict(list)
    for r in data:
        key=(r['scenario'],r['realization'])
        if key in seen:raise ValueError('Duplicate rollout')
        seen.add(key);group[r['scenario']].append(r['mean_queue'])
        pairkey=(m['network'],)+key;pair=(r['demand_hash'],r['sumo_seed'])
        if pairkey in pairing and pairing[pairkey]!=pair:raise ValueError('Unpaired external traffic or SUMO seed')
        pairing[pairkey]=pair
    if len(group)!=23 or any(len(v)!=10 for v in group.values()):raise ValueError('Incomplete prescribed 23×10 evaluation suite')
    for split in ['seen','test']:
        values=[float(np.mean(v)) for s,v in group.items() if (s.startswith('seen_'))==(split=='seen')]
        if len(values)!=(11 if split=='seen' else 12):raise ValueError('Wrong split composition')
        key=(m['network'],m['controller'],m['method'],m['selection'],split,m['seed'])
        if key in cells:raise ValueError('Duplicate comparison cell')
        cells[key]=float(np.mean(values))
        rows.append(dict(network=key[0],controller=key[1],method=key[2],selection=key[3],split=key[4],seed=key[5],mean_queue=cells[key]))

def write(name,records):
    with (out/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

write('seeds.csv',rows);summary=[];paired=[]
for key in sorted({k[:-1] for k in cells}):
    if any(key+(s,) not in cells for s in (101,202,303)):raise ValueError('All three independent training seeds are required')
    values=[cells[key+(s,)] for s in (101,202,303)]
    summary.append(dict(zip(['network','controller','method','selection','split'],key),mean_queue=float(np.mean(values)),training_seed_sd=float(np.std(values,ddof=1)),n_training_seeds=3))
    if key[2]!='online_wce':
        online=key[:2]+('online_wce',)+key[3:]
        if all(online+(s,) in cells for s in (101,202,303)):
            diffs=[cells[online+(s,)]-cells[key+(s,)] for s in (101,202,303)]
            paired.append(dict(network=key[0],controller=key[1],comparator=key[2],selection=key[3],split=key[4],online_minus_comparator=float(np.mean(diffs)),training_seed_sd=float(np.std(diffs,ddof=1))))
write('comparison.csv',summary)
if paired:write('paired_differences.csv',paired)
atomic_json(out/'manifest.json',dict(status='complete',inputs=a.input,training_seeds=[101,202,303],note='SD across independent training seeds; rollout repeats are not independent training runs.'))
