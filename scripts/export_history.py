#!/usr/bin/env python3
"""Export canonical complete episodes, dropping branches discarded by recovery."""
import argparse
import csv
import pickle
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.checkpoint import Checkpoints,inspect_checkpoint
from experiments.storage import read_records

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
a=p.parse_args();run=Path(a.run);output=Path(a.output)
if output.exists():raise FileExistsError(output)
records=read_records(run/'episodes.jsonl.gz');attempts=read_records(run/'attempts.jsonl');selected={}
for attempt in attempts:
    selected={e:r for e,r in selected.items() if e<=attempt['resume_episode']}
    for row in records:
        if row['attempt']==attempt['attempt']:selected[row['episode']]=row
latest=Checkpoints(run).role('latest');inspect_checkpoint(latest)
with (latest/'runner.pkl').open('rb') as f:durable=pickle.load(f)['episode']
rows=[selected[e] for e in sorted(selected) if e<=durable]
if not rows:raise ValueError('No durable complete training episodes yet')
output.parent.mkdir(parents=True,exist_ok=True)
with output.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('Exported %d durable episodes; raw attempt journals preserved.'%len(rows))
