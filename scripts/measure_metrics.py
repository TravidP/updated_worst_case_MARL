#!/usr/bin/env python3
"""Measure queue-buffer storage only, with the original environment's list layout."""
import argparse
import gc
import sys
import time
import tracemalloc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.metrics import EpisodeMetrics
from experiments.storage import atomic_json

def measure(network,lanes,old):
    gc.collect();tracemalloc.start();started=time.monotonic()
    if old:
        lane_rows=[];rows=[];controls=[]
        for t in range(1,6601):
            q=[(t+i)%19 for i in range(lanes)];lane_rows.append(q)
            rows.append(dict(time=t,queue=float(sum(q)),active=200,speed_sum=1000.,inserted=1,completed=1,pending=0,teleports=0,collisions=0))
            if t%5==0:controls.append(dict(time=t,raw_rewards=[-float(q[i%lanes]) for i in range(25 if network=='grid' else 28)]))
    else:
        m=EpisodeMetrics([str(i) for i in range(lanes)])
        for t in range(1,6601):m.add(t,[(t+i)%19 for i in range(lanes)],active=200,speed_sum=1000,inserted=1,completed=1,pending=0)
        assert len(m.recent)==600 and m.trace is None
    current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return dict(retained_bytes=current,peak_traced_bytes=peak,seconds=time.monotonic()-started)

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
results=[]
for network,lanes in [('grid',150),('monaco',116)]:
    old=measure(network,lanes,True);new=measure(network,lanes,False)
    results.append(dict(network=network,seconds=6600,lanes=lanes,legacy_lists=old,streaming=new,
                        retained_reduction_fraction=1-new['retained_bytes']/old['retained_bytes']))
atomic_json(a.output,dict(results=results,scope='Synthetic queue/statistics buffering only; not total model/SUMO RSS or training throughput.'))
