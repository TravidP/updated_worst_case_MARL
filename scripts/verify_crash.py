#!/usr/bin/env python3
"""Kill a pilot mid-episode, reject duplicate ownership, then verify exact resume."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.checkpoint import Checkpoints,inspect_checkpoint
from experiments.storage import atomic_json

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verification',required=True);a=p.parse_args()
folder=Path(a.verification).resolve();case=folder/'grid_dql';output=case/'crash_resume'
os.environ['CBWCE_DATA_ROOT']=str(folder/'shared_inputs')
base=[sys.executable,str(ROOT/'main.py'),'continue','--network','grid','--controller','dql','--seed','9001',
      '--method','baseline','--parent',str(Checkpoints(case/'parent').role('best')),
      '--pilot','--validation',str(folder/'grid_pilot.json'),'--config',str(case/'pilot.ini'),
      '--pilot-block-seconds','10','--validation-repeats','1','--validate-every','1','--max-episodes','1',
      '--output',str(output)]
with (case/'crash.log').open('w') as log:
    child=subprocess.Popen(base,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    deadline=time.monotonic()+180
    try:
        while not (output/'blocks.jsonl.gz').exists():
            if child.poll() is not None:raise RuntimeError('Pilot exited before forced interruption')
            if time.monotonic()>deadline:raise TimeoutError('No episode reached')
            time.sleep(.005)
        latest=Checkpoints(output).role('latest');before=inspect_checkpoint(latest)['hash']
        os.killpg(child.pid,signal.SIGKILL);child.wait()
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
assert inspect_checkpoint(Checkpoints(output).role('latest'))['hash']==before
with (output/'.run.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    duplicate=subprocess.run(base+['--resume'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    assert duplicate.returncode!=0 and 'Another process owns this run' in duplicate.stdout
with (case/'crash_recovery.log').open('w') as log:
    subprocess.run(base+['--resume'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
from tf_compat import tf
one=tf.train.NewCheckpointReader(str(Checkpoints(case/'baseline').role('final')/'controller/variables'))
two=tf.train.NewCheckpointReader(str(Checkpoints(output).role('final')/'controller/variables'))
for key in one.get_variable_to_shape_map():np.testing.assert_array_equal(one.get_tensor(key),two.get_tensor(key))
atomic_json(case/'crash_verification.json',dict(status='passed',signal='SIGKILL',previous_checkpoint_preserved=True,
            duplicate_submission_rejected=True,exact_resumed_training_tensors=True))
