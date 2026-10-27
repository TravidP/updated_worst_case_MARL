#!/usr/bin/env python3
"""Repeat a correctness parent in audit mode and compare learning and queue data."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.checkpoint import Checkpoints
from experiments.storage import read_records,atomic_json

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verification',required=True);p.add_argument('--case',required=True)
a=p.parse_args();folder=Path(a.verification).resolve();case=folder/a.case
network,family=a.case.split('_',1);output=case/'audit_parent'
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0');os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ['CBWCE_DATA_ROOT']=str(folder/'shared_inputs')
argv=[sys.executable,str(ROOT/'main.py'),'parent','--network',network,'--controller',family,'--seed','9001',
      '--pilot','--validation',str(folder/(network+'_pilot.json')),'--config',str(case/'pilot.ini'),
      '--pilot-block-seconds','10','--validation-repeats','1','--validate-every','1','--max-episodes','1',
      '--audit','--output',str(output)]
with (case/'audit.log').open('w') as f:subprocess.run(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
from tf_compat import tf
old=tf.train.NewCheckpointReader(str(Checkpoints(case/'parent').role('final')/'controller/variables'))
new=tf.train.NewCheckpointReader(str(Checkpoints(output).role('final')/'controller/variables'))
for key in old.get_variable_to_shape_map():np.testing.assert_array_equal(old.get_tensor(key),new.get_tensor(key))
original=read_records(case/'parent/episodes.jsonl.gz')[0];audited=read_records(output/'episodes.jsonl.gz')[0]
for key in ('mean_queue','integrated_queue','sample_count','learning_steps','stage_steps','inserted','completed'):
    assert original[key]==audited[key],key
with np.load(output/'audit/episode_000001.npz') as raw:
    assert raw['queue'].sum(1).mean()==audited['mean_queue']
    assert raw['actions'].shape[0]==22 and raw['rewards'].shape==raw['actions'].shape
atomic_json(case/'audit_verification.json',dict(status='passed',case=a.case,exact_training_tensors=True,
            streamed_queue_matches_audit=True,audit_control_steps=22,
            compact_bytes=json.loads((case/'parent/result.json').read_text())['output_bytes'],
            audit_bytes=json.loads((output/'result.json').read_text())['output_bytes']))
