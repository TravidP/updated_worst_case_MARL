#!/usr/bin/env python3
"""Short real-SUMO six-case matrix. Never starts publication training."""
import argparse
import configparser
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.storage import atomic_json
from experiments.protocol import config_path
from experiments.checkpoint import Checkpoints
from experiments.runner import fingerprint

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--case');args=p.parse_args()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0');os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
    os.environ['CBWCE_DATA_ROOT']=str(output/'shared_inputs')
    def command(arguments,log):
        with log.open('w') as f:subprocess.run([sys.executable,str(ROOT/'main.py')]+arguments,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,check=True)
    cases=[];started=time.monotonic();source=fingerprint()
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=str(ROOT),check=True,
                   stdout=(output/'unit.log').open('w'),stderr=subprocess.STDOUT)
    for network in ['grid','monaco']:
        manifest=output/(network+'_pilot.json')
        command(['pilot-manifest','--network',network,'--output',str(manifest)],output/(network+'_prepare.log'))
        for family in ['ma2c','dql','ppo']:
            case=network+'_'+family
            if args.case and args.case!=case:continue
            folder=output/case;folder.mkdir()
            config=configparser.ConfigParser();config.read(config_path(network,family))
            if family=='dql':
                config['MODEL_CONFIG'].update(dict(buffer_size='128',learning_starts='8',batch_size='8',target_interval='20',update_interval='4'))
            cfg=folder/'pilot.ini'
            with cfg.open('w') as f:config.write(f)
            base=['--network',network,'--controller',family,'--seed','9001','--pilot','--validation',str(manifest),
                  '--config',str(cfg),'--pilot-block-seconds','10','--validation-repeats','1','--validate-every','1','--max-episodes','1']
            parent=folder/'parent';command(['parent']+base+['--output',str(parent)],folder/'parent.log')
            parent_cp=Checkpoints(parent).role('best')
            wce=folder/'wce';command(['wce']+base+['--output',str(wce),'--parent',str(parent_cp)],folder/'wce.log')
            wce_cp=Checkpoints(wce).role('best')
            for method in ['baseline','random_group','domain_randomization','fixed_wce','online_wce']:
                child=folder/method
                command_args=['continue']+base+['--method',method,'--parent',str(parent_cp),'--output',str(child)]
                if method.endswith('wce'):command_args+=['--wce',str(wce_cp)]
                command(command_args,folder/(method+'.log'))
                result=json.loads((child/'result.json').read_text())
                assert result['completed_episodes']==1 and result['status']!='failed'
            # A separate run stops at a durable boundary and then resumes the same protocol.
            resumed=folder/'resumed'
            command_args=['continue']+base+['--method','online_wce','--parent',str(parent_cp),'--wce',str(wce_cp),'--output',str(resumed)]
            command(command_args+['--segment-seconds','1'],folder/'segment.log')
            assert json.loads((resumed/'result.json').read_text())['status']=='interrupted'
            command(command_args+['--resume'],folder/'resume.log')
            # Fresh-process recovery must match uninterrupted controller/WCE tensors.
            import numpy as np
            from tf_compat import tf
            a=tf.train.NewCheckpointReader(str(Checkpoints(folder/'online_wce').role('final')/'controller/variables'))
            b=tf.train.NewCheckpointReader(str(Checkpoints(resumed).role('final')/'controller/variables'))
            for key in a.get_variable_to_shape_map():np.testing.assert_allclose(a.get_tensor(key),b.get_tensor(key),rtol=1e-6,atol=1e-7)
            a=tf.train.NewCheckpointReader(str(Checkpoints(folder/'online_wce').role('final')/'wce/variables'))
            b=tf.train.NewCheckpointReader(str(Checkpoints(resumed).role('final')/'wce/variables'))
            for key in a.get_variable_to_shape_map():np.testing.assert_allclose(a.get_tensor(key),b.get_tensor(key),rtol=1e-6,atol=1e-7)
            cases.append(case);atomic_json(output/'progress.json',dict(cases=cases,elapsed_seconds=time.monotonic()-started))
    from experiments.runtime import identity
    if fingerprint()!=source:raise ValueError('Source changed during verification; rerun the matrix')
    gate=dict(status='passed' if len(cases)==6 else 'partial',fingerprint=source,cases=cases,
              runtime=identity(),pilot_block_seconds=10,validation_rollouts=1,
              limitations=['Short correctness pilots, not performance or convergence evidence.','Container/Habrok acceptance requires running this script inside the locked image.'],
              elapsed_seconds=time.monotonic()-started)
    atomic_json(output/'gate.json',gate);print(json.dumps(gate,indent=2))

if __name__=='__main__':main()
