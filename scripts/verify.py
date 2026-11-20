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
from experiments.runtime import configure_threads
configure_threads()
from experiments.storage import atomic_json
from experiments.protocol import config_path
from experiments.checkpoint import Checkpoints
from experiments.runner import fingerprint

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--case')
    p.add_argument('--extended',action='store_true',help='Also check complete horizons, audits, crash recovery and concurrent SUMO isolation')
    p.add_argument('--workers',type=int,choices=[1,2],default=1)
    p.add_argument('--shared-output',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.shared_output and not args.case:p.error('--shared-output requires --case')
    if args.extended and args.case:p.error('--extended requires all six cases')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=args.shared_output)
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0');os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
    os.environ['CBWCE_DATA_ROOT']=str(output/'shared_inputs')
    def command(arguments,log):
        with log.open('w') as f:subprocess.run([sys.executable,str(ROOT/'main.py')]+arguments,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,check=True)
    cases=[];started=time.monotonic();source=fingerprint()
    if not args.shared_output:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=str(ROOT),check=True,
                       stdout=(output/'unit.log').open('w'),stderr=subprocess.STDOUT)
    if args.workers==2 and not args.case:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        for network in ['grid','monaco']:
            command(['pilot-manifest','--network',network,'--output',str(output/(network+'_pilot.json'))],output/(network+'_prepare.log'))
        def case_worker(case):
            with (output/(case+'_worker.log')).open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/verify.py'),'--output',str(output),
                                '--case',case,'--shared-output'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            return case
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(case_worker,n+'_'+a) for n in ['grid','monaco'] for a in ['ma2c','dql','ppo']]
            for future in as_completed(futures):
                cases.append(future.result())
                atomic_json(output/'progress.json',dict(cases=cases,elapsed_seconds=time.monotonic()-started))
    else:
        for network in ['grid','monaco']:
            manifest=output/(network+'_pilot.json')
            if not manifest.exists():
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
                cases.append(case);atomic_json(output/((args.case+'_progress.json') if args.shared_output else 'progress.json'),dict(cases=cases,elapsed_seconds=time.monotonic()-started))
    import numpy as np
    from tf_compat import tf
    checks={}
    if args.extended:
        from experiments.validation import evaluate,load_manifest
        from experiments.evaluation import evaluate_run,read_evaluation
        from experiments.core import digest
        from experiments.scenarios import test_manifest
        # Frozen policies still sample actions: their RNG belongs in a full recovery state.
        for network in ['grid','monaco']:
            folder=output/(network+'_ppo')
            base=['--network',network,'--controller','ppo','--seed','9001','--pilot',
                  '--validation',str(output/(network+'_pilot.json')),'--config',str(folder/'pilot.ini'),
                  '--pilot-block-seconds','10','--validation-repeats','1','--validate-every','1','--max-episodes','2',
                  '--parent',str(Checkpoints(folder/'parent').role('best'))]
            for stage in ['wce','fixed_wce']:
                extra=[] if stage=='wce' else ['--method','fixed_wce','--wce',str(Checkpoints(folder/'wce').role('best'))]
                command_name='wce' if stage=='wce' else 'continue'
                direct=folder/(stage+'_direct_two');resumed=folder/(stage+'_resumed_two')
                command([command_name]+base+extra+['--output',str(direct)],folder/(stage+'_direct_two.log'))
                command([command_name]+base+extra+['--output',str(resumed),'--pilot-stop-after','1'],folder/(stage+'_stop_one.log'))
                assert json.loads((resumed/'result.json').read_text())['completed_episodes']==1
                command([command_name]+base+extra+['--output',str(resumed),'--resume'],folder/(stage+'_resume_two.log'))
                component='wce' if stage=='wce' else 'controller'
                first=tf.train.NewCheckpointReader(str(Checkpoints(direct).role('final')/component/'variables'))
                second=tf.train.NewCheckpointReader(str(Checkpoints(resumed).role('final')/component/'variables'))
                for key in first.get_variable_to_shape_map():np.testing.assert_allclose(first.get_tensor(key),second.get_tensor(key),rtol=1e-6,atol=1e-7)
                from experiments.storage import read_records
                for x,y in zip(read_records(direct/'episodes.jsonl.gz'),read_records(resumed/'episodes.jsonl.gz')):
                    assert x['mean_queue']==y['mean_queue']
        for network in ['grid','monaco']:
            folder=output/(network+'_dql');cp=Checkpoints(folder/'parent').role('best_inference')
            manifest,mhash=load_manifest(output/(network+'_pilot.json'),network)
            checked=evaluate(dict(stage='parent',network=network,controller='dql',seed=9001,checkpoint=str(cp),
                config=str(folder/'pilot.ini'),manifest=manifest,manifest_hash=mhash,repeats=1))
            assert checked['rollouts'][0]['sample_count']==6600 and len(checked['rollouts'][0]['blocks'])==11
            atomic_json(folder/'full_validation.json',checked)
            test=output/(network+'_test.json');atomic_json(test,test_manifest(network))
            destination=folder/'full_evaluation'
            evaluate_run(folder/'parent',test,destination,pilot=True,rollouts=1,scenarios=['seen_Uniform','peak_1.5'])
            before={f.name:f.read_bytes() for f in (destination/'rows').glob('*.json.gz')}
            evaluate_run(folder/'parent',test,destination,pilot=True,rollouts=1,scenarios=['seen_Uniform','peak_1.5'],resume=True)
            assert before=={f.name:f.read_bytes() for f in (destination/'rows').glob('*.json.gz')}
            _,rows=read_evaluation(destination)
            assert len(rows)==2 and all(r['sample_count']==3600 and r['trips']['count']==r['completed'] for r in rows)
        # Independent instances share traffic artifacts but never outputs or SUMO ports.
        children=[]
        for network in ['grid','monaco']:
            folder=output/(network+'_dql')
            log=(folder/'concurrent.log').open('w')
            argv=[sys.executable,str(ROOT/'main.py'),'parent','--network',network,'--controller','dql',
                  '--pilot','--validation',str(output/(network+'_pilot.json')),'--config',str(folder/'pilot.ini'),
                  '--pilot-block-seconds','10','--max-episodes','1','--validate-every','1',
                  '--output',str(folder/'concurrent_parent')]
            children.append((subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT),log,folder))
        for child,log,folder in children:
            try:assert child.wait()==0
            finally:log.close()
            assert json.loads((folder/'concurrent_parent/result.json').read_text())['completed_episodes']==1
        with (output/'crash_verification.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/verify_crash.py'),'--verification',str(output)],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        checks=dict(full_validation=['grid','monaco'],full_evaluation=['grid','monaco'],
                    concurrent_isolation=True,crash_resume=True,evaluation_resume=True,frozen_policy_resume=True)
    from experiments.runtime import identity,verify_runtime
    try:
        verify_runtime();runtime_verified=True;runtime_error=None
    except Exception as exc:
        runtime_verified=False;runtime_error=str(exc)
    if fingerprint()!=source:raise ValueError('Source changed during verification; rerun the matrix')
    gate=dict(status='passed' if len(cases)==6 and args.extended and runtime_verified else 'partial',
              correctness_status='passed' if len(cases)==6 else 'partial',fingerprint=source,cases=cases,checks=checks,
              runtime_verified=runtime_verified,runtime_error=runtime_error,
              runtime=identity(),pilot_block_seconds=10,validation_rollouts=1,
              limitations=['Correctness checks are not convergence evidence.','Publication requires the locked local/container runtime; development-system results do not bypass that gate.'],
              elapsed_seconds=time.monotonic()-started)
    atomic_json((output/args.case/'gate.json') if args.shared_output else (output/'gate.json'),gate);print(json.dumps(gate,indent=2))

if __name__=='__main__':main()
