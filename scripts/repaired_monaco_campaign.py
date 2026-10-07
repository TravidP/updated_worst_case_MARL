"""Twenty current final policies x ten paired complete-OD repaired-map replays."""
import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.core import digest,file_hash,source_hashes,write_json
from experiments.checkpoint import checkpoint_identity,inspect_checkpoint
from experiments.demand import ORDER,save_artifact,check_artifact
from experiments.protocol import settings
from scripts.repaired_monaco_protocol import REPAIR,BASE,BASE_ADD,NET,ADD,read,signature,interface,make_repaired,verify_gate,generate

ENTRY=ROOT/'scripts/repaired_monaco_evaluate.py'
PROTOCOL=ROOT/'scripts/repaired_monaco_protocol.py'


def prepare(out):
    from envs.experiment_env import make_environment
    out.mkdir(parents=True,exist_ok=False)
    repair=read(REPAIR/'repair_manifest.json')
    assert repair['status']=='passed' and file_hash(NET)==repair['repaired_network_hash']
    assert file_hash(ADD)==repair['repaired_additional_hash']
    gate=dict(status='passed',source_assets=dict(network=file_hash(BASE),additional=file_hash(BASE_ADD)),
              target_assets=dict(network=file_hash(NET),additional=file_hash(ADD)),
              repair_manifest_hash=file_hash(REPAIR/'repair_manifest.json'),families={},models=[])
    for family in settings()['controllers']:
        old=new=None
        try:
            old=make_environment('monaco',family,out/'compatibility_runtime'/family/'source',101)
            new=make_repaired(family,out/'compatibility_runtime'/family/'target',101)
            a,b=signature(old,family),signature(new,family)
            stripped_a=dict(a);stripped_b=dict(b);stripped_a.pop('assets');stripped_b.pop('assets')
            assert stripped_a==stripped_b and interface(old)==interface(new)
            gate['families'][family]=dict(source_signature=a,target_signature=b,
                                         source_interface=interface(old),target_interface=interface(new))
            for method in settings()['methods']:
                suite=ROOT/'runs_eval/revised/publication_seed101_v1/monaco'/family/method/'suite.json'
                cp=Path(read(suite)['parent']);origin=checkpoint_identity(cp,'monaco',family,101);bundle=inspect_checkpoint(cp)
                assert origin['stage']=='continue' and origin['method']==method
                assert sorted(g['name'] for g in origin['training_profiles'])==sorted(ORDER)
                assert bundle['signatures']['controller']==a
                with (cp/'controller/state.pkl').open('rb') as stream:state=pickle.load(stream)
                assert state['learning_steps']==2320000
                gate['models'].append(dict(controller=family,method=method,checkpoint=str(cp),checkpoint_hash=bundle['hash'],
                                          learning_steps=2320000,training_profiles=11))
        finally:
            if old is not None:old.terminate()
            if new is not None:new.terminate()
    gate['hash']=digest(gate);write_json(out/'transition_gate.json',gate);verify_gate(out/'transition_gate.json')
    directory=out/'artifacts/monaco';directory.mkdir(parents=True)
    source=REPAIR/'Real_Life_Monaco.csv'
    with source.open() as stream:rows=[dict(origin=r['origin_edge'],dest=r['dest_edge'],rate=float(r['veh_per_hour'])) for r in csv.DictReader(stream)]
    assert len(rows)==14
    scenario=dict(id='monaco_repaired_full14_v1',network='monaco',split='external',family='external',horizon=3600,
                  network_hash=file_hash(NET),network_file=str(NET),source_files=[str(source)],source_hashes={str(source):file_hash(source)},
                  total_rate=sum(r['rate'] for r in rows),mapping_version='repaired_topology_v2_original_600s_generator_v1',
                  provenance_status='repaired_topology_transfer',training_network_hash=file_hash(BASE),
                  repair_manifest=str(REPAIR/'repair_manifest.json'),repair_manifest_hash=file_hash(REPAIR/'repair_manifest.json'),
                  route_validation=dict(status='passed',positive_od=14,valid_od=14,skipped_od=[],original_flow_routes=88,physical_probe_completed=102),
                  label=dict(zh='Monaco 修复地图 · 完整14个OD',en='Monaco repaired map · all 14 OD pairs'))
    env=None;artifacts=[]
    try:
        env=make_repaired('ia2c',out/'route_probe',101)
        env.reset_episode(71001,evaluation=True,horizon=3600)
        for seed in settings()['arrival_seeds']:
            vehicles,audit,routes=generate(env,rows,seed)
            assert audit['skipped']==0 and len(routes)==14
            path=directory/('%s_%d.json'%(scenario['id'],seed))
            artifact=save_artifact(path,vehicles,dict(network='monaco',network_hash=scenario['network_hash'],horizon=3600,
                         arrival_seed=seed,scenario=scenario,campaign_id=out.name,generation_audit=audit,
                         mapping_version=scenario['mapping_version']))
            check_artifact(artifact)
            artifacts.append(dict(path=str(path),hash=artifact['hash'],file_hash=file_hash(path),arrival_seed=seed,scheduled=len(vehicles)))
    finally:
        if env is not None:env.terminate()
    campaign=dict(campaign_id=out.name,training_seed=101,horizon=3600,policy_frozen=True,
                  source_hashes=source_hashes(),scheduler_hash=file_hash(__file__),evaluator_hash=file_hash(ENTRY),
                  protocol_hash=file_hash(PROTOCOL),transition_gate=str(out/'transition_gate.json'),
                  transition_gate_file_hash=file_hash(out/'transition_gate.json'),
                  protocol_notes=['Strict checkpoint restore in the original training environment; frozen inference in the repaired environment.',
                                  'All signature fields except declared network/additional assets match; ordered nodes, lanes, dimensions and neighbors verified live.',
                                  'All fourteen original OD pairs retained, original traffic generator called for six 600-second blocks; no skipped vehicles.',
                                  'Ten paired arrival/SUMO/policy samples; final trained seed101 policies, no retraining and no WCE updates.',
                                  'Repaired topology v2 includes compiler geometry changes and 43 detector adjustments; results are map-transfer tests.',
                                  'Some historical parent bundles remain unavailable; current final bundle integrity verified.',
                                  'Departure jitter is not upper clipped; routes are resolved in an empty network and policy RNG is independent.'],
                  networks=dict(monaco=dict(status='passed',scenario=scenario,models=gate['models'],artifacts=artifacts,expected_rollouts=200)))
    write_json(out/'campaign.json',campaign)
    print(json.dumps(dict(status='prepared',models=20,artifacts=10,positive_od=14,skipped_od=0)),flush=True)


def evaluate_one(out,model,index,smoke):
    campaign=read(out/'campaign.json');entry=campaign['networks']['monaco'];artifact=entry['artifacts'][index]
    assert file_hash(artifact['path'])==artifact['file_hash']
    assert inspect_checkpoint(model['checkpoint'])['hash']==model['checkpoint_hash']
    base=out/('smoke' if smoke else '')/'monaco'/model['controller']/model['method']/'external'/entry['scenario']['id']/('rollout_%02d'%(index+1))
    if any(read(p)['status']=='complete' for p in base.glob('attempt_*/result.json')):return dict(status='already_complete',path=str(base))
    number=1
    while (base/('attempt_%03d'%number)).exists():number+=1
    target=base/('attempt_%03d'%number);base.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(ENTRY),'--controller',model['controller'],'--method',model['method'],
             '--parent',model['checkpoint'],'--artifact',artifact['path'],'--gate',campaign['transition_gate'],
             '--sumo-seed',str(settings()['sumo_seeds'][index]),'--policy-seed',str(int(digest(['evaluation-policy',101,index])[:8],16)),
             '--output',str(target)]
    with (base/('attempt_%03d.process.log'%number)).open('x') as log:code=subprocess.call(command,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
    if code:raise RuntimeError('Evaluation failed: '+str(target))
    result=read(target/'result.json');assert result['status']=='complete' and result['sample_count']==3600
    transfer=read(target/'transfer_audit.json');assert transfer['weights_before']==transfer['weights_after']
    assert [r['time'] for r in read(target/'block_injections.json')]==list(range(0,3600,600))
    return dict(status='complete',path=str(target),mean_queue=result['mean_queue'])


def execute(out,smoke,workers):
    campaign=read(out/'campaign.json');assert campaign['source_hashes']==source_hashes()
    assert file_hash(__file__)==campaign['scheduler_hash'] and file_hash(ENTRY)==campaign['evaluator_hash']
    assert file_hash(PROTOCOL)==campaign['protocol_hash']
    verify_gate(campaign['transition_gate'])
    models=campaign['networks']['monaco']['models'];scenario=campaign['networks']['monaco']['scenario']['id']
    if smoke:models=[m for m in models if m['method']=='baseline']
    else:
        for m in [m for m in models if m['method']=='baseline']:
            path=out/'smoke/monaco'/m['controller']/m['method']/'external'/scenario/'rollout_01'
            assert any(read(p)['status']=='complete' for p in path.glob('attempt_*/result.json'))
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='-1',
                      MPLCONFIGDIR='/tmp/cbwce_mpl',CBWCE_CONCURRENT_WORKERS=str(workers))
    jobs=[(m,i) for m in models for i in range(1 if smoke else 10)];started=time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(evaluate_one,out,m,i,smoke) for m,i in jobs]
        for i,future in enumerate(futures):
            result=future.result();print(json.dumps(dict(index=i+1,total=len(jobs),elapsed=time.time()-started,result=result)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','smoke','run'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.output.resolve())
    else:execute(a.output.resolve(),a.action=='smoke',a.workers)
