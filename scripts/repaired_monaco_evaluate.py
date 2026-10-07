"""Strict-load on training assets, then auditable frozen inference on repaired assets."""
import argparse
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.core import RunRecord,digest,file_hash,validate_rollout,write_json
from experiments.checkpoint import inspect_checkpoint,load_checkpoint,checkpoint_identity
from experiments.demand import check_artifact,profiles
from scripts.repaired_monaco_protocol import make_repaired,signature,interface,verify_gate,read,NET,ADD


def weights(controller):
    hasher=hashlib.sha256()
    for array in controller.sess.run(controller.trainable):
        hasher.update(str(array.shape).encode());hasher.update(array.tobytes())
    return hasher.hexdigest()


def evaluate(a):
    from envs.experiment_env import make_environment
    from agents.controller import Controller
    gate=verify_gate(a.gate,check_models=False)
    checkpoint=Path(a.parent).resolve();bundle=inspect_checkpoint(checkpoint)
    origin=checkpoint_identity(checkpoint,'monaco',a.controller,101)
    assert origin['method']==a.method and origin['stage']=='continue'
    family=gate['families'][a.controller]
    selected=next(m for m in gate['models'] if (m['controller'],m['method'])==(a.controller,a.method))
    assert selected['checkpoint']==str(checkpoint) and selected['checkpoint_hash']==bundle['hash']
    assert bundle['signatures']['controller']==family['source_signature']
    artifact=read(a.artifact);check_artifact(artifact)
    assert artifact['network_hash']==file_hash(NET) and artifact['horizon']==3600
    parent=dict(path=str(checkpoint),hash=bundle['hash'])
    record=RunRecord(a.output,dict(stage='evaluate',network='monaco',controller=a.controller,method=a.method,
                   seed=101,pilot=False,artifact=str(Path(a.artifact).resolve()),artifact_file_hash=file_hash(a.artifact),
                   sumo_seed=a.sumo_seed,policy_seed=a.policy_seed,parents=dict(controller=parent),
                   training_profiles=profiles('monaco'),transition_gate=str(Path(a.gate).resolve()),
                   transition_gate_file_hash=file_hash(a.gate),protocol='repaired_topology_transfer',
                   runtime_script_hash=file_hash(__file__),protocol_script_hash=file_hash(ROOT/'scripts/repaired_monaco_protocol.py')))
    original=env=controller=None
    try:
        # No modified signature and no checkpoint hash override: the standard
        # strict loader sees the actual original training environment.
        original=make_environment('monaco',a.controller,record.path/'training_runtime',101)
        assert signature(original,a.controller)==family['source_signature']
        assert interface(original)==family['source_interface']
        controller=Controller(original,a.controller,101)
        load_checkpoint(checkpoint,dict(controller=controller))
        assert controller.learning_steps==2320000
        original.terminate()
        env=make_repaired(a.controller,record.path/'runtime',101)
        assert signature(env,a.controller)==family['target_signature']
        assert interface(env)==family['target_interface']
        write_json(record.path/'environment.json',dict(assets=env.asset_hashes,lanes=env.metric.lanes,
                   nodes=env.node_names,node_lanes={n:env.nodes[n].ilds_in for n in env.node_names},
                   neighbors={n:env.nodes[n].neighbor for n in env.node_names},
                   configuration={s:dict(env.config[s]) for s in env.config.sections()}))
        before=weights(controller)
        obs=env.reset_episode(a.sumo_seed,evaluation=True,horizon=3600)
        controller.reset();controller.rng.rng['policy']=np.random.RandomState(a.policy_seed)
        env.inject(artifact['vehicles'])
        for i in range(720):
            decision=controller.act(obs,env,False)
            nxt,rewards,done=env.step(decision[0])
            learner_rewards,clipped=controller.observe(obs,decision,rewards,nxt,done,False)
            env.controller_rows[-1].update(learner_rewards=learner_rewards.tolist(),reward_clipped=clipped.astype(int).tolist())
            obs=nxt
            if (i+1)%120==0:
                with (record.path/'progress.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(stage='evaluate',simulation_steps=i+1,learning_steps=controller.learning_steps,
                                                wce_updates=0,wall_seconds=time.monotonic()-record.started))+'\n')
        after=weights(controller)
        assert before==after and controller.learning_steps==2320000
        write_json(record.path/'transfer_audit.json',dict(status='passed',gate_hash=gate['hash'],
                   source_signature=controller.signature,target_signature=signature(env,a.controller),
                   strict_original_checkpoint_load=True,weights_before=before,weights_after=after,
                   learning_steps=2320000,wce_updates=0))
        summary=validate_rollout(env.rows);env.export_episode(record.path/'rollout');env.terminate()
        trips=[t for t in ET.parse(str(env.trip_file)).getroot().findall('tripinfo') if float(t.get('arrival','-1'))>=0]
        summary['completed_trip_denominator']=len(trips)
        for label,attribute in [('travel_time','duration'),('waiting_time','waitingTime'),('time_loss','timeLoss'),('departure_delay','departDelay')]:
            summary['mean_completed_'+label]=sum(float(t.get(attribute,'0')) for t in trips)/len(trips) if trips else None
        summary.update(scheduled=len(artifact['vehicles']),demand_hash=artifact['hash'],
                       effective_sumo_seed=env.effective_seed,checkpoint=parent)
        write_json(record.path/'rollout_summary.json',summary)
        record.finish('complete',**{k:v for k,v in summary.items() if k!='status'})
    except BaseException as exc:
        if not (record.path/'result.json').exists():record.finish('failed',error=str(exc))
        raise
    finally:
        if original is not None:original.terminate()
        if env is not None:env.terminate()
        if controller is not None:controller.sess.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller',required=True,choices=['ia2c','ma2c','iqll','ppo'])
    p.add_argument('--method',required=True)
    for key in ['parent','artifact','gate','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--sumo-seed',type=int,required=True);p.add_argument('--policy-seed',type=int,required=True)
    evaluate(p.parse_args())
