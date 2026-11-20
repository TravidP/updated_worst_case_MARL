#!/usr/bin/env python3
"""Inspect actual SUMO state/action interfaces and TensorFlow parameter shapes."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.runtime import configure_threads
configure_threads()
from experiments.storage import atomic_json
from experiments.runner import fingerprint
from envs.experiment_env import make_environment
from agents.controller import Controller
from agents.wce import WCE


def inspect():
    rows=[]
    def count(variables):
        import math
        return sum(math.prod(v.shape.as_list()) for v in variables)
    for network in ('grid','monaco'):
        for family in ('ma2c','ppo','dql'):
            with tempfile.TemporaryDirectory(prefix='cbwce-inspect-') as work:
                env=make_environment(network,family,work,9001);ctrl=wce=None
                try:
                    env.reset_episode(1)
                    ctrl=Controller(env,family,9001,training=True);wce=WCE(env,9001,training=False)
                    variables=ctrl.online if family=='dql' else ctrl.trainable
                    rows.append(dict(network=network,controller=family,nodes=env.node_names,
                        n_s=list(map(int,env.n_s_ls)),n_a=list(map(int,env.n_a_ls)),
                        n_fp=list(map(int,env.n_f_ls)),monitored_lanes=len(env.metric.lanes),
                        controller_trainable=count(variables),
                        controller_per_agent=[count([v for v in variables if v.name.startswith(
                            ('agent%d/'%i) if family=='dql' else (ctrl.policies[i].name+'/'))]) for i in range(len(env.node_names))],
                        target_parameters=count(ctrl.target) if family=='dql' else 0,
                        replay_bytes=ctrl.replay.nbytes if family=='dql' else 0,
                        wce_input=len(env.wce_observation()),wce_trainable=count(wce.trainable),
                        profile_order=[g['name'] for g in env.groups]))
                finally:
                    if ctrl:ctrl.close()
                    if wce:wce.close()
                    env.terminate()
    return dict(source=fingerprint(),cases=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    result=inspect();atomic_json(a.output,result);print(json.dumps(result,indent=2))
