"""Audited frozen-policy transfer to the explicitly versioned repaired map."""
import csv
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
from experiments.core import ROOT, digest, file_hash, write_json
from experiments.checkpoint import inspect_checkpoint

REPAIR = ROOT/'real_net_subnet/repaired/group12_topology_v2'
BASE = ROOT/'real_net_subnet/data/in/most.net.xml'
BASE_ADD = ROOT/'real_net_subnet/data/in/most.add.xml'
NET = REPAIR/'data/in/most.net.xml'
ADD = REPAIR/'data/in/most.add.xml'


def read(path):
    return json.loads(Path(path).read_text())


def signature(env, family):
    return dict(kind='controller',network=env.network,family=family,
                n_s=list(map(int,env.n_s_ls)),n_a=list(map(int,env.n_a_ls)),
                n_w=list(map(int,env.n_w_ls)),n_f=list(map(int,env.n_f_ls)),
                config={s:dict(env.config[s]) for s in env.config.sections()},
                assets=dict(env.asset_hashes),lanes=env.metric.lanes,reward='learner_boundary_scaled_v3')


def interface(env):
    return dict(nodes=env.node_names,lanes=env.metric.lanes,
                node_lanes={n:env.nodes[n].ilds_in for n in env.node_names},
                neighbors={n:env.nodes[n].neighbor for n in env.node_names},
                n_s=list(map(int,env.n_s_ls)),n_a=list(map(int,env.n_a_ls)),
                n_w=list(map(int,env.n_w_ls)),n_f=list(map(int,env.n_f_ls)))


def make_repaired(family,output,seed=101):
    from envs.experiment_env import MonacoEnvironment, RevisedMixin
    class RepairedEnvironment(MonacoEnvironment):
        def _init_sim(self,seed,gui=False):
            self.net_file=NET
            self.asset_hashes=dict(network=file_hash(NET),additional=file_hash(ADD))
            tree=ET.parse(str(ADD))
            for element in tree.iter():
                if 'file' in element.attrib:element.set('file','/dev/null')
            self.additional=self.work/'repaired_detectors.add.xml'
            tree.write(str(self.additional))
            return RevisedMixin._init_sim(self,seed,gui)
        def inject(self,vehicles):
            self.repaired_blocks=[[v for v in vehicles if v['legacy_block']==i] for i in range(6)]
            self.repaired_injections=[]
        def _simulate(self,seconds):
            if self.cur_sec%600==0 and self.cur_sec<3600:
                block=self.cur_sec//600
                RevisedMixin.inject(self,self.repaired_blocks[block])
                self.repaired_injections.append(dict(block=block,time=self.cur_sec,
                                                     scheduled=len(self.repaired_blocks[block])))
                (self.work.parent/'block_injections.json').write_text(json.dumps(self.repaired_injections,indent=2)+'\n')
            return RevisedMixin._simulate(self,seconds)
    return RepairedEnvironment('monaco',family,output,seed,visualization=False)


def verify_gate(path,check_models=True):
    gate=read(path);body=dict(gate);claimed=body.pop('hash')
    assert digest(body)==claimed and gate['status']=='passed'
    assert gate['source_assets']==dict(network=file_hash(BASE),additional=file_hash(BASE_ADD))
    assert gate['target_assets']==dict(network=file_hash(NET),additional=file_hash(ADD))
    assert file_hash(REPAIR/'repair_manifest.json')==gate['repair_manifest_hash']
    for family,entry in gate['families'].items():
        a=dict(entry['source_signature']);b=dict(entry['target_signature'])
        assert a.pop('assets')==gate['source_assets'] and b.pop('assets')==gate['target_assets']
        assert a==b and entry['source_interface']==entry['target_interface']
        assert len(entry['source_interface']['nodes'])==28 and len(entry['source_interface']['lanes'])==116
    assert len(gate['models'])==20
    assert len({(m['controller'],m['method']) for m in gate['models']})==20
    if check_models:
        for model in gate['models']:
            bundle=inspect_checkpoint(model['checkpoint'])
            assert bundle['hash']==model['checkpoint_hash']
            assert bundle['signatures']['controller']==gate['families'][model['controller']]['source_signature']
    return gate


def generate(env,rows,seed):
    """Call the original traffic generator with recorded, fully reachable OD."""
    from envs.real_net_env import RealNetEnv
    routes={}
    for row in rows:
        stage=env.sim.simulation.findRoute(row['origin'],row['dest'],vType='type1')
        assert stage.edges
        routes[(row['origin'],row['dest'])]=list(stage.edges)
    vehicles=[];samples=[];lookup={};block_state={'index':0,'row':0}
    original_poisson=np.random.poisson
    def poisson(mean):
        count=original_poisson(mean);row=rows[block_state['row']%len(rows)]
        samples.append(dict(block=block_state['index'],origin=row['origin'],destination=row['dest'],sampled=int(count),skipped=False))
        block_state['row']+=1
        return count
    def add_route(route_id,edges):lookup[route_id]=list(edges)
    def add_vehicle(vehID,routeID,typeID,depart):
        edges=lookup[routeID]
        vehicles.append(dict(id=vehID,depart=float(depart),edges=edges,origin=edges[0],destination=edges[-1],legacy_block=block_state['index']))
    def speed(veh_id,factor):
        assert vehicles[-1]['id']==veh_id
        vehicles[-1]['speed_factor']=float(factor)
    adapter=SimpleNamespace(scenarios=[rows]*6,route_cache=set(),sim=SimpleNamespace(
        simulation=env.sim.simulation,route=SimpleNamespace(add=add_route),
        vehicle=SimpleNamespace(add=add_vehicle,setSpeedFactor=speed)))
    previous=np.random.get_state()
    try:
        np.random.seed(seed);np.random.poisson=poisson
        for block in range(6):
            block_state.update(index=block,row=0)
            RealNetEnv._inject_scenario_traffic(adapter,block,block*600)
    finally:
        np.random.poisson=original_poisson;np.random.set_state(previous)
    assert len(vehicles)==sum(s['sampled'] for s in samples)
    assert all(v['speed_factor']>0 for v in vehicles)
    return vehicles,dict(seed=seed,sampled=len(vehicles),scheduled=len(vehicles),skipped=0,
                         depart_at_or_after_horizon=sum(v['depart']>=3600 for v in vehicles),blocks=samples),routes
