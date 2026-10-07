"""Audited legacy demand replay; not a complete fourteen-OD external validation."""
import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.core import digest, file_hash, source_hashes, write_json
from experiments.checkpoint import checkpoint_identity, inspect_checkpoint
from experiments.demand import ORDER, check_artifact, save_artifact
from experiments.protocol import settings
from scripts.external_campaign import read, NETS, SOURCES

ENTRY = ROOT / 'scripts/legacy_monaco_evaluate.py'


def prepare(out):
    from envs.experiment_env import make_environment
    from envs.real_net_env import RealNetEnv
    out.mkdir(parents=True, exist_ok=False)
    directory = out / 'artifacts/monaco'
    directory.mkdir(parents=True)
    source = ROOT / SOURCES['monaco']
    with source.open() as stream:
        rows = [dict(origin=r['origin_edge'], dest=r['dest_edge'], rate=float(r['veh_per_hour']))
                for r in csv.DictReader(stream)]
    env = make_environment('monaco', 'ia2c', out / 'route_probe', 10000)
    routes, failed = {}, []
    try:
        env.reset_episode(10000, evaluation=True, horizon=3600)
        for row in rows:
            stage = env.sim.simulation.findRoute(row['origin'], row['dest'], vType='type1')
            if stage.edges:
                routes[(row['origin'], row['dest'])] = list(stage.edges)
            else:
                failed.append(row)
        # Both ends and SUMO passenger permission are checked by findRoute.
        assert len(routes) == 13 and len(failed) == 1
        scenario = dict(id='monaco_legacy_skip_unreachable_v1', network='monaco', split='external',
                        horizon=3600, network_hash=file_hash(ROOT / NETS['monaco']),
                        source_files=[str(source)], source_hashes={str(source): file_hash(source)},
                        total_rate=sum(r['rate'] for r in rows),
                        routable_rate=sum(r['rate'] for r in rows)-sum(r['rate'] for r in failed),
                        mapping_version='legacy_600s_original_generator_v1',
                        provenance_status='legacy_partial_demand_replay',
                        route_validation=dict(status='passed', scope='Only the legacy-scheduled thirteen routable OD pairs',
                                              original_full_demand_status='blocked', skipped_od=failed),
                        label=dict(en='Monaco legacy demand replay (unreachable OD skipped)',
                                   zh='Monaco 旧需求流程复测（跳过不可达 OD）'))
        artifacts = []
        for seed in settings()['arrival_seeds']:
            vehicles, samples = [], []
            route_lookup = {}
            block_state = {'index': 0, 'row': 0}
            real_poisson = np.random.poisson
            def counted_poisson(mean):
                count = real_poisson(mean)
                row = rows[block_state['row'] % len(rows)]
                samples.append(dict(block=block_state['index'], origin=row['origin'], destination=row['dest'],
                                    sampled=int(count), skipped=(row['origin'], row['dest']) not in routes))
                block_state['row'] += 1
                return count
            def route_add(route_id, edges):
                route_lookup[route_id] = list(edges)
            def vehicle_add(vehID, routeID, typeID, depart):
                edges = route_lookup[routeID]
                vehicles.append(dict(id=vehID, depart=float(depart), edges=edges,
                                     origin=edges[0], destination=edges[-1], legacy_block=block_state['index']))
            def speed_factor(veh_id, factor):
                assert vehicles[-1]['id'] == veh_id
                vehicles[-1]['speed_factor'] = float(factor)
            adapter = SimpleNamespace(scenarios=[rows]*6, route_cache=set(),
                sim=SimpleNamespace(simulation=env.sim.simulation,
                                    route=SimpleNamespace(add=route_add),
                                    vehicle=SimpleNamespace(add=vehicle_add, setSpeedFactor=speed_factor)))
            state = np.random.get_state()
            try:
                np.random.seed(seed)
                np.random.poisson = counted_poisson
                for block in range(6):
                    block_state.update(index=block, row=0)
                    # Execute the original function, including Poisson-before-routing,
                    # silent route skip, jitter, IDs, and speed factor draw order.
                    RealNetEnv._inject_scenario_traffic(adapter, block, block*600)
            finally:
                np.random.poisson = real_poisson
                np.random.set_state(state)
            assert all(v['speed_factor'] > 0 for v in vehicles)
            assert len(vehicles) == sum(s['sampled'] for s in samples if not s['skipped'])
            audit = dict(seed=seed, sampled=sum(s['sampled'] for s in samples),
                         skipped=sum(s['sampled'] for s in samples if s['skipped']), scheduled=len(vehicles),
                         depart_at_or_after_horizon=sum(v['depart'] >= 3600 for v in vehicles), blocks=samples)
            write_json(directory / ('legacy_generation_%d.json' % seed), audit)
            path = directory / ('%s_%d.json' % (scenario['id'], seed))
            artifact = save_artifact(path, vehicles, dict(network='monaco', network_hash=scenario['network_hash'],
                                     horizon=3600, arrival_seed=seed, scenario=scenario, campaign_id=out.name,
                                     mapping_version=scenario['mapping_version'], generation_audit=audit))
            check_artifact(artifact)
            artifacts.append(dict(path=str(path), hash=artifact['hash'], file_hash=file_hash(path),
                                  arrival_seed=seed, scheduled=len(vehicles)))
    finally:
        env.terminate()
    models = []
    for controller in settings()['controllers']:
        for method in settings()['methods']:
            suite_path = ROOT / 'runs_eval/revised/publication_seed101_v1/monaco' / controller / method / 'suite.json'
            cp = Path(read(suite_path)['parent'])
            origin = checkpoint_identity(cp, 'monaco', controller, 101)
            bundle = inspect_checkpoint(cp)
            with (cp / 'controller/state.pkl').open('rb') as f:
                state = pickle.load(f)
            assert state['learning_steps'] == 2320000
            assert sorted(g['name'] for g in origin['training_profiles']) == sorted(ORDER)
            assert bundle['signatures']['controller']['assets']['network'] == scenario['network_hash']
            models.append(dict(controller=controller, method=method, checkpoint=str(cp),
                               checkpoint_hash=bundle['hash'], learning_steps=2320000,
                               training_profiles=11, historical_ancestry='Some old parent bundles unavailable; current bundle verified'))
    campaign = dict(campaign_id=out.name, training_seed=101, horizon=3600, policy_frozen=True,
                    source_hashes=source_hashes(), scheduler_hash=file_hash(__file__), evaluator_hash=file_hash(ENTRY),
                    protocol_notes=['Legacy eight-model checkpoint directories contain metadata only, not restorable weights.',
                                    'Uses current twenty final models, paired arrivals and independent policy RNG.',
                                    'Original generator is called directly; routes resolved on an empty network and cached for all blocks.',
                                    'Legacy originally resolved uncached routes in the live network and shared global arrival/policy RNG.',
                                    'Demand injected at 0,600,1200,1800,2400,3000; original departure jitter is not upper clipped.',
                                    'Current 116-lane per-second queue measurement and current frozen-policy engine are retained.',
                                    'This partial-demand reproduction does not certify the complete fourteen-OD external dataset.'],
                    networks=dict(monaco=dict(status='passed', scenario=scenario, models=models,
                                              artifacts=artifacts, expected_rollouts=200)))
    write_json(out / 'campaign.json', campaign)
    print(json.dumps(dict(status='prepared', artifacts=len(artifacts), models=len(models), skipped_od=failed)), flush=True)


def evaluate_one(out, model, index, smoke):
    entry = read(out / 'campaign.json')['networks']['monaco']
    artifact = entry['artifacts'][index]
    assert file_hash(artifact['path']) == artifact['file_hash']
    assert inspect_checkpoint(model['checkpoint'])['hash'] == model['checkpoint_hash']
    base = out / ('smoke' if smoke else '') / 'monaco' / model['controller'] / model['method'] / 'external' / entry['scenario']['id'] / ('rollout_%02d' % (index+1))
    if any(read(p)['status'] == 'complete' for p in base.glob('attempt_*/result.json')):
        return dict(status='already_complete', path=str(base))
    attempt = 1
    while (base / ('attempt_%03d' % attempt)).exists():
        attempt += 1
    target = base / ('attempt_%03d' % attempt)
    base.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ENTRY), 'evaluate', '--network', 'monaco', '--controller', model['controller'],
           '--method', model['method'], '--seed', '101', '--parent', model['checkpoint'],
           '--artifact', artifact['path'], '--sumo-seed', str(settings()['sumo_seeds'][index]),
           '--policy-seed', str(int(digest(['evaluation-policy', 101, index])[:8], 16)),
           '--no-visualization', '--output', str(target)]
    with (base / ('attempt_%03d.process.log' % attempt)).open('x') as log:
        code = subprocess.call(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError('Evaluation failed: ' + str(base))
    result = read(target / 'result.json')
    assert result['status'] == 'complete' and result['sample_count'] == 3600
    injections = read(target / 'legacy_injections.json')
    assert [r['time'] for r in injections] == list(range(0,3600,600))
    assert sum(r['scheduled'] for r in injections) == artifact['scheduled']
    return dict(status='complete', path=str(target), mean_queue=result['mean_queue'])


def execute(out, smoke, workers):
    campaign = read(out / 'campaign.json')
    assert source_hashes() == campaign['source_hashes']
    assert file_hash(__file__) == campaign['scheduler_hash'] and file_hash(ENTRY) == campaign['evaluator_hash']
    entry = campaign['networks']['monaco']
    models = entry['models']
    if smoke:
        models = [m for m in models if m['method'] == 'baseline']
    else:
        for model in [m for m in models if m['method'] == 'baseline']:
            base = out / 'smoke/monaco' / model['controller'] / model['method'] / 'external' / entry['scenario']['id'] / 'rollout_01'
            assert any(read(p)['status'] == 'complete' for p in base.glob('attempt_*/result.json'))
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLCONFIGDIR='/tmp/cbwce_mpl',
                      CBWCE_CONCURRENT_WORKERS=str(workers), CUDA_VISIBLE_DEVICES='-1')
    jobs = [(m,i) for m in models for i in range(1 if smoke else 10)]
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate_one,out,m,i,smoke) for m,i in jobs]
        for i,future in enumerate(futures):
            print(json.dumps(dict(index=i+1,total=len(jobs),elapsed=time.time()-started,result=future.result())),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','smoke','run'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4,choices=range(1,5))
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.output.resolve())
    else:
        execute(a.output.resolve(),a.action=='smoke',a.workers)
