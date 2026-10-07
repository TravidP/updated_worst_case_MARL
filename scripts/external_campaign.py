#!/usr/bin/env python3
"""Isolated, frozen-policy supplementary demand evaluation (Python 3.6 compatible)."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.core import digest, file_hash, write_json, source_hashes
from experiments.demand import ORDER, materialize, save_artifact, check_artifact
from experiments.checkpoint import checkpoint_identity, inspect_checkpoint
from experiments.protocol import settings
import numpy as np

SOURCES = {'grid': 'data_traffic/demand_5x5_sparse.csv',
           'monaco': 'real_net_subnet/demand_groups/Real_Life_Monaco.csv'}
NETS = {'grid': 'large_grid/data/exp.net.xml',
        'monaco': 'real_net_subnet/data/in/most.net.xml'}
SCENARIOS = {'grid': 'grid_sparse_native_2983', 'monaco': 'monaco_local_most_constant'}


def read(path):
    return json.loads(Path(path).read_text())


def audit_models(network):
    result = []
    for controller in settings()['controllers']:
        for method in settings()['methods']:
            suite_path = ROOT / 'runs_eval/revised/publication_seed101_v1' / network / controller / method / 'suite.json'
            suite = read(suite_path)
            cp = Path(suite['parent'])
            origin = checkpoint_identity(cp, network, controller, 101)
            manifest = inspect_checkpoint(cp)
            if origin['stage'] != 'continue' or origin['method'] != method or suite.get('pilot'):
                raise ValueError('Not a final publication continuation: ' + str(cp))
            with (cp / 'controller/state.pkl').open('rb') as f:
                state = pickle.load(f)
            if state['learning_steps'] != 2320000:
                raise ValueError('Not the final learning budget')
            visited = set()
            unavailable_historical_bundles = []
            def audit_ancestry(path):
                path = Path(path)
                if str(path) in visited:
                    return
                visited.add(str(path))
                orig = checkpoint_identity(path, network, controller, 101)
                bundle = inspect_checkpoint(path) if (path / 'manifest.json').exists() else None
                if bundle is None:
                    unavailable_historical_bundles.append(str(path))
                groups = orig['training_profiles']
                if sorted(g['name'] for g in groups) != sorted(ORDER):
                    raise ValueError('Training ancestry includes external profiles')
                for group in groups:
                    if file_hash(group['source']) != group['source_hash']:
                        raise ValueError('Training source hash changed')
                for parent in orig.get('parents', {}).values():
                    parent_path = Path(parent['path'])
                    parent_bundle = inspect_checkpoint(parent_path) if (parent_path / 'manifest.json').exists() else None
                    if parent_bundle is not None and parent_bundle['hash'] != parent['hash']:
                        raise ValueError('Ancestry checkpoint hash mismatch')
                    audit_ancestry(parent['path'])
                return bundle
            audit_ancestry(cp)
            signature = manifest['signatures']['controller']
            if signature['assets']['network'] != file_hash(ROOT / NETS[network]):
                raise ValueError('Checkpoint network hash mismatch')
            result.append(dict(controller=controller, method=method, checkpoint=str(cp),
                               checkpoint_hash=manifest['hash'], suite=str(suite_path),
                               suite_hash=file_hash(suite_path), ancestry=sorted(visited),
                               learning_steps=state['learning_steps'], training_profiles=11,
                               unavailable_historical_bundles=unavailable_historical_bundles))
    return result


def routes_and_load(network, out, rows):
    # SUMO's actual route loader checks lane permissions and all connections;
    # every positive OD gets a vehicle even if its Poisson count would be zero.
    import sumolib
    import xml.etree.ElementTree as ET
    net = sumolib.net.readNet(str(ROOT / NETS[network]))
    routes, failures = {}, []
    xml = ET.Element('routes')
    ET.SubElement(xml, 'vType', id='type1', vClass='passenger', length='5', accel='5', decel='10', speedDev='0')
    for i, (o, d, rate) in enumerate(rows):
        if rate <= 0:
            continue
        try:
            edges, cost = net.getShortestPath(net.getEdge(o), net.getEdge(d), vClass='passenger')
            if not edges:
                raise ValueError('No passenger-compatible complete path')
            route = [e.getID() for e in edges]
            routes[(o, d)] = route
            vehicle = ET.SubElement(xml, 'vehicle', id='od_%d' % i, type='type1', depart='0')
            ET.SubElement(vehicle, 'route', edges=' '.join(route))
        except Exception as exc:
            failures.append(dict(origin=o, destination=d, rate=rate, error=str(exc)))
    path = out / 'all_positive_od.rou.xml'
    ET.ElementTree(xml).write(str(path))
    if failures:
        return routes, dict(status='blocked', failures=failures, positive_od=len(rows),
                            valid_od=len(routes), check='passenger graph; no partial demand campaign launched')
    command = ['sumo', '--net-file', str(ROOT / NETS[network]), '--route-files', str(path),
               '--end', '10', '--no-step-log', 'true', '--seed', '61001',
               '--error-log', str(out / 'route_errors.log')]
    with (out / 'route_load.log').open('x') as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    errors = (out / 'route_errors.log').read_text() if (out / 'route_errors.log').exists() else ''
    if code or 'Error:' in errors:
        raise ValueError('SUMO positive-OD loading failed: ' + errors)
    return routes, dict(status='passed', positive_od=len(rows), valid_od=len(routes),
                        command=command, route_file_hash=file_hash(path),
                        check='passenger-compatible paths and SUMO actual vehicle route loading')


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    campaign = dict(campaign_id=out.name, training_seed=101, horizon=3600,
                    policy_frozen=True, source_hashes=source_hashes(),
                    scheduler_hash=file_hash(__file__), networks={})
    for network in ('grid', 'monaco'):
        directory = out / 'artifacts' / network
        directory.mkdir(parents=True)
        source = ROOT / SOURCES[network]
        with source.open() as f:
            rows = [(r['origin_edge'], r['dest_edge'], float(r['veh_per_hour'])) for r in csv.DictReader(f)]
        if not all(np.isfinite(r) and r >= 0 for _, _, r in rows) or sum(r for _, _, r in rows) <= 0:
            raise ValueError('Invalid demand')
        rows = [(o, d, r) for o, d, r in rows if r > 0]
        model_audit_error = None
        try:
            models = audit_models(network)
        except Exception as exc:
            models = []
            model_audit_error = str(exc)
        routes, validation = routes_and_load(network, directory, rows)
        write_json(directory / 'route_validation.json', validation)
        scenario = dict(id=SCENARIOS[network], scenario_id=SCENARIOS[network], network=network,
                        split='external', family='external', source_files=[str(source)],
                        source_hashes={str(source): file_hash(source)}, mapping_version='local_csv_v1',
                        network_hash=file_hash(ROOT / NETS[network]), total_rate=sum(r for _, _, r in rows),
                        temporal_schedule=[dict(start=0, duration=3600, kind='constant_od_resampling')],
                        horizon=3600, route_validation=validation,
                        provenance_status='unverified_upstream_mapping' if network == 'grid' else 'local_reconstruction_only',
                        label={'en': 'Grid sparse external candidate' if network == 'grid' else 'Local MoST reconstructed OD',
                               'zh': 'Grid 稀疏外部候选需求' if network == 'grid' else '本地 MoST 重建 OD'})
        artifacts = []
        if validation['status'] == 'passed' and model_audit_error is None:
            for seed in settings()['arrival_seeds']:
                vehicles = materialize(rows, 0, 3600, np.random.RandomState(seed), lambda o, d: routes[(o, d)], 'external')
                target = directory / ('%s_%d.json' % (scenario['id'], seed))
                artifact = save_artifact(target, vehicles, dict(network=network, network_hash=scenario['network_hash'],
                                       horizon=3600, arrival_seed=seed, scenario=scenario,
                                       campaign_id=out.name, mapping_version='local_csv_v1'))
                check_artifact(artifact)
                artifacts.append(dict(path=str(target), hash=artifact['hash'], file_hash=file_hash(target),
                                      arrival_seed=seed, scheduled=len(vehicles)))
        campaign['networks'][network] = dict(scenario=scenario, models=models, artifacts=artifacts,
                                              expected_rollouts=200, model_audit_error=model_audit_error,
                                              status='blocked' if model_audit_error else validation['status'])
    write_json(out / 'campaign.json', campaign)
    print(json.dumps({n: dict(status=v['status'], artifacts=len(v['artifacts'])) for n, v in campaign['networks'].items()}, indent=2))


def evaluate_one(out, network, model, index, smoke=False):
    campaign = read(out / 'campaign.json')
    entry = campaign['networks'][network]
    if entry['status'] != 'passed':
        raise ValueError('Network has not passed route validation')
    artifact = entry['artifacts'][index]
    if file_hash(artifact['path']) != artifact['file_hash']:
        raise ValueError('Artifact changed')
    check_artifact(read(artifact['path']))
    if inspect_checkpoint(model['checkpoint'])['hash'] != model['checkpoint_hash']:
        raise ValueError('Checkpoint changed')
    base = out / ('smoke' if smoke else '') / network / model['controller'] / model['method'] / 'external' / entry['scenario']['id'] / ('rollout_%02d' % (index + 1))
    if base.exists():
        successful = [p for p in base.glob('attempt_*/result.json') if read(p).get('status') == 'complete']
        if successful:
            return dict(status='already_complete', path=str(successful[0]))
    attempt = 1
    while (base / ('attempt_%03d' % attempt)).exists():
        attempt += 1
    target = base / ('attempt_%03d' % attempt)
    base.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / 'main.py'), 'experiment', 'evaluate', '--network', network,
           '--controller', model['controller'], '--method', model['method'], '--seed', '101',
           '--parent', model['checkpoint'], '--artifact', artifact['path'],
           '--sumo-seed', str(settings()['sumo_seeds'][index]),
           '--policy-seed', str(int(digest(['evaluation-policy', 101, index])[:8], 16)),
           '--no-visualization', '--output', str(target)]
    logpath = base / ('attempt_%03d.process.log' % attempt)
    with logpath.open('x') as log:
        code = subprocess.call(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError('Evaluation failed, see ' + str(logpath))
    result = read(target / 'result.json')
    if result['status'] != 'complete' or result['sample_count'] != 3600:
        raise ValueError('Incomplete rollout')
    return dict(status='complete', path=str(target), mean_queue=result['mean_queue'])


def execute(out, network, workers, smoke):
    campaign = read(out / 'campaign.json')
    if source_hashes() != campaign['source_hashes'] or file_hash(__file__) != campaign['scheduler_hash']:
        raise ValueError('Source changed since campaign preparation')
    entry = campaign['networks'][network]
    if entry['status'] != 'passed':
        raise ValueError('Route validation blocked: ' + network)
    models = entry['models']
    if smoke:
        models = [m for m in models if m['method'] == 'baseline']
    else:
        for m in [m for m in models if m['method'] == 'baseline']:
            summary = out / 'smoke' / network / m['controller'] / m['method'] / 'external' / entry['scenario']['id'] / 'rollout_01'
            if not any(read(p).get('status') == 'complete' for p in summary.glob('attempt_*/result.json')):
                raise ValueError('Full controller smoke check required first: ' + m['controller'])
    jobs = [(m, i) for m in models for i in range(1 if smoke else 10)]
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLCONFIGDIR='/tmp/cbwce_mpl',
                      CBWCE_CONCURRENT_WORKERS=str(workers), CUDA_VISIBLE_DEVICES='-1')
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate_one, out, network, m, i, smoke) for m, i in jobs]
        for i, future in enumerate(futures):
            print(json.dumps(dict(index=i+1, total=len(jobs), elapsed=time.time()-started, result=future.result())), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'smoke', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--network', choices=['grid', 'monaco'], default='grid')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('workers must be 1..4')
    if args.action == 'prepare':
        prepare(args.output.resolve())
    else:
        execute(args.output.resolve(), args.network, args.workers, args.action == 'smoke')
