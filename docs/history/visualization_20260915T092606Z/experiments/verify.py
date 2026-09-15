"""Run the full correction gate: deterministic tests and eight SUMO pilot cases."""
import argparse
import concurrent.futures
import gc
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np
from experiments.core import ROOT, METHODS, QueueMetric, file_hash, write_json, source_hashes
from experiments.checkpoint import inspect_checkpoint


def checkpoint_state(path, role):
    inspect_checkpoint(path)
    with (Path(path) / role / 'state.pkl').open('rb') as f:
        return pickle.load(f)


def tensor_hash(path, role):
    from tf_compat import tf
    reader = tf.train.NewCheckpointReader(str(Path(path) / role / 'variables'))
    h = hashlib.sha256()
    for key in sorted(reader.get_variable_to_shape_map()):
        h.update(key.encode())
        h.update(reader.get_tensor(key).tobytes())
    return h.hexdigest()


def worker(path, network, family):
    from experiments.runner import parser, run
    base = Path(path)
    base.mkdir(parents=True, exist_ok=False)
    common = ['--network', network, '--controller', family, '--seed', '9001', '--pilot', '--checkpoint-every', '1']

    def execute(name, options):
        try:
            return run(parser().parse_args(common + ['--output', str(base / name)] + options))
        finally:
            gc.collect()

    parent = execute('parent', ['--stage', 'parent', '--steps', '160'])
    # Compare the optimized measurement transport to direct TraCI values, without learning.
    from envs.experiment_env import make_environment
    from experiments.demand import materialize, mixture
    transport = make_environment(network, family, base / 'subscription_check', 9001)
    try:
        transport.reset_episode(61001)
        transport.prepare_routes()
        transport.verify_subscriptions = True
        vehicles = materialize(mixture(transport.groups, np.ones(11) / 11), 0, 600,
                               np.random.RandomState(51001), transport.resolve_route, 'transport')
        transport.inject(vehicles)
        for _ in range(40):
            transport.step([0] * len(transport.node_names))
    finally:
        transport.terminate()
    offline = execute('offline_wce', ['--stage', 'wce', '--episodes', '2', '--parent', parent])
    assert tensor_hash(parent, 'controller') == tensor_hash(offline, 'controller')
    parent_state = checkpoint_state(parent, 'controller')
    frozen_state = checkpoint_state(offline, 'controller')
    for name in ('learning_steps', 'scheduler_steps', 'backward_calls', 'minibatch_updates'):
        assert parent_state[name] == frozen_state[name]
    assert checkpoint_state(offline, 'wce')['updates'] == 2
    final = {}
    for method in METHODS:
        options = ['--stage', 'continue', '--steps', '2640', '--method', method, '--parent', parent]
        if method in ('fixed_wce', 'online_wce'):
            options += ['--wce', offline]
        final[method] = execute(method, options)
        identity = json.loads((base / method / 'environment.json').read_text())
        node_lanes = {n: identity['node_lanes'][n] for n in identity['nodes']}
        metric = QueueMetric(node_lanes, identity['neighbors'])
        decisions = [json.loads(line) for line in (base / method / 'demand_decisions.jsonl').read_text().splitlines()]
        for ep in (1, 2):
            queues = np.load(str(base / method / ('episode_{:04}.npz'.format(ep))))['queue']
            controls = [json.loads(line) for line in (base / method / ('episode_{:04}.controls.jsonl'.format(ep))).read_text().splitlines()]
            assert len(controls) == 1320
            for index, control in enumerate(controls):
                np.testing.assert_allclose(control['learner_rewards'], metric.rewards(queues[index * 5:(index + 1) * 5], family))
            for block in range(11):
                np.testing.assert_allclose(decisions[(ep - 1) * 11 + block]['wce_reward'], metric.wce(queues[block * 600:(block + 1) * 600]))
    states = {method: checkpoint_state(checkpoint, 'controller') for method, checkpoint in final.items()}
    for state in states.values():
        assert state['learning_steps'] == 2800
    for name in ('learning_steps', 'scheduler_steps', 'backward_calls', 'minibatch_updates', 'batch_lengths'):
        assert all(states[m][name] == states['baseline'][name] for m in METHODS)
    assert tensor_hash(offline, 'wce') == tensor_hash(final['fixed_wce'], 'wce')
    assert tensor_hash(offline, 'wce') != tensor_hash(final['online_wce'], 'wce')
    assert checkpoint_state(final['online_wce'], 'wce')['updates'] == 4
    resumed = execute('resumed_online', ['--stage', 'continue', '--steps', '2640', '--method', 'online_wce',
                      '--parent', parent, '--wce', offline, '--resume', str(base / 'online_wce' / 'checkpoint_000001320')])
    for role in ('controller', 'wce'):
        assert tensor_hash(resumed, role) == tensor_hash(final['online_wce'], role)
    artifact = execute('demand', ['--stage', 'demand'])
    demand = json.loads(Path(artifact).read_text())
    evaluations = []
    for method in METHODS:
        output = execute('eval_' + method, ['--stage', 'evaluate', '--parent', final[method], '--artifact', artifact])
        result = json.loads(Path(output).read_text())
        assert result['effective_sumo_seed'] == 61001 and result['sample_count'] == 3600
        assert result['demand_hash'] == demand['hash']
        evaluations.append(result)
    execute('eval_repeat', ['--stage', 'evaluate', '--parent', final['baseline'], '--artifact', artifact])
    a = np.load(str(base / 'eval_baseline' / 'rollout.npz'))
    b = np.load(str(base / 'eval_repeat' / 'rollout.npz'))
    np.testing.assert_array_equal(a['queue'], b['queue'])
    for label, option, value in [('sumo_seed', '--sumo-seed', '61002'), ('policy_seed', '--policy-seed', '71002')]:
        output = execute('eval_' + label, ['--stage', 'evaluate', '--parent', final['baseline'],
                         '--artifact', artifact, option, value])
        result = json.loads(Path(output).read_text())
        assert result['demand_hash'] == demand['hash']
        assert result['effective_sumo_seed'] == (61002 if label == 'sumo_seed' else 61001)
    try:
        execute('eval_interrupted', ['--stage', 'evaluate', '--parent', final['baseline'],
                '--artifact', artifact, '--fail-after-steps', '40'])
        raise AssertionError('Fault injection did not interrupt evaluation')
    except RuntimeError as exc:
        assert str(exc) == 'Deliberate pilot evaluation interruption'
    failed = json.loads((base / 'eval_interrupted' / 'result.json').read_text())
    assert failed['status'] == 'failed'
    assert not (base / 'eval_interrupted' / 'rollout_summary.json').exists()
    case = {'status': 'passed', 'network': network, 'controller': family,
            'corrections': ['C{:02}'.format(i) for i in range(1, 11)],
            'expected_final_steps': 2800, 'observed_final_steps': {m: s['learning_steps'] for m, s in states.items()},
            'expected_online_wce_updates': 4, 'observed_online_wce_updates': 4,
            'demand_hash': demand['hash'], 'evaluation_rollouts': 8,
            'frozen_controller_equal': True, 'fixed_wce_equal': True, 'online_wce_changed': True,
            'resumed_next_episode_equal': True, 'paired_repeat_equal': True,
            'interrupted_rollout_excluded': True,
            'backward_calls': {m: s['backward_calls'] for m, s in states.items()},
            'minibatch_updates_per_agent': {m: s['minibatch_updates'] for m, s in states.items()}}
    write_json(base / 'checks.json', case)
    return case


def launch_case(root, network, family):
    name = network + '_' + family
    with (root / (name + '.log')).open('x') as log:
        result = subprocess.run([sys.executable, '-m', 'revision.verify', '--worker',
                                 '--output', str(root / name), '--network', network, '--controller', family],
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Pilot failed: {} (see {}.log)'.format(name, name))
    print('PASS pilot ' + name, flush=True)
    return json.loads((root / name / 'checks.json').read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--network', choices=['grid', 'monaco'])
    p.add_argument('--controller', choices=['ia2c', 'ma2c', 'iqll', 'ppo'])
    args = p.parse_args()
    if args.worker:
        worker(args.output, args.network, args.controller)
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    from experiments.demand import profiles
    inputs = list((ROOT / 'config').rglob('*.ini'))
    inputs += list((ROOT / 'data_traffic/revised').rglob('*.csv'))
    inputs += list((ROOT / 'real_net_subnet/demand_groups/revised').rglob('*.csv'))
    inputs += [ROOT / p for p in ['large_grid/data/exp.net.xml', 'large_grid/data/exp.add.xml',
                                  'real_net_subnet/data/in/most.net.xml', 'real_net_subnet/data/in/most.add.xml']]
    inputs += [Path(g['source']) for network in ('grid', 'monaco') for g in profiles(network)]
    input_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in inputs}
    os.environ['REVISION_VERIFICATION_WORKERS'] = str(args.workers)
    with (root / 'unit.log').open('x') as log:
        unit = subprocess.run([sys.executable, '-m', 'revision.tests'], stdout=log, stderr=subprocess.STDOUT)
    if unit.returncode:
        raise RuntimeError('Unit verification failed: ' + str(root / 'unit.log'))
    with (root / 'workflow.log').open('x') as log:
        workflow = subprocess.run([sys.executable, '-m', 'tests.test_workflow'], stdout=log, stderr=subprocess.STDOUT)
    if workflow.returncode:
        raise RuntimeError('Workflow verification failed: ' + str(root / 'workflow.log'))
    print('PASS deterministic verification suite', flush=True)
    cases = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(launch_case, root, network, family) for network in ('grid', 'monaco')
                   for family in ('ia2c', 'ma2c', 'iqll', 'ppo')]
        for future in concurrent.futures.as_completed(futures):
            cases.append(future.result())
    for network in ('grid', 'monaco'):
        assert len({c['demand_hash'] for c in cases if c['network'] == network}) == 1
    for path, expected in hashes.items():
        if file_hash(ROOT / path) != expected:
            raise ValueError('Source changed while verification was running')
    evidence = {'unit.log': file_hash(root / 'unit.log'), 'workflow.log': file_hash(root / 'workflow.log')}
    for case in cases:
        path = case['network'] + '_' + case['controller'] + '/checks.json'
        evidence[path] = file_hash(root / path)
    gate = {'status': 'passed', 'corrections': ['C{:02}'.format(i) for i in range(1, 11)],
            'source_hashes': hashes, 'input_hashes': input_hashes, 'evidence': evidence, 'cases': cases,
            'pilot_seed': 9001, 'concurrent_workers': args.workers,
            'publication_training_started': False}
    write_json(root / 'gate.json', gate)
    print('PASS all ten corrections: ' + str(root / 'gate.json'), flush=True)


if __name__ == '__main__':
    main()
