"""Acceptance checks for the complete isolated supplementary matrix."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.core import digest, file_hash
from experiments.checkpoint import inspect_checkpoint
from experiments.demand import check_artifact
from experiments.protocol import settings


def read(path):
    return json.loads(Path(path).read_text())


def validate(campaign_path, network):
    campaign = read(campaign_path)
    entry = campaign['networks'][network]
    assert entry['status'] == 'passed', 'Campaign preflight blocked'
    root = Path(campaign_path).parent / network
    scenario = entry['scenario']
    assert scenario['route_validation']['status'] == 'passed'
    from scripts.external_campaign import NETS
    repaired = scenario.get('provenance_status') == 'repaired_topology_transfer'
    if repaired:
        from scripts.repaired_monaco_protocol import verify_gate, NET, REPAIR
        assert network == 'monaco' and Path(scenario['network_file']) == NET
        gate = verify_gate(campaign['transition_gate'])
        for field, script in [('scheduler_hash','repaired_monaco_campaign.py'),
                              ('evaluator_hash','repaired_monaco_evaluate.py'),
                              ('protocol_hash','repaired_monaco_protocol.py')]:
            assert file_hash(ROOT/'scripts'/script) == campaign[field]
        assert file_hash(campaign['transition_gate']) == campaign['transition_gate_file_hash']
        assert file_hash(NET) == scenario['network_hash']
        assert file_hash(REPAIR/'repair_manifest.json') == scenario['repair_manifest_hash']
        assert scenario['route_validation']['positive_od'] == 14
    else:
        assert file_hash(ROOT / NETS[network]) == scenario['network_hash']
    for path, expected in scenario['source_hashes'].items():
        assert file_hash(path) == expected
    expected_keys = {(m['controller'], m['method'], i + 1) for m in entry['models'] for i in range(10)}
    artifacts = {}
    for index, artifact in enumerate(entry['artifacts']):
        assert file_hash(artifact['path']) == artifact['file_hash']
        payload = read(artifact['path']); check_artifact(payload)
        assert payload['hash'] == artifact['hash']
        assert payload['scenario'] == scenario
        assert payload['arrival_seed'] == settings()['arrival_seeds'][index]
        artifacts[index + 1] = payload
    accepted, failures = {}, []
    models = {(m['controller'], m['method']): m for m in entry['models']}
    for model in models.values():
        assert inspect_checkpoint(model['checkpoint'])['hash'] == model['checkpoint_hash']
    for path in sorted(root.glob('*/*/external/*/rollout_*/attempt_*/result.json')):
        rel = path.relative_to(root).parts
        controller, method, split, sid, rollout, attempt, _ = rel
        index = int(rollout.split('_')[1]); key = (controller, method, index)
        assert key in expected_keys and sid == scenario['id']
        result = read(path)
        if result['status'] != 'complete':
            failures.append(str(path)); continue
        assert key not in accepted, 'Duplicate successful attempt'
        manifest = read(path.with_name('manifest.json'))
        content = dict(manifest); claimed = content.pop('manifest_hash')
        assert digest(content) == claimed
        model = models[(controller, method)]; artifact = artifacts[index]
        assert manifest['parents']['controller'] == {'path': model['checkpoint'], 'hash': model['checkpoint_hash']}
        assert manifest['network'] == network and manifest['method'] == method and manifest['seed'] == 101
        assert manifest['stage'] == 'evaluate' and not manifest['pilot']
        assert manifest['source_hashes'] == campaign['source_hashes']
        assert manifest['artifact'] == entry['artifacts'][index-1]['path']
        assert manifest['artifact_file_hash'] == file_hash(manifest['artifact'])
        assert manifest['sumo_seed'] == settings()['sumo_seeds'][index-1]
        assert manifest['policy_seed'] == int(digest(['evaluation-policy', 101, index-1])[:8], 16)
        summary = read(path.with_name('rollout_summary.json'))
        assert summary['checkpoint'] == manifest['parents']['controller']
        assert summary['demand_hash'] == artifact['hash'] and summary['scheduled'] == len(artifact['vehicles'])
        assert summary['effective_sumo_seed'] == manifest['sumo_seed']
        assert summary['horizon'] == summary['sample_count'] == 3600
        env = read(path.with_name('environment.json'))
        assert env['assets']['network'] == scenario['network_hash']
        if repaired:
            family = gate['families'][controller]
            audit = read(path.with_name('transfer_audit.json'))
            assert audit['status'] == 'passed' and audit['gate_hash'] == gate['hash']
            assert audit['source_signature'] == family['source_signature']
            assert audit['target_signature'] == family['target_signature']
            assert audit['strict_original_checkpoint_load'] is True
            assert audit['weights_before'] == audit['weights_after']
            assert audit['learning_steps'] == 2320000 and audit['wce_updates'] == 0
            assert manifest['transition_gate_file_hash'] == campaign['transition_gate_file_hash']
            assert manifest['runtime_script_hash'] == campaign['evaluator_hash']
            assert manifest['protocol_script_hash'] == campaign['protocol_hash']
            assert env['assets'] == gate['target_assets']
            for name in ('nodes', 'lanes', 'node_lanes', 'neighbors'):
                assert env[name] == family['target_interface'][name]
            blocks = read(path.with_name('block_injections.json'))
            assert [r['time'] for r in blocks] == list(range(0,3600,600))
            assert [r['scheduled'] for r in blocks] == [sum(v['legacy_block']==i for v in artifact['vehicles']) for i in range(6)]
            generation = artifact['generation_audit']
            assert generation['skipped'] == 0 and generation['sampled'] == generation['scheduled'] == len(artifact['vehicles'])
            assert len(generation['blocks']) == 84
            for block in range(6):
                rows = [r for r in generation['blocks'] if r['block'] == block]
                assert len(rows) == len({(r['origin'],r['destination']) for r in rows}) == 14
                assert not any(r['skipped'] for r in rows)
        with np.load(path.with_name('rollout.npz'), allow_pickle=False) as data:
            q, t = data['queue'], data['time']
            assert q.shape == (3600, 150 if network == 'grid' else 116)
            assert np.isfinite(q).all() and (q >= 0).all()
            assert np.array_equal(t, np.arange(1, 3601))
            assert np.isclose(q.sum(axis=1).mean(), summary['mean_queue'])
        progress = [json.loads(line) for line in path.with_name('progress.jsonl').read_text().splitlines()]
        assert len(progress) == 6 and progress[-1]['simulation_steps'] == 720
        assert all(p['learning_steps'] == 2320000 and p['wce_updates'] == 0 for p in progress)
        accepted[key] = str(path.with_name('rollout_summary.json'))
    assert set(accepted) == expected_keys, 'Incomplete matrix: %d/200' % len(accepted)
    outcome = {'status': 'passed', 'rollouts': 200, 'groups': 20, 'failed_attempts': failures,
               'summaries': sorted(accepted.values()), 'scenario': scenario}
    if repaired:
        outcome.update(repaired_transfer_verified=200, full_od_verified=14)
    return outcome


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--network', choices=['grid', 'monaco'], required=True)
    args = p.parse_args()
    result = validate(args.campaign, args.network)
    print(json.dumps(result, indent=2))
