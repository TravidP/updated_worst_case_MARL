"""Read-only analysis of the site's seed-101 results and selected training logs.

Run from the repository root: python3 reports/wce_analysis_20261006/analyze.py
Only writes analysis products beside this script; never starts training/SUMO.
"""
import csv
import hashlib
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
METHODS = ['baseline', 'random_group', 'domain_randomization', 'fixed_wce', 'online_wce']
CONTROLLERS = ['ia2c', 'ma2c', 'iqll', 'ppo']
SITE = ROOT / 'docs/evaluation_workbook/grid_results_site/dist/data'


def readcsv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def savecsv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def logrows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    aggregate, paired, windows, manifests, scenarios = [], [], [], [], []
    validations = []
    for network in ['grid', 'monaco']:
        base = SITE if network == 'grid' else SITE / 'networks/monaco'
        summary = readcsv(base / 'metrics_summary.csv')
        rollouts = readcsv(base / 'rollout_metrics.csv')
        assert len(summary) == 460 and len(rollouts) == 4600
        grouped = {}
        for r in rollouts:
            key = (r['controller'], r['method'], r['scenario'])
            grouped.setdefault(key, []).append(r)
        for r in summary:
            key = (r['controller'], r['method'], r['scenario'])
            group = grouped[key]
            assert len(group) == 10
            assert math.isclose(st.mean(float(x['mean_queue']) for x in group),
                                float(r['mean_queue_mean']), rel_tol=1e-7)
        for scenario in set(r['scenario'] for r in rollouts):
            for index in range(1, 11):
                rs = [r for r in rollouts if r['scenario'] == scenario and
                      r['rollout'] == 'rollout_%02d' % index]
                assert len(rs) == 20
                assert len({(r['demand_hash'], r['arrival_seed'], r['sumo_seed']) for r in rs}) == 1
        validations.append(dict(network=network, summary_rows=460, rollout_rows=4600,
                                summary_reconciled=True, paired_exogenous_inputs=True))
        for controller in CONTROLLERS:
            for split in ['seen', 'test']:
                rows = [r for r in summary if r['controller'] == controller and r['split'] == split]
                scene_names = sorted({r['scenario'] for r in rows})
                winners = Counter()
                for name in scene_names:
                    rs = [r for r in rows if r['scenario'] == name]
                    winner = min(rs, key=lambda r: float(r['mean_queue_mean']))
                    winners[winner['method']] += 1
                    scenarios.extend(dict(network=network, controller=controller, split=split,
                                          family=r['family'], scenario=name, method=r['method'],
                                          mean_queue=float(r['mean_queue_mean']),
                                          sd=float(r['mean_queue_sd']), winner=winner['method']) for r in rs)
                bq = {r['scenario']: float(r['mean_queue_mean']) for r in rows if r['method'] == 'baseline'}
                for method in METHODS:
                    rs = [r for r in rows if r['method'] == method]
                    q = [float(r['mean_queue_mean']) for r in rs]
                    worst = max(rs, key=lambda r: float(r['mean_queue_mean']))
                    aggregate.append(dict(network=network, controller=controller, split=split, method=method,
                                          mean_queue=st.mean(q), worst_queue=max(q),
                                          worst3_queue=st.mean(sorted(q)[-3:]),
                                          worst_scenario=worst['scenario'], wins=winners[method],
                                          improves_baseline=sum(float(r['mean_queue_mean']) < bq[r['scenario']] for r in rs)))
                raw = {(r['method'], r['scenario'], r['rollout']): r for r in rollouts
                       if r['controller'] == controller and r['split'] == split}
                for comparator in METHODS[:-1]:
                    # One difference per index after equally averaging the scenarios.
                    d = np.array([st.mean(float(raw['online_wce', s, 'rollout_%02d' % i]['mean_queue']) -
                                         float(raw[comparator, s, 'rollout_%02d' % i]['mean_queue'])
                                         for s in scene_names) for i in range(1, 11)])
                    label = '|'.join(['wce_analysis_20261006', network, controller, split, comparator])
                    seed = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
                    rng = np.random.RandomState(seed)
                    boot = d[rng.randint(0, 10, (10000, 10))].mean(axis=1)
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    paired.append(dict(network=network, controller=controller, split=split, comparator=comparator,
                                       online_minus_comparator=float(d.mean()), ci_low=float(lo), ci_high=float(hi),
                                       bootstrap_seed=seed, bootstrap_samples=10000, n_paired_indices=10))
            for method in METHODS:
                suite = json.loads((ROOT / 'runs_eval/revised/publication_seed101_v1' /
                                    network / controller / method / 'suite.json').read_text())
                checkpoint = Path(suite['parent'])
                if not checkpoint.is_absolute():
                    checkpoint = ROOT / checkpoint
                run = checkpoint.parent
                manifest = json.loads((run / 'manifest.json').read_text())
                result = json.loads((run / 'result.json').read_text())
                assert result['status'] == 'complete' and result['learning_steps'] == 2320000
                manifests.append(dict(network=network, controller=controller, method=method,
                                      run=str(run.relative_to(ROOT)), checkpoint=checkpoint.name,
                                      python=manifest['runtime']['python'], tensorflow=manifest['runtime']['tensorflow'],
                                      learning_steps=result['learning_steps'],
                                      model_config=manifest['controller_effective_config']['MODEL_CONFIG'],
                                      source_hashes=manifest['source_hashes']))
                if method not in ['fixed_wce', 'online_wce']:
                    continue
                decisions = logrows(run / 'demand_decisions.jsonl')
                episodes = logrows(run / 'episode_metrics.jsonl')
                assert len(decisions) == 11000 and len(episodes) == 1000
                names = [p['name'] for p in manifest['training_profiles']]
                for period, ds in [('first100', decisions[:1100]), ('last100', decisions[-1100:])]:
                    entropy = [-sum(w * math.log(w) for w in r['weights'] if w > 0) for r in ds]
                    avg = [st.mean(r['weights'][k] for r in ds) for k in range(11)]
                    top = max(range(11), key=lambda k: avg[k])
                    windows.append(dict(network=network, controller=controller, method=method, period=period,
                                        window_count=len(ds), mean_weight_entropy=st.mean(entropy),
                                        mean_max_weight=st.mean(max(r['weights']) for r in ds),
                                        top_mean_profile=names[top], top_mean_weight=avg[top],
                                        mean_training_queue=st.mean(r['wce_raw_reward'] for r in ds),
                                        max_episode_reward_clip_fraction=max(r['reward_clip_fraction'] for r in episodes)))
    savecsv('aggregate_metrics.csv', aggregate)
    savecsv('paired_effects.csv', paired)
    savecsv('wce_weight_diagnostics.csv', windows)
    savecsv('scenario_rankings.csv', scenarios)
    (OUT / 'selected_training_evidence.json').write_text(json.dumps(manifests, indent=2) + '\n')
    (OUT / 'validation.json').write_text(json.dumps(validations, indent=2) + '\n')
    for row in paired:
        if row['split'] == 'test':
            print(row['network'], row['controller'], row['comparator'],
                  'delta %.3f [%.3f, %.3f]' % (row['online_minus_comparator'], row['ci_low'], row['ci_high']))


if __name__ == '__main__':
    main()
