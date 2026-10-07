#!/usr/bin/env python3
"""Paired supplementary statistics; exploratory bootstrap CIs and Holm tests."""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_external_results import validate
from experiments.core import write_json


def analyze(campaign, network, output):
    accepted = validate(campaign, network)
    output.mkdir(parents=True, exist_ok=False)
    groups = {}
    for p in accepted['summaries']:
        path = Path(p)
        manifest = json.loads(path.with_name('manifest.json').read_text())
        result = json.loads(path.read_text())
        groups.setdefault((manifest['controller'], manifest['method']), []).append((manifest['sumo_seed'], result))
    for key in groups:
        groups[key] = [r for seed, r in sorted(groups[key])]
    metrics = ['mean_queue', 'integrated_queue', 'peak_queue', 'mean_speed', 'scheduled',
               'inserted', 'completed', 'remaining', 'pending', 'teleports', 'collisions']
    summary = []
    for (controller, method), records in sorted(groups.items()):
        row = dict(controller=controller, method=method, n=10)
        for metric in metrics:
            values = [r.get(metric) for r in records]
            row[metric] = None if any(v is None for v in values) else dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)))
        summary.append(row)
    rng = np.random.RandomState(20261006)
    comparisons = []
    for controller in sorted({c for c, m in groups}):
        pairs = [(m, 'baseline') for m in ['random_group', 'domain_randomization', 'fixed_wce', 'online_wce']]
        pairs.append(('online_wce', 'fixed_wce'))
        for method, reference in pairs:
            a = np.array([r['mean_queue'] for r in groups[(controller, reference)]])
            b = np.array([r['mean_queue'] for r in groups[(controller, method)]])
            delta = a - b  # positive = method improves queue
            boot = delta[rng.randint(0, 10, (50000, 10))].mean(axis=1)
            signs = np.array(list(itertools.product([-1, 1], repeat=10)))
            null = np.abs((signs * delta).mean(axis=1))
            pvalue = float(np.mean(null >= abs(delta.mean()) - 1e-12))
            percentage = [float(100*d/v) if v != 0 else None for d, v in zip(delta, a)]
            comparisons.append(dict(controller=controller, method=method, reference=reference, n=10,
                                    mean_reference=float(a.mean()), mean_method=float(b.mean()),
                                    paired_queue_reduction_mean=float(delta.mean()),
                                    paired_queue_reduction_sd=float(delta.std(ddof=1)),
                                    exploratory_paired_bootstrap_ci95=list(map(float, np.percentile(boot, [2.5, 97.5]))),
                                    wins=int(sum(delta > 0)), ties=int(sum(delta == 0)),
                                    per_rollout_percent_reduction=percentage,
                                    mean_percent_reduction=None if any(v is None for v in percentage) else float(np.mean(percentage)),
                                    two_sided_exact_signflip_p=pvalue))
    previous = 0.
    for rank, i in enumerate(sorted(range(len(comparisons)), key=lambda i: comparisons[i]['two_sided_exact_signflip_p'])):
        adjusted = min(1., (len(comparisons)-rank)*comparisons[i]['two_sided_exact_signflip_p'])
        previous = max(previous, adjusted)
        comparisons[i]['holm_adjusted_p'] = previous
    result = dict(network=network, scenario=accepted['scenario'], training_seed=101,
                  inference_scope='Evaluation arrival/SUMO/policy sampling only; no across-training-seed claim.',
                  ci_method='50,000 paired bootstrap resamples, percentile 95%, exploratory and unadjusted.',
                  test_method='Two-sided exact paired sign-flip test (2^10 assignments); Holm across all 20 pre-specified comparisons. Assumes symmetry of paired differences under the null.',
                  summary=summary, comparisons=comparisons)
    write_json(output / 'paired_statistics.json', result)
    with (output / 'queue_comparisons.csv').open('x', newline='') as f:
        keys = ['controller','method','reference','mean_reference','mean_method','paired_queue_reduction_mean','wins','mean_percent_reduction','holm_adjusted_p']
        writer = csv.DictWriter(f, keys, extrasaction='ignore'); writer.writeheader(); writer.writerows(comparisons)
    print(json.dumps({'status':'complete','output':str(output),'comparisons':len(comparisons)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--network', choices=['grid','monaco'], required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); analyze(a.campaign, a.network, a.output)
