"""Guarded minimal tuning campaign for revised controller configurations.

This module creates isolated candidate INIs, launches paired pilot runs and
promotes a candidate only after the recorded 66k validation satisfies every
protocol gate.  It never treats training reward as a selection metric.
"""
import argparse
import configparser
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.configuration import controller_config_path, load_controller_config
from experiments.core import ROOT, file_hash, write_json


def round_125(value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError('P95 reward magnitude must be positive and finite')
    exponent = math.floor(math.log10(value))
    base = 10. ** exponent
    choices = np.asarray([1., 2., 5., 10.]) * base
    return float(choices[np.argmin(np.abs(choices - value))])


def raw_rewards(controls):
    values = []
    with Path(controls).open() as stream:
        for line in stream:
            row = json.loads(line)
            values.extend(abs(float(x)) for x in row['raw_rewards'])
    if not values:
        raise ValueError('No raw rewards found in ' + str(controls))
    return np.asarray(values, dtype=np.float64)


def candidate_scales(values, existing):
    center = round_125(float(np.percentile(values, 95)))
    return center, sorted(set([center * .5, center, center * 2., float(existing)]))


def write_candidate(source, target, reward_norm, lr_factor=1.):
    parser = configparser.ConfigParser()
    parser.read(str(source))
    model = parser['MODEL_CONFIG']
    model['reward_norm'] = '{:.12g}'.format(reward_norm)
    model['reward_clip'] = '2.0'
    model['lr_init'] = '{:.12g}'.format(model.getfloat('lr_init') * lr_factor)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = configparser.ConfigParser()
        existing.read(str(target))
        expected = {section: dict(parser[section]) for section in parser.sections()}
        observed = {section: dict(existing[section]) for section in existing.sections()}
        if observed != expected:
            raise ValueError('Existing candidate differs from requested configuration: ' + str(target))
        return target
    with target.open('x') as stream:
        parser.write(stream)
    return target


def create_candidates(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = controller_config_path(args.network, args.controller)
    base = load_controller_config(args.network, args.controller)[0]
    values = raw_rewards(args.controls)
    center, scales = candidate_scales(values, base['MODEL_CONFIG'].getfloat('reward_norm'))
    rows = []
    for scale in scales:
        path = output / 'offline' / ('reward_norm_{:g}.ini'.format(scale))
        write_candidate(source, path, scale)
        rows.append({'network': args.network, 'controller': args.controller,
                     'reward_norm': scale, 'reward_clip': 2., 'lr_factor': 1.,
                     'path': str(path), 'sha256': file_hash(path),
                     'status': 'pending_offline'})
    write_json(output / 'campaign.json', {
        'network': args.network, 'controller': args.controller,
        'controls': str(Path(args.controls).resolve()),
        'raw_reward_p95': float(np.percentile(values, 95)),
        'rounded_center': center, 'candidates': rows,
    })
    write_table(output / 'candidates.csv', rows)
    return output / 'campaign.json'


def write_table(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def offline_results(path):
    results = []
    for summary in Path(path).glob('offline/*/summary.json'):
        row = json.loads(summary.read_text())
        row['summary'] = str(summary)
        results.append(row)
    eligible = [r for r in results if r.get('passed')]
    eligible.sort(key=lambda r: (r['median_relative_mse'],
                                 -r['median_clipping_factor']))
    return results, eligible[:2]


def launch_offline(args):
    campaign_path = Path(args.campaign).resolve()
    campaign = json.loads(campaign_path.read_text())
    trajectory = Path(campaign['controls']).parent
    for candidate in campaign['candidates']:
        config = Path(candidate['path'])
        output = campaign_path.parent / 'offline' / config.stem
        command = [sys.executable, '-m', 'experiments.offline_screen', 'fit',
                   '--network', campaign['network'], '--controller', campaign['controller'],
                   '--trajectory', str(trajectory), '--config', str(config),
                   '--output', str(output)]
        if (output / 'summary.json').exists():
            continue
        if output.exists():
            raise ValueError('Incomplete offline output must be preserved; start a new campaign: ' + str(output))
        subprocess.check_call(command, cwd=str(ROOT))


def materialize_pair_candidates(campaign_path):
    campaign_path = Path(campaign_path)
    campaign = json.loads(campaign_path.read_text())
    root = campaign_path.parent
    _, finalists = offline_results(root)
    if not finalists:
        raise ValueError('No offline candidate passed all gates')
    source = controller_config_path(campaign['network'], campaign['controller'])
    rows = []
    for result in finalists:
        for lr_factor in (1., .5):
            scale = result['reward_norm']
            name = 'reward_norm_{:g}_lr_{:g}.ini'.format(scale, lr_factor)
            target = root / 'paired' / 'configs' / name
            write_candidate(source, target, scale, lr_factor)
            rows.append({'reward_norm': scale, 'lr_factor': lr_factor,
                         'path': str(target), 'sha256': file_hash(target)})
    write_table(root / 'paired_candidates.csv', rows)
    return rows


def runner_command(network, controller, config, output, steps, monitor_every):
    return [sys.executable, '-m', 'revision.runner', '--stage', 'parent',
            '--network', network, '--controller', controller, '--output', str(output),
            '--pilot', '--seed', '9002', '--config', str(config), '--steps', str(steps),
            '--monitor-every', str(monitor_every), '--monitor-rollouts', '3',
            '--checkpoint-every', str(monitor_every)]


def launch_pairs(args):
    campaign_path = Path(args.campaign).resolve()
    campaign = json.loads(campaign_path.read_text())
    rows = materialize_pair_candidates(campaign_path)
    for row in rows:
        label = Path(row['path']).stem
        output = campaign_path.parent / 'paired' / 'runs' / label
        output.parent.mkdir(parents=True, exist_ok=True)
        command = runner_command(campaign['network'], campaign['controller'],
                                 row['path'], output, 2640, 1)
        if (output / 'result.json').exists():
            result = json.loads((output / 'result.json').read_text())
            if result.get('status') == 'complete':
                continue
        if output.exists():
            raise ValueError('Incomplete paired output must be preserved; start a new campaign: ' + str(output))
        command_path = output.parent / (label + '_command.json')
        if not command_path.exists():
            write_json(command_path, {'command': command})
        subprocess.check_call(command, cwd=str(ROOT))


def gradient_ratio(run):
    ratios = []
    path = Path(run) / 'learner_metrics.jsonl'
    if path.exists():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('role') == 'learner' and 'actor_grad_norm' in row:
                    ratios.append(float(row['critic_grad_norm']) /
                                  max(float(row['actor_grad_norm']), 1e-12))
    return float(np.median(ratios)) if ratios else 0.


def launch_conditionals(args):
    campaign_path = Path(args.campaign).resolve()
    campaign = json.loads(campaign_path.read_text())
    if campaign['controller'] not in ('ia2c', 'ma2c'):
        return
    root = campaign_path.parent / 'paired'
    for run in sorted((root / 'runs').glob('*')):
        if gradient_ratio(run) <= 10:
            continue
        source = root / 'configs' / (run.name + '.ini')
        for suffix, key, value in (
                ('value025', 'value_coef', '0.25'),
                ('advnorm', 'adv_norm', 'true')):
            parser = configparser.ConfigParser()
            parser.read(str(source))
            parser['MODEL_CONFIG'][key] = value
            name = run.name + '_' + suffix
            target = root / 'configs' / (name + '.ini')
            if target.exists():
                existing = configparser.ConfigParser()
                existing.read(str(target))
                if ({s: dict(existing[s]) for s in existing.sections()} !=
                        {s: dict(parser[s]) for s in parser.sections()}):
                    raise ValueError('Existing conditional candidate differs: ' + str(target))
            else:
                with target.open('x') as stream:
                    parser.write(stream)
            output = root / 'runs' / name
            command = runner_command(campaign['network'], campaign['controller'],
                                     target, output, 2640, 1)
            if (output / 'result.json').exists():
                result = json.loads((output / 'result.json').read_text())
                if result.get('status') == 'complete':
                    continue
            if output.exists():
                raise ValueError('Incomplete conditional output must be preserved; start a new campaign: ' + str(output))
            command_path = root / (name + '_command.json')
            if not command_path.exists():
                write_json(command_path, {'command': command})
            subprocess.check_call(command, cwd=str(ROOT))


def monitor_summaries(run):
    summaries = []
    for path in sorted((Path(run) / 'monitoring').glob('round_*/summary.json')):
        value = json.loads(path.read_text())
        if value.get('status') == 'complete':
            summaries.append(value)
    return summaries


def clipping_median(run):
    values = []
    path = Path(run) / 'learner_metrics.jsonl'
    if path.exists():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('role') == 'learner' and 'clipping_factor' in row:
                    values.append(float(row['clipping_factor']))
    return float(np.median(values)) if values else 0.


def entropy_collapsed(run):
    values = []
    path = Path(run) / 'learner_metrics.jsonl'
    if path.exists():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('role') == 'learner' and 'entropy' in row:
                    values.append(float(row['entropy']))
    if len(values) < 20:
        return False
    return float(np.median(values[-10:])) < .1 * max(float(np.median(values[:10])), 1e-12)


def paired_gate(run):
    summaries = monitor_summaries(run)
    if len(summaries) != 3:
        return {'passed': False, 'reason': 'requires monitors at 0, 1320 and 2640'}
    initial, final = summaries[0]['metrics'], summaries[-1]['metrics']
    collapsed = entropy_collapsed(run)
    passed = (final['mean_total_queue'] <= initial['mean_total_queue'] and
              final['completed'] >= initial['completed'] and
              final['teleports'] <= initial['teleports'] and not collapsed)
    return {'passed': bool(passed), 'initial': initial, 'final': final,
            'queue_change': final['mean_total_queue'] - initial['mean_total_queue'],
            'completed_change': final['completed'] - initial['completed'],
            'teleports_change': final['teleports'] - initial['teleports'],
            'median_clipping_factor': clipping_median(run),
            'entropy_collapsed': collapsed,
            'median_critic_actor_grad_ratio': gradient_ratio(run)}


def rank_pairs(args):
    root = Path(args.campaign).resolve().parent
    rows = []
    for run in sorted((root / 'paired' / 'runs').glob('*')):
        result = paired_gate(run)
        result['run'] = str(run)
        result['config'] = str(root / 'paired' / 'configs' / (run.name + '.ini'))
        rows.append(result)
    eligible = [r for r in rows if r['passed']]
    eligible.sort(key=lambda r: (r['queue_change'], -r['completed_change'],
                                 r['teleports_change'], -r['median_clipping_factor']))
    write_json(root / 'paired_ranking.json', {'rows': rows,
                                              'winner': eligible[0] if eligible else None})
    if not eligible:
        raise ValueError('No paired candidate passed; do not launch 66k validation')
    return eligible[0]


def launch_final(args):
    campaign_path = Path(args.campaign).resolve()
    campaign = json.loads(campaign_path.read_text())
    winner = rank_pairs(args)
    output = campaign_path.parent / 'final_66k' / 'run'
    command = runner_command(campaign['network'], campaign['controller'],
                             winner['config'], output, 66000, 10)
    if (output / 'result.json').exists():
        result = json.loads((output / 'result.json').read_text())
        if result.get('status') == 'complete':
            return
    if output.exists():
        raise ValueError('Incomplete 66k output must be preserved; start a new campaign: ' + str(output))
    command_path = campaign_path.parent / 'final_66k' / 'command.json'
    command_path.parent.mkdir(parents=True, exist_ok=True)
    if not command_path.exists():
        write_json(command_path, {'command': command})
    subprocess.check_call(command, cwd=str(ROOT))


def final_gate(run):
    summaries = monitor_summaries(run)
    if len(summaries) < 2:
        return {'passed': False, 'reason': 'missing fixed monitors'}
    queues = np.asarray([x['metrics']['mean_total_queue'] for x in summaries])
    initial, final = summaries[0]['metrics'], summaries[-1]['metrics']
    slope = float(np.polyfit(np.arange(len(queues)), queues, 1)[0])
    clip = clipping_median(run)
    passed = (final['mean_total_queue'] <= initial['mean_total_queue'] and
              final['completed'] >= initial['completed'] and slope <= 0 and clip >= .01)
    return {'passed': bool(passed), 'initial': initial, 'final': final,
            'monitor_queue_slope': slope, 'median_clipping_factor': clip}


def promote(args):
    campaign_path = Path(args.campaign).resolve()
    campaign = json.loads(campaign_path.read_text())
    root = campaign_path.parent
    winner = json.loads((root / 'paired_ranking.json').read_text()).get('winner')
    if not winner:
        raise ValueError('No paired winner exists')
    result = final_gate(root / 'final_66k' / 'run')
    gate_path = root / 'final_66k' / 'gate.json'
    if gate_path.exists():
        if json.loads(gate_path.read_text()) != result:
            raise ValueError('Existing final gate differs from recomputed result')
    else:
        write_json(gate_path, result)
    target = controller_config_path(campaign['network'], campaign['controller'])
    if result['passed']:
        shutil.copyfile(winner['config'], target)
        load_controller_config(campaign['network'], campaign['controller'])
        status = 'promoted'
    else:
        status = '当前算法/结构下无合格配置'
    english = '# Revised tuning result\n\nStatus: {}\n\nFinal gate: `{}`\n'.format(status, json.dumps(result, sort_keys=True))
    chinese = '# Revised 调参结果\n\n状态：{}\n\n最终门槛：`{}`\n'.format(status, json.dumps(result, ensure_ascii=False, sort_keys=True))
    (root / 'report_en.md').write_text(english)
    (root / 'report_zh.md').write_text(chinese)
    return status


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    # Keep this CLI usable in the project's Python 3.6 conda environment.
    # add_subparsers(required=True) is only accepted from Python 3.7 onward.
    sub = p.add_subparsers(dest='command')
    sub.required = True
    create = sub.add_parser('create')
    create.add_argument('--network', choices=['grid', 'monaco'], required=True)
    create.add_argument('--controller', choices=['ia2c', 'ma2c', 'iqll', 'ppo'], required=True)
    create.add_argument('--controls', required=True)
    create.add_argument('--output', required=True)
    for name in ('launch-offline', 'launch-pairs', 'launch-conditionals',
                 'rank-pairs', 'launch-final', 'promote'):
        command = sub.add_parser(name)
        command.add_argument('--campaign', required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    actions = {'create': create_candidates, 'launch-offline': launch_offline,
               'launch-pairs': launch_pairs,
               'launch-conditionals': launch_conditionals,
               'rank-pairs': rank_pairs, 'launch-final': launch_final,
               'promote': promote}
    result = actions[args.command](args)
    if result is not None:
        print(result)


if __name__ == '__main__':
    main()
