#!/usr/bin/env python3
"""Run the guarded revised tuning campaign for all eight combinations.

Completed phases are detected from their immutable artifacts and skipped.  An
incomplete run directory is never deleted or overwritten; start a new matrix
root if a phase was interrupted inside a single candidate run.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PHASES = ('collect', 'create', 'offline', 'pairs', 'conditionals',
          'rank', 'final', 'promote')
COMBINATIONS = tuple((network, controller)
                     for network in ('grid', 'monaco')
                     for controller in ('ia2c', 'ma2c', 'iqll', 'ppo'))


def read_json(path):
    return json.loads(Path(path).read_text())


def complete_result(path):
    path = Path(path)
    return path.exists() and read_json(path).get('status') == 'complete'


def paths(root, network, controller):
    base = root / ('{}_{}'.format(network, controller))
    return {
        'base': base,
        'trajectory': base / 'trajectory',
        'controls': base / 'trajectory' / 'frozen_episode.controls.jsonl',
        'search': base / 'search',
        'campaign': base / 'search' / 'campaign.json',
    }


def phase_complete(phase, item, controller):
    trajectory, search = item['trajectory'], item['search']
    if phase == 'collect':
        return all((trajectory / name).exists() for name in
                   ('summary.json', 'batch.pkl', 'frozen_episode.controls.jsonl'))
    if phase == 'create':
        return item['campaign'].exists()
    if not item['campaign'].exists():
        return False
    campaign = read_json(item['campaign'])
    if phase == 'offline':
        return all((search / 'offline' / Path(row['path']).stem / 'summary.json').exists()
                   for row in campaign['candidates'])
    if phase == 'pairs':
        table = search / 'paired_candidates.csv'
        if not table.exists():
            return False
        import csv
        with table.open() as stream:
            names = [Path(row['path']).stem for row in csv.DictReader(stream)]
        return bool(names) and all(complete_result(
            search / 'paired' / 'runs' / name / 'result.json') for name in names)
    if phase == 'conditionals':
        if controller not in ('ia2c', 'ma2c'):
            return True
        table = search / 'paired_candidates.csv'
        if not table.exists():
            return False
        import csv
        from experiments.tuning import gradient_ratio
        with table.open() as stream:
            names = [Path(row['path']).stem for row in csv.DictReader(stream)]
        required = []
        for name in names:
            run = search / 'paired' / 'runs' / name
            if gradient_ratio(run) > 10:
                required.extend([name + '_value025', name + '_advnorm'])
        return all(complete_result(
            search / 'paired' / 'runs' / name / 'result.json') for name in required)
    if phase == 'rank':
        path = search / 'paired_ranking.json'
        return path.exists() and read_json(path).get('winner') is not None
    if phase == 'final':
        return complete_result(search / 'final_66k' / 'run' / 'result.json')
    if phase == 'promote':
        return all((search / name).exists() for name in
                   ('report_en.md', 'report_zh.md')) and (search / 'final_66k' / 'gate.json').exists()
    raise ValueError('Unknown phase: ' + phase)


def command_for(phase, item, network, controller):
    python = sys.executable
    if phase == 'collect':
        return [python, '-m', 'experiments.offline_screen', 'collect',
                '--network', network, '--controller', controller,
                '--output', str(item['trajectory'])]
    if phase == 'create':
        return [python, '-m', 'experiments.tuning', 'create',
                '--network', network, '--controller', controller,
                '--controls', str(item['controls']), '--output', str(item['search'])]
    names = {
        'offline': 'launch-offline', 'pairs': 'launch-pairs',
        'conditionals': 'launch-conditionals', 'rank': 'rank-pairs',
        'final': 'launch-final', 'promote': 'promote',
    }
    return [python, '-m', 'experiments.tuning', names[phase],
            '--campaign', str(item['campaign'])]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def select_combinations(value):
    if not value:
        return COMBINATIONS
    requested = []
    for token in value.split(','):
        parts = token.strip().split('/')
        if len(parts) != 2 or tuple(parts) not in COMBINATIONS:
            raise ValueError('Invalid --only combination: ' + token)
        requested.append(tuple(parts))
    return tuple(requested)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True,
                        help='New or existing matrix root; never use a formal run directory')
    parser.add_argument('--through', choices=PHASES, default='promote',
                        help='Stop after this phase (default: complete campaign and guarded promotion)')
    parser.add_argument('--only', help='Comma-separated subsets such as grid/ia2c,monaco/ppo')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-on-failure', action='store_true',
                        help='Default is to record a failure and continue with other combinations')
    args = parser.parse_args(argv)
    root = Path(args.output).resolve()
    combinations = select_combinations(args.only)
    selected_phases = PHASES[:PHASES.index(args.through) + 1]
    summary = {'root': str(root), 'through': args.through, 'dry_run': args.dry_run,
               'started_unix': time.time(), 'combinations': {}}
    failures = []

    blocked = set()
    for network, controller in combinations:
        summary['combinations'][network + '/' + controller] = {'phases': {}}

    for phase in selected_phases:
        print('\n========== PHASE {} =========='.format(phase), flush=True)
        for network, controller in combinations:
            label = network + '/' + controller
            item = paths(root, network, controller)
            state = summary['combinations'][label]
            if label in blocked:
                state['phases'][phase] = 'blocked_by_previous_failure'
                continue
            if phase_complete(phase, item, controller):
                state['phases'][phase] = 'already_complete'
                print('SKIP {} {}: complete'.format(label, phase), flush=True)
                continue
            command = command_for(phase, item, network, controller)
            print('RUN  {} {}: {}'.format(label, phase, ' '.join(command)), flush=True)
            if args.dry_run:
                state['phases'][phase] = 'dry_run'
                continue
            root.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.check_call(command, cwd=str(ROOT))
                if not phase_complete(phase, item, controller):
                    raise RuntimeError('Phase returned without complete artifacts')
                state['phases'][phase] = 'complete'
                atomic_json(root / 'matrix_summary.json', summary)
            except BaseException as exc:
                state['phases'][phase] = 'failed'
                state['error'] = repr(exc)
                failures.append({'combination': label, 'phase': phase,
                                 'error': repr(exc)})
                blocked.add(label)
                print('FAIL {} {}: {}'.format(label, phase, exc), file=sys.stderr, flush=True)
                if args.stop_on_failure:
                    summary['failures'] = failures
                    atomic_json(root / 'matrix_summary.json', summary)
                    raise

    summary['finished_unix'] = time.time()
    summary['failures'] = failures
    if not args.dry_run:
        atomic_json(root / 'matrix_summary.json', summary)
    print('\nMatrix finished: {} failure(s)'.format(len(failures)), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
