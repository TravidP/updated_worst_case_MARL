"""Training matrix with complete preflight validation and up to eight workers."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.checkpoint import checkpoint_identity, inspect_checkpoint
from experiments.protocol import settings

SCRIPTS = {
    'parent': '01_parent.sh',
    'wce': '02_wce.sh',
    'baseline': '03_baseline.sh',
    'random_group': '04_random_group.sh',
    'domain_randomization': '05_domain_randomization.sh',
    'fixed_wce': '06_fixed_wce.sh',
    'online_wce': '07_online_wce.sh',
}
METHODS = ['baseline', 'random_group', 'domain_randomization', 'fixed_wce', 'online_wce']


def _exit_status(status):
    return status if status >= 0 else 128 - status


def launch_jobs(commands):
    """Run one controller batch and stop the remaining jobs after any failure."""
    state = {'children': [], 'signal': None}
    output_lock = threading.Lock()
    readers = []

    def forward_output(child):
        for line in child.stdout:
            with output_lock:
                sys.stdout.write(line)
                sys.stdout.flush()

    def signal_children():
        for child in state['children']:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass

    def interrupt(signum, frame):
        state['signal'] = signum
        signal_children()

    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, interrupt)
        # Each Python/SUMO tree owns a process group, so cancellation reaches SUMO too.
        for cmd in commands:
            if state['signal'] is not None:
                break
            child = subprocess.Popen(
                cmd, cwd=str(ROOT), start_new_session=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
            state['children'].append(child)
            reader = threading.Thread(target=forward_output, args=(child,))
            reader.daemon = True
            reader.start()
            readers.append(reader)
        first_failure = 0
        cancelled = False
        remaining = set(state['children'])
        while remaining:
            completed = []
            for child in remaining:
                status = child.poll()
                if status is None:
                    continue
                completed.append(child)
                if status and not first_failure:
                    first_failure = _exit_status(status)
            remaining.difference_update(completed)
            if first_failure and remaining and not cancelled:
                signal_children()
                cancelled = True
            if remaining and not completed:
                time.sleep(0.1)
        for reader in readers:
            reader.join()
        return 128 + state['signal'] if state['signal'] is not None else first_failure
    except BaseException:
        signal_children()
        for child in state['children']:
            child.wait()
        for reader in readers:
            reader.join()
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def launch_job(cmd):
    """Backward-compatible single-job entry point used by focused tests."""
    return launch_jobs([cmd])


def path_from_root(value):
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def read_result(value, identity, stage, pilot):
    """Read JSON/integrity metadata only; never unpickle or instantiate models."""
    result_path = path_from_root(value)
    if result_path.name != 'result.json':
        raise ValueError('Select an exact completed result.json: ' + str(result_path))
    result = json.loads(result_path.read_text())
    if result.get('status') != 'complete':
        raise ValueError('Selected run is not complete: ' + str(result_path))
    checkpoint = path_from_root(result['checkpoint'])
    if checkpoint.parent != result_path.parent:
        raise ValueError('Checkpoint must remain alongside its originating result/manifest')
    origin = checkpoint_identity(checkpoint, *identity)
    if origin.get('stage') != stage or origin.get('pilot') is not pilot:
        raise ValueError('Selected checkpoint stage or pilot/publication mode mismatch')
    if result.get('manifest_hash') != origin['manifest_hash']:
        raise ValueError('Result and originating manifest do not match')
    bundle = inspect_checkpoint(checkpoint)
    required = ['controller', 'wce'] if stage == 'wce' else ['controller']
    if any(name not in bundle['signatures'] for name in required):
        raise ValueError('Missing controller/WCE state in selected checkpoint')
    if not pilot:
        p = settings()
        expected_simulation = (p['parent_steps'] if stage == 'parent' else
                               p['offline_episodes'] * p['training_seconds'] // p['control_seconds'])
        if (result.get('learning_steps') != p['parent_steps'] or
                result.get('stage_simulation_steps') != expected_simulation):
            raise ValueError('Selected run does not have the prescribed publication budget')
    return str(checkpoint), bundle, result


def selections(path, identities, need_wce, pilot):
    data = json.loads(path_from_root(path).read_text())
    if data.get('version') != 1 or not isinstance(data.get('runs'), list):
        raise ValueError('Checkpoint selection file requires version=1 and a runs list')
    rows = {}
    for row in data['runs']:
        key = (row['network'], row['controller'], row['seed'])
        if key in rows:
            raise ValueError('Duplicate checkpoint selection: ' + str(key))
        rows[key] = row
    resolved = {}
    for key in identities:
        if key not in rows:
            raise ValueError('Missing checkpoint selection: ' + str(key))
        row = rows[key]
        parent, parent_bundle, parent_result = read_result(row['parent_result'], key, 'parent', pilot)
        wce = None
        if need_wce:
            wce, wce_bundle, wce_result = read_result(row['wce_result'], key, 'wce', pilot)
            if wce_bundle['parents'].get('controller', {}).get('hash') != parent_bundle['hash']:
                raise ValueError('WCE was trained against a different parent: ' + str(key))
            if wce_result['learning_steps'] != parent_result['learning_steps']:
                raise ValueError('Frozen WCE run changed the controller learning budget')
        resolved[key] = (parent, wce)
    return resolved


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=list(SCRIPTS) + ['all_continuations'], required=True)
    p.add_argument('--mode', choices=['pilot', 'publication'], default=os.environ.get('CBWCE_MODE', 'pilot'))
    p.add_argument('--seed', type=int, help='Publication: 101 only; pilot default: 9001')
    p.add_argument('--checkpoints', default=os.environ.get('CBWCE_CHECKPOINTS'),
                   help='JSON mapping of explicit completed parent/WCE result.json files')
    p.add_argument('--write-template', help='Create a new selection JSON and exit; never overwrite')
    p.add_argument('--gate', default=os.environ.get('CBWCE_GATE'))
    p.add_argument('--monitor-every',type=int,default=50)
    p.add_argument('--monitor-rollouts',type=int,default=3)
    p.add_argument('--steps', type=int, help='Pilot continuation only')
    p.add_argument('--episodes', type=int, help='Pilot offline WCE only')
    p.add_argument('--workers', type=int, default=int(os.environ.get('CBWCE_WORKERS', '4')),
                   help='Concurrent controller jobs per method batch (default: 4; maximum: 8)')
    display = p.add_mutually_exclusive_group()
    display.add_argument('--visualization', dest='display', action='store_const', const='on')
    display.add_argument('--no-visualization', dest='display', action='store_const', const='off')
    p.add_argument('--dry-run', action='store_true')
    return p


def run(args):
    if args.mode not in ('pilot', 'publication'):
        raise ValueError('Invalid mode')
    pilot = args.mode == 'pilot'
    if not 1 <= args.workers <= 8:
        raise ValueError('Workers must be between 1 and 8')
    p = settings()
    pub_seeds = p['training_seeds']
    seeds = [args.seed if args.seed is not None else
                int(os.environ.get('CBWCE_SEED') or (9001 if pilot else 101))]
    for seed in seeds:
        if not 0 < seed < 4294967296 or ((seed in pub_seeds) == pilot):
            raise ValueError('Invalid seed for selected mode')
    identities = [(n, c, s) for n in p['networks'] for c in p['controllers'] for s in seeds]
    if args.write_template:
        if args.dry_run:
            raise ValueError('Choose --write-template OR --dry-run; a preview must not write files')
        path = path_from_root(args.write_template)
        rows = [dict(network=n, controller=c, seed=s,
                     parent_result='/replace/with/{}/{}/seed_{}/parent/run_id/result.json'.format(n, c, s),
                     wce_result='/replace/with/{}/{}/seed_{}/wce/run_id/result.json'.format(n, c, s))
                for n, c, s in identities]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump({'version': 1, 'runs': rows}, stream, indent=2)
            stream.write('\n')
        print('Selection template / 检查点选择模板: ' + str(path))
        return 0
    if os.environ.get('CBWCE_RESUME'):
        raise ValueError('Unset CBWCE_RESUME; use a single-stage script for recovery')
    methods = METHODS if args.stage == 'all_continuations' else [args.stage]
    if args.stage == 'parent':
        resolved = {}
    else:
        if not args.checkpoints:
            raise ValueError('Select --checkpoints FILE or create one with --write-template FILE')
        resolved = selections(args.checkpoints, identities,
                              any(m in ('fixed_wce', 'online_wce') for m in methods), pilot)
    flags = ['--mode', args.mode]
    for option in ('gate', 'steps', 'episodes', 'monitor_every', 'monitor_rollouts'):
        value = getattr(args, option)
        if value is not None:
            flags += ['--' + option.replace('_','-'), str(value)]
    if args.display:
        flags += ['--visualization' if args.display == 'on' else '--no-visualization']
    jobs = []
    for method in methods:
        for key in identities:
            n, c, seed = key
            cmd = ['bash', str(ROOT / 'scripts/training' / SCRIPTS[method]),
                   '--network', n, '--controller', c, '--seed', str(seed)] + flags
            if method != 'parent':
                parent, wce = resolved[key]
                cmd += ['--parent', parent]
                if method in ('fixed_wce', 'online_wce'):
                    cmd += ['--wce', wce]
            jobs.append((method, key, cmd))
    # Validate the entire selection and all launcher arguments before the first job.
    # The existing runner still validates runtime, model signatures and gate at launch.
    for _, _, cmd in jobs:
        check = subprocess.run(cmd + ['--dry-run'], cwd=str(ROOT), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
        if check.returncode:
            raise ValueError(check.stderr.strip() or check.stdout.strip())
    print('Preflight / 启动前检查: {} jobs valid; up to {} concurrent workers.'.format(
        len(jobs), args.workers), flush=True)
    batches = []
    for method in methods:
        selected = [job for job in jobs if job[0] == method]
        for start in range(0, len(selected), args.workers):
            batches.append(selected[start:start + args.workers])
    completed = 0
    for batch_index, batch in enumerate(batches, 1):
        labels = ', '.join('{}/seed_{}'.format(key[1], key[2]) for _, key, _ in batch)
        networks = '+'.join(sorted(set(key[0] for _, key, _ in batch)))
        print('\nPARALLEL BATCH / 并行批次 [{}/{}]: {} / {} / workers={} [{}]'.format(
            batch_index, len(batches), batch[0][0], networks, len(batch), labels), flush=True)
        commands = [cmd + (['--dry-run'] if args.dry_run else []) for _, _, cmd in batch]
        status = launch_jobs(commands)
        if status:
            print('Stopped / 已停止: failed batch was cancelled; no further batches will start '
                  '(exit {}).'.format(status), file=sys.stderr)
            return status
        completed += len(batch)
        print('Batch complete / 批次完成: {}/{} jobs.'.format(completed, len(jobs)), flush=True)
    print('All {} {} / 全部完成.'.format(len(jobs), 'previews completed' if args.dry_run else 'jobs completed'))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(run(parser().parse_args()))
    except KeyboardInterrupt:
        print('\nInterrupted / 已中断: no further jobs will start.', file=sys.stderr)
        sys.exit(130)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print('ERROR / 错误: {}'.format(exc), file=sys.stderr)
        sys.exit(2)
