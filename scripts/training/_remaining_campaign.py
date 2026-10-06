"""Resume-aware four-worker publication campaign used by script 16."""
import argparse
import fcntl
import importlib.util
import json
import os
import pickle
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.checkpoint import checkpoint_identity, inspect_checkpoint
from experiments.core import digest
from experiments.protocol import output_root, settings
from experiments.runner import require_gate

MATRIX_SPEC = importlib.util.spec_from_file_location(
    'cbwce_training_matrix', str(ROOT / 'scripts/training/_matrix.py'))
MATRIX = importlib.util.module_from_spec(MATRIX_SPEC)
MATRIX_SPEC.loader.exec_module(MATRIX)

METHODS = ('random_group', 'domain_randomization', 'fixed_wce', 'online_wce')
GOAL = 1320000
KEEP_EVERY = 10
CLEAN_INTERVAL_SECONDS = 15 * 60
CLEAN_MIN_AGE_SECONDS = 60 * 60
LOW_DISK_BYTES = 30 * 1024 ** 3
STOP_DISK_BYTES = 20 * 1024 ** 3
INITIAL_DISK_BYTES = 40 * 1024 ** 3
MEMORY_PER_WORKER_BYTES = 3 * 1024 ** 3
EPISODE_RE = re.compile(r'^episode_(\d{4})\.(?:npz|jsonl|controls\.jsonl)$')
TRIP_RE = re.compile(r'^trips_(\d+)\.xml$')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--gate')
    p.add_argument('--force-new-gate', action='store_true')
    p.add_argument('--selection', default=str(
        ROOT / 'runs_eval/revised/selections/publication_seed101.json'))
    return p


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise ValueError('Cannot determine available memory from /proc/meminfo')


def resource_preflight(workers, enforce_disk):
    cpu_count = os.cpu_count() or 1
    memory = available_memory()
    disk = shutil.disk_usage(str(ROOT)).free
    print('RESOURCES cpu={} workers={} memory_available_gib={:.1f} disk_free_gib={:.1f}'.format(
        cpu_count, workers, memory / float(1024 ** 3), disk / float(1024 ** 3)))
    if cpu_count < workers:
        raise ValueError('Requested workers exceed available logical CPUs')
    if memory < workers * MEMORY_PER_WORKER_BYTES:
        raise ValueError('Need at least 3 GiB available memory per worker')
    if enforce_disk and disk < INITIAL_DISK_BYTES:
        raise ValueError('At least 40 GiB free disk is required before starting')


def validate_run_manifest(path, key, method, parent_hash, wce_hash):
    manifest = read_json(path)
    body = dict(manifest)
    claimed = body.pop('manifest_hash', None)
    if claimed is None or digest(body) != claimed:
        raise ValueError('run manifest integrity failure')
    network, controller, seed = key
    expected = (network, controller, seed, 'continue', method, False)
    observed = tuple(manifest.get(name) for name in
                     ('network', 'controller', 'seed', 'stage', 'method', 'pilot'))
    if observed != expected:
        raise ValueError('run identity mismatch: {}'.format(observed))
    parents = manifest.get('parents', {})
    if parents.get('controller', {}).get('hash') != parent_hash:
        raise ValueError('selected parent hash mismatch')
    if method in ('fixed_wce', 'online_wce'):
        if parents.get('wce', {}).get('hash') != wce_hash:
            raise ValueError('selected WCE hash mismatch')
    if (manifest.get('monitor_every'), manifest.get('monitor_rollouts')) != (50, 3):
        raise ValueError('publication monitoring settings mismatch')
    return manifest


def valid_resume(checkpoint, key, method, expected_signatures=None):
    checkpoint = Path(checkpoint)
    bundle = inspect_checkpoint(checkpoint)
    checkpoint_identity(checkpoint, *key)
    required = {'controller', 'wce'} if method in ('fixed_wce', 'online_wce') else {'controller'}
    if not required.issubset(set(bundle.get('signatures', {}))):
        raise ValueError('checkpoint is missing model state')
    if expected_signatures:
        for role, expected in expected_signatures.items():
            if bundle.get('signatures', {}).get(role) != expected:
                raise ValueError('{} model signature mismatch'.format(role))
    with (checkpoint / 'runner.pkl').open('rb') as stream:
        state = pickle.load(stream)
    step = int(checkpoint.name.split('_')[-1])
    if (state.get('stage'), state.get('method'), state.get('goal')) != ('continue', method, GOAL):
        raise ValueError('checkpoint stage/method/budget mismatch')
    if state.get('stage_steps') != step or not 0 <= step <= GOAL:
        raise ValueError('checkpoint step does not match runner state')
    if (state.get('monitor_every'), state.get('monitor_rollouts')) != (50, 3):
        raise ValueError('checkpoint monitoring settings mismatch')
    return step


def task_root(key, method):
    network, controller, seed = key
    return output_root('continue', network, method) / network / controller / (
        'seed_' + str(seed)) / method


def classify(key, method, parent, wce):
    parent_bundle = inspect_checkpoint(parent)
    wce_bundle = inspect_checkpoint(wce) if wce else None
    parent_hash = parent_bundle['hash']
    wce_hash = wce_bundle['hash'] if wce_bundle else None
    expected_signatures = {'controller': parent_bundle['signatures']['controller']}
    if wce_bundle:
        expected_signatures['wce'] = wce_bundle['signatures']['wce']
    root = task_root(key, method)
    resumes = []
    invalid = []
    for run in sorted(root.glob('publication_*')) if root.exists() else []:
        manifest_path = run / 'manifest.json'
        if not manifest_path.is_file():
            invalid.append('{}: missing manifest'.format(run))
            continue
        try:
            validate_run_manifest(manifest_path, key, method, parent_hash, wce_hash)
        except Exception as exc:
            invalid.append('{}: {}'.format(run, exc))
            continue
        result_path = run / 'result.json'
        if result_path.is_file():
            try:
                result = read_json(result_path)
                checkpoint = Path(result.get('checkpoint', ''))
                if (result.get('status') == 'complete' and
                        result.get('stage_simulation_steps') == GOAL and
                        checkpoint.parent == run and
                        valid_resume(checkpoint, key, method, expected_signatures) == GOAL):
                    return {'status': 'complete', 'checkpoint': str(checkpoint),
                            'step': GOAL, 'warnings': invalid}
            except Exception as exc:
                invalid.append('{}: invalid result ({})'.format(run, exc))
        for checkpoint in run.glob('checkpoint_[0-9]*'):
            try:
                step = valid_resume(checkpoint, key, method, expected_signatures)
                if step < GOAL:
                    resumes.append((step, checkpoint.stat().st_mtime, str(checkpoint)))
            except Exception as exc:
                invalid.append('{}: {}'.format(checkpoint, exc))
    if resumes:
        step, _, checkpoint = max(resumes)
        return {'status': 'resume', 'checkpoint': checkpoint, 'step': step,
                'warnings': invalid}
    if invalid:
        return {'status': 'invalid', 'step': 0, 'errors': invalid}
    return {'status': 'fresh', 'step': 0, 'warnings': []}


def resolve_jobs(selection):
    protocol = settings()
    identities = [(n, c, 101) for n in protocol['networks'] for c in protocol['controllers']]
    resolved = MATRIX.selections(Path(selection), identities, True, False)
    jobs = []
    for method in METHODS:
        for key in identities:
            parent, wce = resolved[key]
            selected_wce = wce if method in ('fixed_wce', 'online_wce') else None
            state = classify(key, method, parent, selected_wce)
            jobs.append(dict(method=method, key=key, parent=parent, wce=wce, state=state))
    return jobs


def label(job):
    return '{}/{}/{}'.format(job['method'], job['key'][0], job['key'][1])


def command_for(job, gate, dry_run=False):
    method = job['method']
    network, controller, seed = job['key']
    cmd = ['bash', str(ROOT / 'scripts/training' / MATRIX.SCRIPTS[method]),
           '--mode', 'publication', '--network', network, '--controller', controller,
           '--seed', str(seed), '--parent', job['parent'], '--gate', str(gate),
           '--monitor-every', '50', '--monitor-rollouts', '3', '--no-visualization']
    if method in ('fixed_wce', 'online_wce'):
        cmd += ['--wce', job['wce']]
    if job['state']['status'] == 'resume':
        cmd += ['--resume', job['state']['checkpoint']]
    if dry_run:
        cmd += ['--dry-run']
    return cmd


def matching_gate(explicit=None, workers=4):
    candidates = [Path(explicit)] if explicit else sorted(
        (ROOT / 'runs_eval/revised/verification').glob('*/gate.json'),
        key=lambda p: p.stat().st_mtime, reverse=True)
    errors = []
    for path in candidates:
        try:
            require_gate(str(path))
            if read_json(path).get('concurrent_workers') != workers:
                raise ValueError('gate worker count does not match requested workers')
            return path.resolve(), errors
        except Exception as exc:
            errors.append('{}: {}'.format(path, exc))
    return None, errors


def generate_gate(workers):
    stamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    output = ROOT / 'runs_eval/revised/verification' / ('{}_workers_port_lock_'.format(workers) + stamp)
    cmd = [sys.executable, '-u', 'main.py', 'experiment', 'verify',
           '--workers', str(workers), '--output', str(output)]
    print('Generating verification gate / 正在生成验证门槛: {}'.format(output), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))
    gate = output / 'gate.json'
    require_gate(str(gate))
    return gate.resolve()


class RuntimeControl(object):
    def __init__(self):
        self.stop = threading.Event()
        self.children = []
        self.lock = threading.Lock()

    def add(self, child):
        with self.lock:
            self.children.append(child)

    def remove(self, child):
        with self.lock:
            if child in self.children:
                self.children.remove(child)

    def interrupt(self):
        self.stop.set()
        with self.lock:
            children = list(self.children)
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass


def cleanup_once(log, emergency=False):
    now = time.time()
    age = 15 * 60 if emergency else CLEAN_MIN_AGE_SECONDS
    deleted = 0
    roots = [ROOT / 'output_coevolution/revised', ROOT / 'output_coevolution_real/revised']
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if not path.is_file() or 'publication_' not in str(path):
                continue
            match = EPISODE_RE.match(path.name)
            if path.parent.name == 'runtime':
                match = TRIP_RE.match(path.name)
            if not match or int(match.group(1)) % KEEP_EVERY == 0:
                continue
            try:
                if now - path.stat().st_mtime > age:
                    log.write('DELETE {}\n'.format(path))
                    log.flush()
                    path.unlink()
                    deleted += 1
            except FileNotFoundError:
                pass
    free = shutil.disk_usage(str(ROOT)).free
    log.write('CLEANUP {} deleted={} free_gib={:.1f} emergency={}\n'.format(
        datetime.now().isoformat(), deleted, free / float(1024 ** 3), emergency))
    log.flush()
    return free


def cleanup_loop(control, log_path):
    with log_path.open('a') as log:
        while not control.stop.is_set():
            free = cleanup_once(log)
            if free < LOW_DISK_BYTES:
                free = cleanup_once(log, emergency=True)
            if free < STOP_DISK_BYTES:
                log.write('EMERGENCY STOP: less than 20 GiB remains\n')
                log.flush()
                control.interrupt()
                return
            control.stop.wait(CLEAN_INTERVAL_SECONDS)


def run_commands(jobs, gate, workers, control, retry=False):
    pending = [j for j in jobs if j['state']['status'] in ('fresh', 'resume')]
    running = {}
    results = {}
    output_lock = threading.Lock()
    readers = []
    attempts = {}

    def update(job):
        wce = job['wce'] if job['method'] in ('fixed_wce', 'online_wce') else None
        try:
            state = classify(job['key'], job['method'], job['parent'], wce)
        except Exception as exc:
            state = {'status': 'invalid', 'step': 0, 'errors': [str(exc)]}
        return dict(job, state=state)

    def finished(job, status):
        name = label(job)
        results[name] = status
        if retry and not control.stop.is_set():
            current = update(job)
            state = current['state']['status']
            if state == 'complete':
                results[name] = 0
            elif state in ('fresh', 'resume') and attempts[name] < 3:
                pending.append(current)
                print('RETRY QUEUED {} attempt={}/3'.format(name, attempts[name]+1), flush=True)
            else:
                results[name] = status or 1
                print('FAILED {} state={} attempts={}'.format(name, state, attempts[name]), flush=True)

    def forward(child, name):
        for line in child.stdout:
            with output_lock:
                sys.stdout.write('[{}] {}'.format(name, line))
                sys.stdout.flush()

    while (pending or running) and not control.stop.is_set():
        while pending and len(running) < workers and not control.stop.is_set():
            job = pending.pop(0)
            if retry:
                job = update(job)
                if job['state']['status'] not in ('fresh', 'resume'):
                    results[label(job)] = 0 if job['state']['status'] == 'complete' else 1
                    print('SKIP {} {}'.format(label(job), job['state']), flush=True)
                    continue
            attempts[label(job)] = attempts.get(label(job), 0) + 1
            print('LAUNCH {} attempt={}/3 slots={}/{}'.format(
                label(job), attempts[label(job)], len(running)+1, workers), flush=True)
            try:
                child = subprocess.Popen(command_for(job, gate), cwd=str(ROOT),
                    start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    universal_newlines=True, bufsize=1)
            except OSError as exc:
                print('LAUNCH FAILED {}: {}'.format(label(job), exc), flush=True)
                finished(job, 1)
                continue
            control.add(child)
            running[child] = job
            reader = threading.Thread(target=forward, args=(child, label(job)))
            reader.daemon = True
            reader.start()
            readers.append(reader)
        completed = []
        for child, job in list(running.items()):
            status = child.poll()
            if status is not None:
                control.remove(child)
                completed.append(child)
                finished(job, status)
        for child in completed:
            running.pop(child)
        if running and not completed:
            time.sleep(.2)
    if control.stop.is_set():
        control.interrupt()
    for child, job in list(running.items()):
        results[label(job)] = child.wait()
        control.remove(child)
    for reader in readers:
        reader.join()
    return results


def refresh(method, selection):
    return [job for job in resolve_jobs(selection) if job['method'] == method]


def print_plan(jobs):
    counts = {name: 0 for name in ('complete', 'resume', 'fresh', 'invalid')}
    saved = 0
    for job in jobs:
        state = job['state']
        counts[state['status']] += 1
        saved += state.get('step', 0)
        detail = ''
        if state['status'] == 'resume':
            detail = ' step={} checkpoint={}'.format(state['step'], state['checkpoint'])
        elif state['status'] == 'invalid':
            detail = ' errors={}'.format('; '.join(state['errors']))
        print('{:<42} {:<8}{}'.format(label(job), state['status'].upper(), detail))
    remaining = sum(GOAL - job['state'].get('step', 0) for job in jobs
                    if job['state']['status'] != 'complete')
    print('SUMMARY complete={complete} resume={resume} fresh={fresh} invalid={invalid} '
          'recoverable_steps={} remaining_steps={}'.format(saved, remaining, **counts))
    return counts


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.workers <= 8:
        raise ValueError('Workers must be between 1 and 8')
    resource_preflight(args.workers, not args.dry_run)
    selection = Path(args.selection).resolve()
    jobs = resolve_jobs(selection)
    counts = print_plan(jobs)
    print('SCHEDULER global queue: {} workers; refill across methods; at most 2 retries per task'.format(args.workers))
    if counts['invalid']:
        raise ValueError('Invalid task outputs must be inspected before training')
    gate, gate_errors = matching_gate(args.gate, args.workers)
    if args.gate and gate is None:
        raise ValueError('Selected gate is not valid for current source: ' + '; '.join(gate_errors))
    if args.force_new_gate:
        gate = None
    if args.dry_run:
        print('GATE {}'.format(gate if gate else 'NEW {}-worker gate required'.format(args.workers)))
        preview_gate = gate or Path('/new/passing/gate.json')
        for job in jobs:
            if job['state']['status'] in ('fresh', 'resume'):
                check = subprocess.run(command_for(job, preview_gate, True), cwd=str(ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                if check.returncode:
                    raise ValueError(check.stderr.strip() or check.stdout.strip())
        print('DRY RUN PASSED: no gate, training output, or cleanup was created.')
        return 0

    lock_path = Path('/tmp/cbwce_remaining_seed101.lock')
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError('Another remaining-training campaign is already running')
        if gate is None:
            gate = generate_gate(args.workers)
        log_path = ROOT / 'runs_eval/revised' / (
            'training_cleanup_' + datetime.utcnow().strftime('%Y%m%dT%H%M%SZ') + '.log')
        control = RuntimeControl()
        cleaner = threading.Thread(target=cleanup_loop, args=(control, log_path))
        cleaner.daemon = True
        cleaner.start()
        old_handlers = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda *_: control.interrupt())
        failures = []
        try:
            # Method order is priority, not a barrier: refill every free slot.
            current = resolve_jobs(selection)
            run_commands(current, gate, args.workers, control, retry=True)
        finally:
            control.stop.set()
            control.interrupt()
            cleaner.join()
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        final_jobs = resolve_jobs(selection)
        final_counts = print_plan(final_jobs)
        print('Gate: {}'.format(gate))
        print('Cleanup log: {}'.format(log_path))
        usage = shutil.disk_usage(str(ROOT))
        print('Disk free: {:.1f} GiB'.format(usage.free / float(1024 ** 3)))
        failures = sorted(set(failures + [label(j) for j in final_jobs
                              if j['state']['status'] != 'complete']))
        if failures:
            print('FAILED OR INCOMPLETE / 失败或未完成:')
            unfinished = {label(job): job for job in final_jobs
                          if job['state']['status'] != 'complete'}
            for name in failures:
                print('  ' + name)
                if name in unfinished and unfinished[name]['state']['status'] != 'invalid':
                    command = command_for(unfinished[name], gate)
                    print('    COMMAND: ' + ' '.join(shlex.quote(part) for part in command))
            return 1
        if final_counts['complete'] != 32:
            return 1
        print('ALL 32 REMAINING CONTINUATION TASKS COMPLETE.')
        return 0
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        print('ERROR / 错误: {}'.format(exc), file=sys.stderr)
        sys.exit(2)
