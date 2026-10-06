"""Read-only publication dashboard. No training imports, pickle loads or model writes."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = ('baseline', 'random_group', 'domain_randomization', 'fixed_wce', 'online_wce')


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def stamp(path):
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0


class Lines:
    """Incremental JSONL cache; incomplete trailing bytes are retried next scan."""
    def __init__(self):
        self.cache = {}

    def get(self, path):
        if not path.exists():
            return []
        stat = path.stat()
        entry = self.cache.get(str(path))
        if not entry or entry['inode'] != stat.st_ino or stat.st_size < entry['offset']:
            entry = dict(inode=stat.st_ino, offset=0, rows=[])
            self.cache[str(path)] = entry
        with path.open('rb') as stream:
            stream.seek(entry['offset'])
            for line in stream:
                if not line.endswith(b'\n'):
                    break
                entry['offset'] += len(line)
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        entry['rows'].append(row)
                except (ValueError, UnicodeError):
                    continue
        return entry['rows']


class Collector:
    def __init__(self, root, selection):
        self.root = Path(root).resolve()
        self.selection = Path(selection)
        self.lines = Lines()
        self.monitors = {}
        self.previous = {}

    def path(self, value):
        p = Path(value)
        p = (self.root / p).resolve() if not p.is_absolute() else p.resolve()
        if self.root not in p.parents:
            raise ValueError('Record outside repository: ' + str(p))
        return p

    def monitor(self, folder):
        summary = folder / 'summary.json'
        if not summary.exists():
            return None
        # Cache only immutable successful rounds; missing reward files are retried.
        key = (str(folder), summary.stat().st_mtime_ns)
        if key in self.monitors:
            return self.monitors[key]
        obj = read(summary)
        if obj.get('status') != 'complete':
            return None
        point = dict(step=obj['stage_simulation_steps'], episode=obj['episode'],
                     learning_steps=obj['learning_steps'], time=stamp(summary),
                     queue=obj['metrics'].get('mean_total_queue'),
                     waiting=obj['metrics'].get('mean_current_wait_seconds'),
                     path=str(folder.relative_to(self.root)))
        count = len(obj.get('rollouts', []))
        for field, source in [('raw', 'raw_rewards'), ('learner', 'learner_rewards')]:
            rewards = []
            for index in range(1, count + 1):
                rows = self.lines.get(folder / ('rollout_%02d' % index) / 'rollout.controls.jsonl')
                if len(rows) != 120 or [r.get('time') for r in rows] != list(range(5, 601, 5)):
                    continue
                if not all(r.get(source) and all(isinstance(v, (int, float)) and math.isfinite(v)
                           for v in r[source]) for r in rows):
                    continue
                rewards.append(statistics.mean(statistics.mean(r[source]) for r in rows))
            point[field] = statistics.mean(rewards) if count and len(rewards) == count else None
            point[field + '_sd'] = (statistics.stdev(rewards) if len(rewards) > 1 else 0) if point[field] is not None else None
        point['rollouts'] = count
        if point['raw'] is not None and point['learner'] is not None:
            self.monitors[key] = point
            # These immutable raw rows need not remain in RAM after aggregation.
            for index in range(1, count + 1):
                self.lines.cache.pop(str(folder / ('rollout_%02d' % index) / 'rollout.controls.jsonl'), None)
        return point

    def identity(self, m, item, stage, method, parent_hash, wce_hash):
        if (m.get('network'), m.get('controller'), m.get('seed'), m.get('stage'), m.get('method'), m.get('pilot', False)) != (
                item['network'], item['controller'], item['seed'], stage, method, False):
            return False
        parents = m.get('parents', {})
        if stage != 'parent' and parents.get('controller', {}).get('hash') != parent_hash:
            return False
        if stage == 'continue' and method in ('fixed_wce', 'online_wce') and parents.get('wce', {}).get('hash') != wce_hash:
            return False
        return True

    def chain(self, leaf, manifests):
        chain, seen, limit = [], set(), float('inf')
        while leaf:
            if leaf in seen:
                raise ValueError('Cycle in resume ancestry')
            seen.add(leaf)
            m = manifests[leaf]
            chain.append((leaf, limit))
            resume = m.get('resume')
            if not resume:
                break
            cp = self.path(resume)
            if not re.fullmatch(r'checkpoint_\d+', cp.name):
                raise ValueError('Invalid resume checkpoint name: ' + str(cp))
            if cp.parent not in manifests:
                # Historical cleanup can remove ancestors; surviving logs/results
                # still describe the cumulative stage position of the selected run.
                break
            limit = min(limit, int(cp.name.split('_')[-1]))
            leaf = cp.parent
        return list(reversed(chain))

    def wall_time(self, run):
        """Elapsed time resets per attempt; never sum cumulative progress rows."""
        result_file = run / 'result.json'
        result = read(result_file) if result_file.exists() else {}
        seconds = result.get('wall_seconds')
        final = isinstance(seconds, (float, int)) and math.isfinite(seconds) and seconds >= 0
        if not final:
            values = [r.get('wall_seconds') for r in self.lines.get(run / 'progress.jsonl')]
            seconds = max((v for v in values if isinstance(v, (float, int)) and math.isfinite(v) and v >= 0), default=0)
        return dict(path=str(run.relative_to(self.root)), seconds=seconds,
                    lower_bound=not final, source='result.json' if final else 'progress.jsonl')

    def branch(self, leaf, manifests, goal, now):
        progress, episodes, monitors, checkpoints = {}, {}, {}, []
        last = 0
        warnings = []
        chain = self.chain(leaf, manifests)
        wall_attempts = [self.wall_time(run) for run, _ in chain]
        first_manifest = manifests[chain[0][0]]
        missing_wall_history = bool(first_manifest.get('resume') and self.path(first_manifest['resume']).parent not in manifests)
        if missing_wall_history:
            warnings.append('早期恢复目录缺失或身份不匹配，历史曲线不完整；保留当前累计进度，已完成episode按日志中的回合编号显示。')
        for run, limit in chain:
            for row in self.lines.get(run / 'progress.jsonl'):
                step = row.get('simulation_steps', 0)
                if step <= limit:
                    progress[step] = row
            for row in self.lines.get(run / 'episode_metrics.jsonl'):
                step = row.get('stage_simulation_steps', 0)
                if step <= limit:
                    episodes[step] = row
            for cp in run.glob('checkpoint_[0-9]*'):
                if not re.fullmatch(r'checkpoint_\d+', cp.name):
                    continue
                step = int(cp.name.split('_')[-1])
                if step > limit or not (cp / 'manifest.json').exists():
                    continue
                try:
                    meta = read(cp / 'manifest.json')
                    if meta.get('files') and all((cp / name).is_file() for name in meta['files']):
                        checkpoints.append((step, str(cp.relative_to(self.root))))
                except (OSError, ValueError):
                    pass
            for folder in run.glob('monitoring/round_*'):
                try:
                    point = self.monitor(folder)
                    if point and point['step'] <= limit:
                        monitors[point['step']] = point
                except (OSError, ValueError, KeyError) as exc:
                    warnings.append(str(folder.relative_to(self.root)) + ': ' + str(exc))
            last = max(last, stamp(run / 'progress.jsonl'), stamp(run / 'episode_metrics.jsonl'))
        result = read(leaf / 'result.json') if (leaf / 'result.json').exists() else {}
        m = manifests[leaf]
        latest = progress[max(progress)] if progress else {}
        cp_step, cp_path = max(checkpoints, default=(0, None))
        step = max(max(progress, default=0), max(episodes, default=0), cp_step, result.get('stage_simulation_steps', 0))
        if latest.get('goal'):
            goal = latest['goal']
        elif m.get('steps') and m['stage'] != 'wce':
            goal = m['steps']
        elif m['stage'] == 'wce' and m.get('episodes'):
            goal = m['episodes'] * 1320
        complete = result.get('status') == 'complete' and result.get('stage_simulation_steps') == goal
        status = 'complete' if complete else 'unknown'
        if not complete:
            if result and result.get('status') != 'complete':
                status = 'failed'
            elif last and now - last < 3600:
                status = 'recent'
            elif last:
                status = 'stale'
        # A recent incomplete round is evidence of monitor activity, not proof of a live process.
        pending = [f for f in leaf.glob('monitoring/round_*')
                   if not (f / 'summary.json').exists() and not (f / 'failure.json').exists()
                   and now - max(stamp(f), stamp(f / 'runtime')) < 1800]
        if pending and status not in ('complete', 'failed'):
            status = 'monitoring'
        training_fields = ('episode', 'stage_simulation_steps', 'complete_episode',
                           'mean_raw_reward', 'mean_learner_reward')
        series = [{key: episodes[k].get(key) for key in training_fields} for k in sorted(episodes)]
        return dict(path=str(leaf.relative_to(self.root)), status=status, step=step, goal=goal,
                    percent=round(100 * min(step / goal, 1), 2),
                    episode=max(latest.get('episode', 0), max((r['episode'] for r in series), default=0)),
                    completed_episodes=max((r['episode'] for r in series if r.get('complete_episode')), default=0),
                    partial_episodes=[r['episode'] for r in series if not r.get('complete_episode')],
                    learning_steps=result.get('learning_steps', latest.get('learning_steps')),
                    wce_updates=result.get('wce_updates', latest.get('wce_updates', 0)),
                    checkpoint_step=cp_step, checkpoint=cp_path, last_log=last,
                    last_monitor=max((r['time'] for r in monitors.values()), default=0),
                    monitor=[monitors[k] for k in sorted(monitors)], training=series,
                    wall_seconds=sum(r['seconds'] for r in wall_attempts), wall_attempts=wall_attempts,
                    wall_lower_bound=missing_wall_history or any(r['lower_bound'] for r in wall_attempts),
                    warnings=warnings, process='unknown')

    def scan(self):
        now = time.time()
        selection = read(self.selection)['runs']
        protocol = read(self.root / 'config/revised/protocol.json')
        tasks = []
        for item in selection:
            parent_result = read(self.path(item['parent_result']))
            wce_result = read(self.path(item['wce_result']))
            parent_hash = read(self.path(parent_result['checkpoint']) / 'manifest.json')['hash']
            wce_meta = read(self.path(wce_result['checkpoint']) / 'manifest.json')
            if wce_meta.get('parents', {}).get('controller', {}).get('hash') != parent_hash:
                raise ValueError('Selection WCE / parent mismatch')
            for group in ('parent', 'wce') + METHODS:
                stage, method = (group, 'baseline') if group in ('parent', 'wce') else ('continue', group)
                goal = (protocol['parent_steps'] if stage == 'parent' else
                        protocol['offline_episodes'] * protocol['training_seconds'] // protocol['control_seconds']
                        if stage == 'wce' else protocol['continuation_steps'])
                if stage in ('parent', 'wce'):
                    chosen = self.path(item[stage + '_result']).parent
                    candidates = list(chosen.parent.glob('publication_*'))
                else:
                    base = ('runs' if method == 'baseline' else
                            'output_coevolution' if item['network'] == 'grid' else 'output_coevolution_real')
                    task_root = self.root / base / 'revised' / item['network'] / item['controller'] / ('seed_' + str(item['seed'])) / method
                    candidates = list(task_root.glob('publication_*'))
                manifests, warnings = {}, []
                for run in candidates:
                    try:
                        m = read(run / 'manifest.json')
                        if self.identity(m, item, stage, method, parent_hash, wce_meta['hash']):
                            manifests[run.resolve()] = m
                        else:
                            warnings.append('身份不匹配，未计入: ' + str(run.relative_to(self.root)))
                    except (OSError, ValueError) as exc:
                        warnings.append(str(run.relative_to(self.root)) + ': ' + str(exc))
                ancestors = {self.path(m['resume']).parent for m in manifests.values() if m.get('resume')}
                leaves = [self.path(chosen)] if stage in ('parent', 'wce') else [r for r in manifests if r not in ancestors]
                branches = []
                for leaf in leaves:
                    try:
                        branches.append(self.branch(leaf, manifests, goal, now))
                    except (OSError, ValueError, KeyError) as exc:
                        warnings.append(str(leaf) + ': ' + str(exc))
                # One representative per logical task, never sum independent branches.
                branches.sort(key=lambda r: (r['status'] == 'complete', r['step'], r['last_log']), reverse=True)
                task = dict(id='/'.join((group, item['network'], item['controller'], str(item['seed']))),
                            group=group, network=item['network'], controller=item['controller'], seed=item['seed'],
                            stage=stage, status='unknown' if warnings else 'fresh', step=0, goal=goal, percent=0,
                            episode=0, completed_episodes=0, partial_episodes=[], checkpoint_step=0,
                            learning_steps=None, wce_updates=0, last_log=0, last_monitor=0, process='unknown',
                            wall_seconds=0, wall_lower_bound=bool(warnings), wall_attempts=[],
                            monitor=[], training=[], branches=branches, warnings=warnings)
                if branches:
                    task.update({k: v for k, v in branches[0].items() if k != 'warnings'})
                    task['warnings'] += branches[0]['warnings']
                    # Shared resume ancestors count once, even with multiple leaves.
                    attempts = {r['path']: r for b in branches for r in b['wall_attempts']}
                    task['wall_attempts'] = list(attempts.values())
                    task['wall_seconds'] = sum(r['seconds'] for r in attempts.values())
                    task['wall_lower_bound'] = any(b['wall_lower_bound'] for b in branches)
                tasks.append(task)
        memory = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith(('MemAvailable:', 'MemTotal:')):
                key, value, _ = line.split()
                memory[key.rstrip(':')] = int(value) * 1024
        # Read-only /proc matching; absence never implies stopped (PID namespace isolation).
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                tokens = (proc / 'cmdline').read_bytes().decode().split('\0')
                if '--output' not in tokens or not any(t.endswith('main.py') for t in tokens):
                    continue
                output = self.path(tokens[tokens.index('--output') + 1])
                for task in tasks:
                    if task.get('path') and self.path(task['path']) == output:
                        task['process'] = '可见 PID ' + proc.name
            except (OSError, ValueError, IndexError, UnicodeError):
                pass
        total = sum(t['goal'] for t in tasks)
        done = sum(min(t['step'], t['goal']) for t in tasks)
        return dict(updated_at=now, tasks=tasks, total=len(tasks), completed=sum(t['status'] == 'complete' for t in tasks),
                    wall_seconds=sum(t['wall_seconds'] for t in tasks),
                    wall_lower_bound=any(t['wall_lower_bound'] for t in tasks),
                    steps=done, goal=total, remaining=total-done,
                    disk_free=shutil.disk_usage(self.root).free, memory_available=memory.get('MemAvailable'),
                    memory_total=memory.get('MemTotal'), selection=str(self.selection),
                    checkpoint_note='checkpoint 仅读取元数据和文件存在性；恢复完整性由训练启动器校验。')


class State:
    def __init__(self, collector, cache, interval):
        self.collector, self.cache, self.interval = collector, cache, interval
        self.lock = threading.Lock()
        self.data = read(cache) if cache.exists() else {}
        self.error = None
        self.next_refresh = time.time()
        self.refreshing = False
        self.event = threading.Event()

    def refresh(self):
        if not self.lock.acquire(blocking=False):
            return
        self.refreshing = True
        try:
            data = self.collector.scan()
            atomic(self.cache, data)
            self.data, self.error = data, None
        except Exception as exc:
            self.error = dict(time=time.time(), message=str(exc))
            print('SCAN ERROR', self.error, flush=True)
        finally:
            self.next_refresh = time.time() + self.interval
            self.refreshing = False
            self.lock.release()

    def payload(self):
        return dict(self.data, error=self.error, next_refresh=self.next_refresh, refreshing=self.refreshing)

    def loop(self):
        while True:
            self.event.wait(max(0, self.next_refresh-time.time()))
            self.event.clear()
            self.refresh()


def serve(args, directory):
    lock = (directory / 'service.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = State(Collector(ROOT, args.selection), directory / 'snapshot.json', args.interval)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, data, mime='application/json; charset=utf-8'):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def local(self):
            hosts = ('127.0.0.1:' + str(args.port), 'localhost:' + str(args.port))
            return self.headers.get('Host') in hosts and self.headers.get('Origin', 'http://' + hosts[0]) in ['http://' + h for h in hosts]

        def do_GET(self):
            if not self.local():
                return self.send(403, {'error': 'Local access only'})
            route = urlparse(self.path).path
            if route == '/api/snapshot':
                payload = state.payload()
                tag = '"' + hashlib.sha256(json.dumps([payload.get('updated_at'), payload.get('error'),
                         payload.get('next_refresh'), payload.get('refreshing')]).encode()).hexdigest() + '"'
                if self.headers.get('If-None-Match') == tag:
                    self.send_response(304)
                    self.send_header('ETag', tag)
                    self.end_headers()
                    return
                data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('ETag', tag)
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.wfile.write(data)
                return
            if route == '/api/health':
                return self.send(200, {'service': 'training-dashboard', 'pid': os.getpid()})
            assets = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'), '/style.css': ('style.css', 'text/css')}
            if route in assets:
                name, mime = assets[route]
                return self.send(200, (HERE / name).read_bytes(), mime + '; charset=utf-8')
            return self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.local() or self.headers.get('X-Dashboard-Refresh') != '1':
                return self.send(403, {'error': 'Invalid refresh request'})
            if self.path != '/api/refresh':
                return self.send(404, {'error': 'Not found'})
            if not state.refreshing:
                state.event.set()
            return self.send(202, {'accepted': True})

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    atomic(directory / 'pid.json', dict(pid=os.getpid(), port=args.port,
                                        ticks=Path('/proc/self/stat').read_text().split()[21]))
    state.refresh()
    threading.Thread(target=state.loop, daemon=True).start()
    print('Dashboard ready http://127.0.0.1:%s' % args.port, flush=True)
    server.serve_forever()


def active(directory):
    try:
        pid = read(directory / 'pid.json')
        proc = Path('/proc') / str(pid['pid'])
        tokens = (proc / 'cmdline').read_bytes().decode().split('\0')
        return pid if str(HERE / 'dashboard.py') in tokens and 'serve' in tokens and (proc / 'stat').read_text().split()[21] == pid['ticks'] else None
    except (OSError, ValueError, KeyError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='start', choices=['start', 'serve', 'status', 'stop', 'scan'])
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--interval', type=int, default=1800)
    parser.add_argument('--selection', type=Path, default=ROOT / 'runs_eval/revised/selections/publication_seed101.json')
    args = parser.parse_args()
    if args.interval < 1 or not 1 <= args.port <= 65535:
        parser.error('Invalid interval or port')
    directory = ROOT / 'runs_eval/revised/dashboard'
    directory.mkdir(parents=True, exist_ok=True)
    if args.command == 'scan':
        data = Collector(ROOT, args.selection).scan()
        atomic(directory / 'snapshot.json', data)
        print(json.dumps({k: data[k] for k in ('total', 'completed', 'steps', 'remaining')}, ensure_ascii=False))
        return
    if args.command == 'serve':
        return serve(args, directory)
    pid = active(directory)
    if args.command == 'stop':
        if pid:
            os.kill(pid['pid'], signal.SIGTERM)
        print('Dashboard stopped' if pid else 'No visible dashboard process; no signal sent')
    elif args.command == 'status':
        print(json.dumps(pid) if pid else 'Dashboard process not visible; check localhost:8766 if running outside this PID namespace')
    elif pid:
        print('Already running: http://127.0.0.1:%s' % pid['port'])
    else:
        with (directory / 'server.log').open('ab') as log:
            proc = subprocess.Popen([sys.executable, str(HERE / 'dashboard.py'), 'serve', '--port', str(args.port),
                                     '--interval', str(args.interval), '--selection', str(args.selection.resolve())],
                                    cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                    start_new_session=True)
        for _ in range(100):
            time.sleep(.1)
            if proc.poll() is not None:
                raise SystemExit('Dashboard could not start; see ' + str(directory / 'server.log'))
            if active(directory):
                print('Dashboard: http://127.0.0.1:%s (first scan in progress)' % args.port)
                return
        raise SystemExit('Startup pending; see ' + str(directory / 'server.log'))


if __name__ == '__main__':
    main()
