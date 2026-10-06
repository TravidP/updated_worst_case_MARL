#!/usr/bin/env python3
"""Read-only local HTTP monitor for the CB-WCE Phase E publication campaign."""
import argparse
import json
import os
import shutil
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / 'runs_eval/revised/publication_seed101_v1'
INDEX = Path(__file__).resolve().parent / 'dist/index.html'
EXPECTED_ROLLOUTS = 9200
EXPECTED_SUITES = 40
METHODS = ['baseline', 'random_group', 'domain_randomization', 'fixed_wce', 'online_wce']
HISTORY = deque(maxlen=360)
LAST_HISTORY_AT = 0.0


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {} if default is None else default


def iso_time(timestamp):
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec='seconds')


def process_counts(output_root):
    counts = {'scheduler': 0, 'evaluation': 0, 'sumo': 0}
    needle = str(output_root).encode('utf-8')
    proc = Path('/proc')
    if not proc.exists():
        return counts
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ')
        except OSError:
            continue
        if needle not in command:
            continue
        if b'run_publication_parallel.py' in command:
            counts['scheduler'] += 1
        elif b'main.py experiment evaluate' in command:
            counts['evaluation'] += 1
        elif b'sumo ' in command:
            counts['sumo'] += 1
    return counts


def suite_state(suite, output_root, scheduler_results):
    output = ROOT / suite['output']
    completed_paths = list(output.glob('*/*/rollout_*/attempt_001/rollout_summary.json')) if output.exists() else []
    result_paths = list(output.glob('*/*/rollout_*/attempt_001/result.json')) if output.exists() else []
    attempts = list(output.glob('*/*/rollout_*/attempt_001')) if output.exists() else []
    completed = len(completed_paths)
    failed = 0
    for result_path in result_paths:
        if not (result_path.parent / 'rollout_summary.json').exists():
            failed += int(read_json(result_path, {}).get('status') == 'failed')
    suite_result = output / 'suite_result.json'
    scheduler_result = scheduler_results.get(suite['id'], {})
    if suite_result.is_file():
        status = 'complete'
    elif scheduler_result.get('returncode') not in (None, 0) or failed:
        status = 'failed'
    elif output.exists():
        status = 'running'
    else:
        status = 'queued'
    current = None
    unfinished = [path for path in attempts if not (path / 'result.json').exists()]
    if unfinished:
        latest = max(unfinished, key=lambda path: path.stat().st_mtime)
        parts = latest.relative_to(output).parts
        if len(parts) >= 4:
            current = {'split': parts[0], 'scenario': parts[1], 'rollout': parts[2].replace('rollout_', '')}
    latest_paths = completed_paths + result_paths
    updated_at = max([path.stat().st_mtime for path in latest_paths] + ([output.stat().st_mtime] if output.exists() else [0]))
    return {
        'id': suite['id'],
        'network': suite['id'].split('-')[1],
        'controller': suite['id'].split('-')[2],
        'method': suite['id'].split('-', 3)[3],
        'status': status,
        'completed': completed,
        'failed': failed,
        'expected': 230,
        'percent': round(completed / 230.0 * 100, 1),
        'current': current,
        'updated_at': iso_time(updated_at) if updated_at else None,
    }


def make_snapshot(output_root):
    global LAST_HISTORY_AT
    now = time.time()
    plan_path = output_root / '_scheduler_plan.json'
    result_path = output_root / '_scheduler_result.json'
    plan = read_json(plan_path, {'workers': 4, 'suites': []})
    scheduler_result = read_json(result_path, {}) if result_path.exists() else {}
    scheduler_rows = {row.get('id'): row for row in scheduler_result.get('results', [])}
    suites = [suite_state(row, output_root, scheduler_rows) for row in plan.get('suites', [])]
    completed = sum(row['completed'] for row in suites)
    failed_rollouts = sum(row['failed'] for row in suites)
    status_counts = {name: sum(row['status'] == name for row in suites)
                     for name in ('complete', 'running', 'queued', 'failed')}
    processes = process_counts(output_root)
    if scheduler_result:
        campaign_status = 'complete' if not scheduler_result.get('failed_suites') and not scheduler_result.get('interrupted') else 'failed'
    elif processes['scheduler'] or processes['evaluation']:
        campaign_status = 'running'
    elif output_root.exists():
        campaign_status = 'paused'
    else:
        campaign_status = 'waiting'
    start = plan_path.stat().st_mtime if plan_path.exists() else None
    elapsed = max(0, now - start) if start else 0
    rate_hour = completed / elapsed * 3600 if elapsed > 0 and completed else 0
    eta_seconds = (EXPECTED_ROLLOUTS - completed) / (completed / elapsed) if elapsed > 0 and completed else None
    if completed >= EXPECTED_ROLLOUTS:
        eta_seconds = 0
    disk = shutil.disk_usage(str(ROOT))
    recent = []
    recent_paths = []
    for row in suites:
        output = ROOT / next((item['output'] for item in plan.get('suites', []) if item['id'] == row['id']), '')
        if output.exists():
            recent_paths.extend(output.glob('*/*/rollout_*/attempt_001/rollout_summary.json'))
    for path in sorted(recent_paths, key=lambda item: item.stat().st_mtime, reverse=True)[:12]:
        rel = path.relative_to(output_root).parts
        if len(rel) >= 7:
            recent.append({'time': iso_time(path.stat().st_mtime), 'network': rel[0], 'controller': rel[1],
                           'method': rel[2], 'split': rel[3], 'scenario': rel[4],
                           'rollout': rel[5].replace('rollout_', '')})
    if now - LAST_HISTORY_AT >= 8:
        HISTORY.append({'time': int(now), 'completed': completed})
        LAST_HISTORY_AT = now
    return {
        'generated_at': iso_time(now),
        'campaign': {
            'status': campaign_status,
            'completed_rollouts': completed,
            'expected_rollouts': EXPECTED_ROLLOUTS,
            'failed_rollouts': failed_rollouts,
            'percent': round(completed / float(EXPECTED_ROLLOUTS) * 100, 2),
            'elapsed_seconds': int(elapsed),
            'eta_seconds': int(eta_seconds) if eta_seconds is not None else None,
            'rate_per_hour': round(rate_hour, 1),
            'started_at': iso_time(start) if start else None,
            'workers_configured': plan.get('workers', 4),
            'processes': processes,
            'suites': status_counts,
            'disk_free_gib': round(disk.free / float(1024 ** 3), 1),
            'disk_total_gib': round(disk.total / float(1024 ** 3), 1),
        },
        'suites': suites,
        'recent': recent,
        'history': list(HISTORY),
        'scheduler_result': {
            'exists': result_path.exists(),
            'completed_suites': scheduler_result.get('completed_suites'),
            'failed_suites': scheduler_result.get('failed_suites'),
            'not_started_suites': scheduler_result.get('not_started_suites'),
            'interrupted': scheduler_result.get('interrupted'),
        },
    }


class Handler(BaseHTTPRequestHandler):
    output_root = DEFAULT_OUTPUT

    def send_bytes(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = self.path.split('?', 1)[0]
        if route in ('/', '/index.html'):
            self.send_bytes(INDEX.read_bytes(), 'text/html; charset=utf-8')
        elif route == '/api/status':
            body = json.dumps(make_snapshot(self.output_root), ensure_ascii=False).encode('utf-8')
            self.send_bytes(body, 'application/json; charset=utf-8')
        elif route == '/health':
            self.send_bytes(b'{"status":"ok"}', 'application/json')
        else:
            self.send_bytes(b'Not found', 'text/plain; charset=utf-8', 404)

    def log_message(self, format_string, *args):
        if self.path != '/api/status':
            BaseHTTPRequestHandler.log_message(self, format_string, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8877)
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--snapshot', action='store_true')
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    if args.snapshot:
        print(json.dumps(make_snapshot(output_root), indent=2, ensure_ascii=False))
        return
    Handler.output_root = output_root
    server = HTTPServer((args.host, args.port), Handler)
    print('Phase E monitor: http://{}:{}/'.format(args.host, args.port), flush=True)
    print('Watching: {}'.format(output_root), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
