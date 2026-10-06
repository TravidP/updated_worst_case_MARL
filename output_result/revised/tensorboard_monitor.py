"""Read-only live JSONL -> TensorBoard exporter. Does not alter training inputs/code.

Usage: deeprlsc Python tensorboard_monitor.py --output NEW_DIRECTORY
Includes completed and newly appearing non-pilot revised training runs.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
ROOTS = ['runs', 'output_adversary', 'output_adversary_monaco',
         'output_coevolution', 'output_coevolution_real']
TAGS = {'mean_queue': 'queue/mean_total_vehicles_recent_600_seconds',
        'learning_steps': 'progress/controller_learning_steps',
        'episode': 'progress/episode', 'wce_updates': 'progress/wce_updates',
        'wall_seconds': 'timing/elapsed_seconds'}


class Mirror:
    def __init__(self, path, output):
        self.path = path
        self.manifest = json.loads((path.parent / 'manifest.json').read_text())
        self.offset = 0
        self.finished = False
        self.rows = 0
        self.start_time = (path.parent / 'manifest.json').stat().st_mtime
        self.label = '/'.join([self.manifest['network'], self.manifest['controller'],
            'seed_' + str(self.manifest['seed']),
            self.manifest['method'] if self.manifest['stage'] == 'continue' else self.manifest['stage'],
            path.parent.name])
        self.writer = tf.summary.FileWriter(str(output / 'events' / self.label))

    def write(self, values, step, elapsed):
        summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=float(v))
                                   for k, v in values.items() if v is not None and math.isfinite(float(v))])
        # Reconstruct observation wall time from run creation plus logged elapsed time.
        self.writer.add_event(tf.Event(wall_time=self.start_time + float(elapsed),
                                      step=int(step), summary=summary))

    def poll(self):
        count = 0
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    break  # leave a partially appended line for the next poll
                row = json.loads(line.decode('utf8'))
                step = row['simulation_steps']
                values = {tag: row[key] for key, tag in TAGS.items() if key in row}
                values['progress/stage_simulation_steps'] = step
                if row.get('goal'):
                    values['progress/stage_percent'] = 100. * step / row['goal']
                self.write(values, step, row.get('wall_seconds', 0))
                self.offset = stream.tell()
                self.rows += 1
                count += 1
        result_path = self.path.parent / 'result.json'
        status = 'no final result (may be running)'
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
            except ValueError:
                result = {}  # result writer may not yet have finished its atomic-length write
            status = result.get('status', status)
            if not self.finished and status in ('complete', 'failed', 'interrupted'):
                step = result.get('stage_simulation_steps', result.get('completed_stage_steps', 0))
                values = {'status/complete': int(status == 'complete')}
                if status == 'complete':
                    values['progress/stage_percent'] = 100.
                    values['progress/controller_learning_steps'] = result['learning_steps']
                    values['progress/stage_simulation_steps'] = step
                self.write(values, step, result.get('wall_seconds', 0))
                self.finished = True
        self.writer.flush()
        return {'source': str(self.path), 'tensorboard_run': self.label,
                'status': status, 'progress_points': self.rows, 'new_points': count}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--port', type=int, default=6006)
    args = p.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'README.md').write_text(
        '# Live TensorBoard mirror\n\nOriginal training records are read-only. '
        'Each series is exported from progress.jsonl, refreshed every 10 seconds. '
        'New publication training runs under the five revised output roots are discovered automatically.\n\n'
        'The horizontal step axis is stage-local joint simulation transitions (5 simulated seconds each). '
        'The queue metric is the recent 600-second mean total queue on controlled incoming lanes, '
        'not an episode average or evaluation score. It is normally logged every 120 transitions. '
        'A completion marker is exported at the exact final step; no missing queue point is invented. '
        'Losses were not recorded and cannot be reconstructed. Wall times are approximated from the '
        'run manifest modification time plus recorded elapsed seconds.\n')
    mirrors = {}

    def poll():
        for root in ROOTS:
            for path in (ROOT / root / 'revised').glob('*/*/seed_*/*/*/progress.jsonl'):
                if path in mirrors:
                    continue
                try:
                    manifest = json.loads((path.parent / 'manifest.json').read_text())
                except (OSError, ValueError):
                    continue
                if manifest.get('pilot') or manifest.get('stage') not in ('parent', 'wce', 'continue'):
                    continue
                mirrors[path] = Mirror(path, output)
                print('Watching: ' + mirrors[path].label, flush=True)
        records = [m.poll() for m in mirrors.values()]
        tmp = output / 'runs.json.tmp'
        tmp.write_text(json.dumps({'updated_utc': datetime.datetime.utcnow().isoformat(),
                                  'runs': records}, indent=2))
        tmp.replace(output / 'runs.json')

    def stop(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, stop)
    server = None
    try:
        poll()
        command = [str(Path(sys.executable).parent / 'tensorboard'), '--logdir', str(output / 'events'),
                   '--host', '127.0.0.1', '--port', str(args.port), '--reload_interval', '10',
                   '--window_title', 'CB-WCE training history', '--samples_per_plugin', 'scalars=0']
        with (output / 'tensorboard.log').open('w') as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        print('TensorBoard: http://127.0.0.1:{}/#scalars'.format(args.port), flush=True)
        while server.poll() is None:
            time.sleep(10)
            poll()
        raise RuntimeError('TensorBoard stopped; inspect ' + str(output / 'tensorboard.log'))
    except KeyboardInterrupt:
        print('Stopping monitor; training processes are untouched.', flush=True)
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            server.wait()
        for mirror in mirrors.values():
            mirror.writer.close()


if __name__ == '__main__':
    main()
