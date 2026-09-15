"""Experiment invariants, independent random streams, and immutable records."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

METHODS = ('baseline', 'random_group', 'domain_randomization', 'fixed_wce', 'online_wce')
ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    files = [p for folder in ('experiments', 'revision', 'agents', 'envs', 'tests') for p in (ROOT / folder).rglob('*.py')]
    files += [ROOT / 'main.py', ROOT / 'tf_compat.py', ROOT / 'large_grid/data/build_file.py']
    files += list((ROOT / 'config' / 'revised').glob('*.json'))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    """Exclusive creation prevents an attempt from replacing historical inputs."""
    with open(str(path), 'x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


class Streams:
    NAMES = ('demand', 'policy', 'wce', 'replay', 'sumo', 'selection', 'initialization')

    def __init__(self, seed):
        self.seeds = {name: int(digest([int(seed), name])[:8], 16) for name in self.NAMES}
        self.rng = {name: np.random.RandomState(s) for name, s in self.seeds.items()}

    def __getitem__(self, name):
        return self.rng[name]

    def state(self):
        return {name: rng.get_state() for name, rng in self.rng.items()}

    def restore(self, state):
        if set(state) != set(self.NAMES):
            raise ValueError('Incompatible random stream state')
        for name, value in state.items():
            self.rng[name].set_state(value)


class QueueMetric:
    def __init__(self, node_lanes, neighbors):
        self.nodes = list(node_lanes)
        self.lanes = sorted(set(lane for lanes in node_lanes.values() for lane in lanes))
        self.indices = {n: [self.lanes.index(l) for l in sorted(set(node_lanes[n]))]
                        for n in self.nodes}
        # Local attribution must reconcile with the globally deduplicated domain.
        assigned = [i for values in self.indices.values() for i in values]
        if len(assigned) != len(set(assigned)):
            raise ValueError('A monitored lane belongs to multiple controller nodes')
        self.neighbors = neighbors

    def rewards(self, samples, family):
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != len(self.lanes) or not len(samples):
            raise ValueError('Invalid queue sample dimensions')
        if not np.all(np.isfinite(samples)) or np.any(samples < 0):
            raise ValueError('Invalid queue values')
        local = np.array([samples[:, self.indices[n]].sum(axis=1).mean() for n in self.nodes])
        if family == 'ma2c':
            reward = [-local[i] - .9 * sum(local[self.nodes.index(k)] for k in self.neighbors[n])
                      for i, n in enumerate(self.nodes)]
        else:
            reward = np.repeat(-local.sum(), len(local))
        return np.asarray(reward) / 100.

    @staticmethod
    def wce(samples):
        samples = np.asarray(samples, dtype=float)
        if len(samples) != 600:
            raise ValueError('WCE requires exactly 600 per-second measurements')
        return float(samples.sum(axis=1).mean() / 100.)


def validate_rollout(rows, horizon=3600):
    times = [r['time'] for r in rows]
    if times != list(range(1, horizon + 1)):
        raise ValueError('Incomplete or duplicated evaluation timestamps')
    queues = np.asarray([r['queue'] for r in rows], dtype=float)
    if not np.all(np.isfinite(queues)) or np.any(queues < 0):
        raise ValueError('Invalid queue series')
    vehicle_seconds = sum(r['active'] for r in rows)
    return {'status': 'complete', 'horizon': horizon, 'sample_count': len(rows),
            'mean_queue': float(queues.mean()), 'integrated_queue': float(queues.sum()),
            'peak_queue': float(queues.max()),
            'mean_speed': (sum(r['speed_sum'] for r in rows) / vehicle_seconds
                           if vehicle_seconds else None),
            'speed_denominator_vehicle_seconds': vehicle_seconds,
            'inserted': sum(r['inserted'] for r in rows),
            'completed': sum(r['completed'] for r in rows),
            'remaining': rows[-1]['active'], 'pending': rows[-1]['pending'],
            'teleports': sum(r['teleports'] for r in rows),
            'collisions': sum(r['collisions'] for r in rows)}


class RunRecord:
    def __init__(self, path, inputs):
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.times = {}
        self.inputs = dict(inputs)
        self.inputs['manifest_version'] = 1
        self.inputs['source_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(ROOT)).decode().strip()
        self.inputs['source_hashes'] = source_hashes()
        self.inputs['runtime'] = {'python': platform.python_version(), 'platform': platform.platform(),
                                 'numpy': np.__version__, 'tensorflow': getattr(sys.modules.get('tensorflow'), '__version__', None),
                                 'tensorflow_threads': 1, 'logical_cpus': os.cpu_count(),
                                 'blas_threads': os.environ.get('OPENBLAS_NUM_THREADS', 'unspecified'),
                                 'omp_threads': os.environ.get('OMP_NUM_THREADS', 'unspecified'),
                                 'concurrent_workers': os.environ.get('REVISION_VERIFICATION_WORKERS', 'unspecified')}
        self.hash = digest(self.inputs)
        write_json(self.path / 'manifest.json', dict(self.inputs, manifest_hash=self.hash))

    def add_time(self, name, seconds):
        self.times[name] = self.times.get(name, 0.) + seconds

    def finish(self, status, **details):
        write_json(self.path / 'result.json', dict(details, status=status,
                   manifest_hash=self.hash, wall_seconds=time.monotonic() - self.started,
                   component_seconds=self.times))
