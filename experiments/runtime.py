"""CPU thread configuration and explicit local/container publication locks."""
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path


def configure_threads(intra=None, inter=None):
    """Call before importing NumPy/TensorFlow. Explicit inherited settings persist."""
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        os.environ.setdefault(name, '1')
    if intra is not None:
        os.environ['TF_NUM_INTRAOP_THREADS'] = str(intra)
    if inter is not None:
        os.environ['TF_NUM_INTEROP_THREADS'] = str(inter)
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        if int(os.environ[name]) < 1:
            raise ValueError('Positive thread count required: ' + name)


def session_config(config):
    from tf_compat import tf
    options = tf.ConfigProto(
        intra_op_parallelism_threads=int(os.environ.get('TF_NUM_INTRAOP_THREADS',
            config.getint('tf_intra_op_threads', fallback=1))),
        inter_op_parallelism_threads=int(os.environ.get('TF_NUM_INTEROP_THREADS',
            config.getint('tf_inter_op_threads', fallback=1))), device_count={'GPU': 0})
    timeout = config.getint('tf_operation_timeout_ms', fallback=0)
    if timeout > 0:
        options.operation_timeout_in_ms = timeout
    return options


def identity():
    from experiments.core import file_hash
    from experiments.protocol import ROOT
    import numpy
    import tensorflow
    import traci
    import sumolib
    executable = shutil.which('sumo')
    if executable is None:
        raise ValueError('SUMO executable is not on PATH')
    packages = {}
    for line in (ROOT/'packaging/requirements.in').read_text().splitlines():
        if '==' in line:
            name = line.split('==')[0]
            try: packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: packages[name] = None
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu = next((line.split(':', 1)[1].strip() for line in
                    Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), cpu)
    return dict(python=platform.python_version(), numpy=numpy.__version__, tensorflow=tensorflow.__version__,
        sumo=subprocess.check_output([executable, '--version'], text=True).splitlines()[0],
        sumo_sha256=file_hash(executable), traci=getattr(traci, '__version__', 'unknown'),
        sumolib=getattr(sumolib, '__version__', 'unknown'), packages=packages,
        dependency_lock_sha256=file_hash(ROOT/'packaging/requirements.lock'),
        system=platform.platform(), cpu=cpu,
        threads={name: os.environ.get(name, '1') for name in
                 ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                  'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')},
        onednn=os.environ.get('TF_ENABLE_ONEDNN_OPTS', '0'))


def check_baseline(actual):
    from experiments.protocol import ROOT
    if actual['python'].split('.')[:2] != ['3', '10']:
        raise ValueError('Publication requires Python 3.10')
    if not re.search(r'\bVersion 1\.26\.0\b|\bv?1\.26\.0\b', actual['sumo']):
        raise ValueError('Publication requires released SUMO 1.26.0 (not a development build)')
    if actual['traci'] != '1.26.0' or actual['sumolib'] != '1.26.0':
        raise ValueError('SUMO/TraCI/sumolib must all be 1.26.0')
    for line in (ROOT/'packaging/requirements.in').read_text().splitlines():
        if '==' in line:
            name, version = line.split('==')
            if actual['packages'].get(name) != version:
                raise ValueError('Publication dependency mismatch: ' + line)
    if actual['numpy'] != '1.26.4' or actual['tensorflow'] != '2.15.1':
        raise ValueError('Loaded NumPy/TensorFlow differs from the dependency lock')


def lock_path(path=None):
    from experiments.protocol import ROOT
    return Path(path or os.environ.get('CBWCE_RUNTIME_LOCK', ROOT/'packaging/runtime.lock.json')).resolve()


def create_local_lock(path):
    from experiments.core import digest
    from experiments.storage import atomic_json
    path = lock_path(path)
    if path.exists(): raise FileExistsError(path)
    actual = identity()
    check_baseline(actual)
    subprocess.run([os.sys.executable, '-m', 'pip', 'check'], check=True)
    body = dict(schema='runtime_lock_v1', profile='local', runtime=actual)
    atomic_json(path, dict(body, hash=digest(body)))
    return path


def verify_runtime(path=None):
    from experiments.core import digest
    actual = identity()
    check_baseline(actual)
    path = lock_path(path)
    if not path.is_file(): raise ValueError('Publication requires an explicit local/container runtime lock')
    frozen = json.loads(path.read_text())
    if frozen.get('profile') == 'local':
        body = {k: v for k, v in frozen.items() if k != 'hash'}
        if frozen.get('schema') != 'runtime_lock_v1' or frozen.get('hash') != digest(body):
            raise ValueError('Invalid local runtime lock')
    elif not frozen.get('image_sha256'):
        raise ValueError('Container runtime lock requires image_sha256')
    if frozen.get('runtime') != actual:
        raise ValueError('Runtime lock mismatch; verify this environment again')
    return actual


def verify_gate(path):
    from experiments.runner import fingerprint
    actual = verify_runtime()
    gate = json.loads(Path(path).read_text())
    cases = {n+'_'+a for n in ('grid', 'monaco') for a in ('ma2c', 'dql', 'ppo')}
    if (gate.get('status') != 'passed' or gate.get('fingerprint') != fingerprint() or
            gate.get('runtime') != actual or set(gate.get('cases', [])) != cases):
        raise ValueError('Missing/stale/incomplete verification gate or runtime mismatch')
    checks=gate.get('checks',{})
    if (checks.get('full_validation')!=['grid','monaco'] or checks.get('full_evaluation')!=['grid','monaco'] or
            not all(checks.get(k) is True for k in ('concurrent_isolation','crash_resume','evaluation_resume','frozen_policy_resume'))):
        raise ValueError('Publication requires the extended verification checks')
    return actual
