"""Publication runtime identity; pilot runs may exercise other installations."""
import platform
import subprocess

def identity():
    import numpy
    import tensorflow
    import traci
    import sumolib
    return dict(python=platform.python_version(),numpy=numpy.__version__,tensorflow=tensorflow.__version__,
                sumo=subprocess.check_output(['sumo','--version'],text=True).splitlines()[0],
                traci=getattr(traci,'__version__','unknown'),sumolib=getattr(sumolib,'__version__','unknown'))

def verify_runtime():
    from experiments.protocol import ROOT
    import json
    actual=identity()
    if actual['python'].split('.')[:2]!=['3','10'] or actual['numpy']!='1.26.4' or actual['tensorflow']!='2.15.1':
        raise ValueError('Publication runtime does not match the v8.1 lock')
    if '1.26.0' not in actual['sumo'] or any(actual[k]!='1.26.0' for k in ('traci','sumolib')):
        raise ValueError('SUMO/TraCI/sumolib must all be 1.26.0')
    lock=ROOT/'packaging/runtime.lock.json'
    if not lock.exists():raise ValueError('Publication requires a verified container/runtime lock')
    frozen=json.loads(lock.read_text())
    if frozen.get('runtime')!=actual or not frozen.get('image_sha256'):
        raise ValueError('Runtime or image lock mismatch')
    return actual
