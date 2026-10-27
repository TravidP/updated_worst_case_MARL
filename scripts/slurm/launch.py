#!/usr/bin/env python3
"""Trusted login-produced job matrix; isolated extraction and signal forwarding."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

job=json.loads(Path(sys.argv[1]).read_text())[int(sys.argv[2])]
release=Path(job['code_release']).resolve();meta=json.loads(Path(str(release)+'.manifest.json').read_text())
if sha(release)!=meta['archive_sha256']:raise ValueError('Code release checksum mismatch')
image=Path(job['image']).resolve()
if sha(image)!=job['image_sha256']:raise ValueError('Image checksum mismatch')
work=Path(os.environ['TMPDIR'])/'cbwce';work.mkdir(exist_ok=False)
with tarfile.open(release) as tar:
    for member in tar.getmembers():
        if member.issym() or member.islnk() or (work/member.name).resolve().is_relative_to(work) is False:
            raise ValueError('Unsafe release member')
    tar.extractall(work)
for relative,expected in meta['files'].items():
    if sha(work/relative)!=expected:raise ValueError('Extracted release checksum mismatch')
runtime_lock=json.loads((work/'packaging/runtime.lock.json').read_text())
if runtime_lock['image_sha256']!=job['image_sha256']:raise ValueError('Code release and image lock differ')
gate_path=Path(job['argv'][job['argv'].index('--gate')+1])
gate=json.loads(gate_path.read_text())
if gate.get('status')!='passed' or gate.get('runtime')!=runtime_lock['runtime']:
    raise ValueError('Verification must pass inside this exact locked runtime')
env=os.environ.copy();bindings=[str(work),os.environ['TMPDIR']]
for kind in ['data','run','cache']:
    path=Path(job[kind+'_root']).resolve();path.mkdir(parents=True,exist_ok=True)
    bindings.append(str(path));env['APPTAINERENV_CBWCE_'+kind.upper()+'_ROOT']=str(path)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TF_ENABLE_ONEDNN_OPTS','MPLCONFIGDIR','TMPDIR']:
    if k in env:env['APPTAINERENV_'+k]=env[k]
command=['apptainer','exec','--cleanenv','--bind',','.join(bindings),str(image),'python',str(work/'main.py')]+job['argv']
child=subprocess.Popen(command,cwd=str(work),env=env,start_new_session=True)
def stop(signum,frame):
    try:os.killpg(child.pid,signal.SIGUSR1)
    except ProcessLookupError:pass
for s in [signal.SIGUSR1,signal.SIGTERM,signal.SIGINT]:signal.signal(s,stop)
code=child.wait()
output=Path(job['argv'][job['argv'].index('--output')+1])
if output.is_dir():
    task=os.environ.get('SLURM_ARRAY_TASK_ID');array=os.environ.get('SLURM_ARRAY_JOB_ID')
    receipt=dict(exit_code=code,job_id=(array+'_'+task) if array and task else os.environ.get('SLURM_JOB_ID'))
    fd,tmp=tempfile.mkstemp(dir=output,prefix='.scheduler-')
    with os.fdopen(fd,'w') as f:json.dump(receipt,f)
    os.replace(tmp,output/'scheduler.json')
sys.exit(code)
