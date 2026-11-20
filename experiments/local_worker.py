"""Durable child supervisor with a single owner and a verifiable exit receipt."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from experiments.storage import atomic_json


def main():
    request=Path(sys.argv[1]);spec=json.loads(request.read_text());child=None
    def forward(signum,frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
        signal.signal(sig,forward)
    with request.with_suffix('.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        owner=request.with_suffix('.owner.json')
        if owner.exists() or Path(spec['receipt']).exists():return
        start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
        atomic_json(owner,dict(pid=os.getpid(),start=start,request_hash=spec['request_hash']))
        started=time.time()
        try:
            with open(spec['log'],'a') as log:
                child=subprocess.Popen(spec['argv'],cwd=spec['cwd'],env=dict(os.environ,**spec['environment']),
                                       stdout=log,stderr=subprocess.STDOUT)
                code=child.wait()
            receipt=dict(request_hash=spec['request_hash'],exit_code=code,started=started,ended=time.time(),argv=spec['argv'])
        except BaseException as exc:
            receipt=dict(request_hash=spec['request_hash'],exit_code=1,started=started,ended=time.time(),error=repr(exc))
        atomic_json(spec['receipt'],receipt)
        sys.exit(0 if receipt['exit_code']==0 else 1)

if __name__=='__main__':main()
