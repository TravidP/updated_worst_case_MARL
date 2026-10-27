"""Atomic files and canonical result shards; all writers have one owner."""
import gzip
import json
import os
import uuid
from pathlib import Path

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    try:
        with tmp.open('x') as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists(): tmp.unlink()

def append_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(str(path), 'at') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')

def read_records(path):
    path = Path(path)
    if not path.exists(): return []
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(str(path), 'rt') as f:
        return [json.loads(line) for line in f if line.strip()]
