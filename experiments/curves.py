"""Read compact training and validation series without rereading historical rollouts each poll."""
import gzip
import json
import os
import threading
from functools import lru_cache

from experiments.dispatch import read_json
from experiments.storage import read_records


def point(row, kind):
    values=dict(episode=row['episode'],attempt=row.get('attempt',''))
    if kind=='training':
        values.update(queue=row['mean_queue'],rewards=row.get('rewards',{}))
    else:
        rollouts=row.get('rollouts',[])
        values.update(score=row['score'],rewards={})
        if rollouts:
            for role in rollouts[0].get('rewards',{}):
                stats=[r['rewards'][role] for r in rollouts]
                count=sum(s['samples'] for s in stats)
                values['rewards'][role]={k:sum(s[k]*s['samples'] for s in stats)/count for k in ('raw','learner')}
    return values


class CurveLog:
    def __init__(self):
        self.lock=threading.Lock()
        self.rows=[]
        self.offset=0
        self.identity=None

    def read(self,path,kind):
        try:f=path.open('rb')
        except FileNotFoundError:return []
        with self.lock,f:
            stat=os.fstat(f.fileno())
            identity=(stat.st_dev,stat.st_ino)
            if identity!=self.identity or stat.st_size<self.offset:
                self.rows=[];self.offset=0;self.identity=identity
            if stat.st_size>self.offset:
                f.seek(self.offset)
                # append_record publishes one complete gzip member per line. If a
                # writer is inside a member, defer this suffix to the next poll.
                suffix=f.read(stat.st_size-self.offset)
                try:
                    data=gzip.decompress(suffix)
                except EOFError:
                    return list(self.rows)
                except gzip.BadGzipFile:
                    if not suffix.endswith(b'\x1f'):
                        raise
                    # Reading just the first magic byte of the next gzip header
                    # raises BadGzipFile even if the preceding members are valid.
                    gzip.decompress(suffix[:-1])
                    return list(self.rows)
                new_rows=[point(json.loads(line),kind) for line in data.splitlines() if line]
                self.rows.extend(new_rows)
                self.offset=stat.st_size
            return list(self.rows)


@lru_cache(maxsize=24)
def log_cache(path):
    return CurveLog()


def current_series(rows,attempts):
    """Exclude work superseded by rollback, even before the new attempt emits a point."""
    by_attempt={}
    for r in rows:by_attempt.setdefault(r['attempt'],[]).append(r)
    current={}
    if not attempts:
        return list({r['episode']:r for r in rows}.values())
    for a in attempts:
        boundary=a['resume_episode']
        current={e:r for e,r in current.items() if e<=boundary}
        for r in by_attempt.get(a['attempt'],[]):current[r['episode']]=r
    return [current[e] for e in sorted(current)]


def read_curves(directory):
    attempts=read_records(directory/'attempts.jsonl')
    result={}
    for kind,name in [('training','episodes.jsonl.gz'),('validation','validation.jsonl.gz')]:
        path=directory/name
        result[kind]=current_series(log_cache(path).read(path,kind),attempts)
    result['live']=read_json(directory/'progress.json',{})
    return result
