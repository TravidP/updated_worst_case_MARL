"""Bounded, exact per-second statistics. Audit retention is opt-in."""
from collections import deque
import numpy as np

class EpisodeMetrics:
    def __init__(self, lanes, audit=False):
        self.lanes = list(lanes); self.audit = audit
        self.count = 0; self.total = 0.; self.square = 0.; self.peak = 0.
        self.recent = deque(maxlen=600)
        self.last_lanes = np.zeros(len(lanes), dtype=np.float64)
        self.lane_sums = np.zeros((3,len(lanes)), dtype=np.float64)
        self.window_counts = np.zeros(3, dtype=np.int64)
        self.totals = dict(inserted=0, completed=0, teleports=0, collisions=0, active=0, speed_sum=0.)
        self.last = {}; self.minutes = []; self.minute_sum = 0.; self.minute_peak = 0.
        self.trace = [] if audit else None

    def add(self, second, queues, **values):
        q = np.asarray(queues, dtype=np.float64)
        if second != self.count+1 or q.shape != self.last_lanes.shape or not np.isfinite(q).all() or (q<0).any():
            raise ValueError('Invalid timestamp or queue measurement')
        self.count = second; self.last_lanes[:] = q
        total = float(q.sum()); self.total += total; self.square += total*total
        self.peak = max(self.peak,total); self.recent.append(total)
        self.lane_sums[0] += q; self.window_counts[0] += 1
        if 1200 < second <= 2400:
            self.lane_sums[1] += q; self.window_counts[1] += 1
        if second > 2400:
            self.lane_sums[2] += q; self.window_counts[2] += 1
        for key in self.totals: self.totals[key] += values.get(key,0)
        self.last = values
        self.minute_sum += total; self.minute_peak = max(self.minute_peak,total)
        if second % 60 == 0:
            self.minutes.append([second,self.minute_sum/60,self.minute_peak])
            self.minute_sum=0.; self.minute_peak=0.
        if self.audit: self.trace.append(q.copy())

    def summary(self, horizon, detail=False):
        if self.count != horizon: raise ValueError('Incomplete rollout')
        mean=self.total/horizon
        result=dict(horizon=horizon, sample_count=self.count, mean_queue=mean,
                    integrated_queue=self.total, queue_variance=max(0.,self.square/horizon-mean*mean),
                    peak_queue=self.peak, pending=self.last.get('pending',0), remaining=self.last.get('active',0),
                    mean_speed=self.totals['speed_sum']/self.totals['active'] if self.totals['active'] else None,
                    **{k:self.totals[k] for k in ('inserted','completed','teleports','collisions')})
        if detail:
            result.update(minutes=self.minutes, lanes=self.lanes,
                          lane_windows=[(a/max(int(n),1)).tolist() for a,n in zip(self.lane_sums,self.window_counts)],
                          window_counts=self.window_counts.tolist())
        return result

class Diagnostics:
    def __init__(self): self.values={}
    def add(self, rows):
        for row in rows:
            for key,value in row.items():
                if key=='agent': continue
                value=float(value)
                if not np.isfinite(value): raise FloatingPointError('Nonfinite diagnostic')
                n,s,ss,lo,hi=self.values.get(key,(0,0.,0.,value,value))
                self.values[key]=(n+1,s+value,ss+value*value,min(lo,value),max(hi,value))
        rows.clear()
    def flush(self):
        result={k:dict(count=n,mean=s/n,variance=max(0.,ss/n-(s/n)**2),minimum=lo,maximum=hi)
                for k,(n,s,ss,lo,hi) in self.values.items()}
        self.values.clear(); return result
