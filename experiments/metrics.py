"""Bounded, exact per-second statistics. Audit retention is opt-in."""
from collections import deque
import numpy as np

METRICS_SCHEMA = 'evaluation_metrics_v2'

class EpisodeMetrics:
    def __init__(self, lanes, audit=False):
        self.lanes = list(lanes); self.audit = audit
        self.count = 0; self.total = 0.; self.square = 0.; self.peak = 0.
        self.recent = deque(maxlen=600)
        self.last_lanes = np.zeros(len(lanes), dtype=np.float64)
        self.all_lane_sums = np.zeros(len(lanes), dtype=np.float64)
        self.lane_sums = np.zeros((3,len(lanes)), dtype=np.float64)
        self.window_counts = np.zeros(3, dtype=np.int64)
        self.totals = dict(inserted=0, completed=0, teleports=0, collisions=0, active=0, speed_sum=0.)
        self.last = {}; self.minutes = []; self.minute_sum = 0.; self.minute_peak = 0.
        self.trace = [] if audit else None
        self.detail_totals = None

    def enable_detail(self, horizon):
        if self.count:raise ValueError('Detailed metrics must start before the first second')
        self.detail_totals = np.empty(horizon, dtype=np.float64)

    def add(self, second, queues, **values):
        q = np.asarray(queues, dtype=np.float64)
        if second != self.count+1 or q.shape != self.last_lanes.shape or not np.isfinite(q).all() or (q<0).any():
            raise ValueError('Invalid timestamp or queue measurement')
        self.count = second; self.last_lanes[:] = q
        total = float(q.sum()); self.total += total; self.square += total*total
        self.peak = max(self.peak,total); self.recent.append(total)
        self.all_lane_sums += q
        if second <= 1200:
            self.lane_sums[0] += q; self.window_counts[0] += 1
        if 1200 < second <= 2400:
            self.lane_sums[1] += q; self.window_counts[1] += 1
        if 2400 < second <= 3600:
            self.lane_sums[2] += q; self.window_counts[2] += 1
        if self.detail_totals is not None:self.detail_totals[second-1] = total
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
        result=dict(metrics_schema=METRICS_SCHEMA,horizon=horizon, sample_count=self.count, mean_queue=mean,
                    integrated_queue=self.total, queue_variance=max(0.,self.square/horizon-mean*mean),
                    peak_queue=self.peak, pending=self.last.get('pending',0), remaining=self.last.get('active',0),
                    mean_speed=self.totals['speed_sum']/self.totals['active'] if self.totals['active'] else None,
                    **{k:self.totals[k] for k in ('inserted','completed','teleports','collisions')})
        result['completion_rate'] = self.totals['completed']/self.totals['inserted'] if self.totals['inserted'] else None
        result['active_vehicle_seconds'] = self.totals['active']
        result['speed_sum'] = self.totals['speed_sum']
        if detail:
            result.update(minutes=self.minutes, lanes=self.lanes,
                          lane_window_names=['pre_peak','peak','post_peak'],
                          lane_window_bounds=[[0,1200],[1200,2400],[2400,3600]],
                          lane_full_mean=(self.all_lane_sums/horizon).tolist(),
                          lane_windows=[(a/max(int(n),1)).tolist() for a,n in zip(self.lane_sums,self.window_counts)],
                          window_counts=self.window_counts.tolist())
            totals = self.detail_totals
            if totals is None and self.trace is not None:totals=np.asarray(self.trace).sum(axis=1)
            if totals is None:raise ValueError('Detailed metrics were not enabled before this rollout')
            result['p95_queue']=float(np.percentile(totals,95))
            tail=np.asarray(self.recent,dtype=float)
            x=np.arange(len(tail),dtype=float);x-=x.mean()
            result['last_600_s_slope']=float(np.dot(x,tail)/np.dot(x,x)) if len(tail)==600 else None
            result['window_mean_queue']=[float(a.sum()/n) if n else None for a,n in zip(self.lane_sums,self.window_counts)]
            result['window_integrated_queue']=[float(a.sum()) for a in self.lane_sums]
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


class RewardMean:
    """Mean immediate reward per learner sample, before and after the learner transform."""
    def __init__(self):
        self.raw = 0.
        self.learner = 0.
        self.count = 0

    def add(self, raw, learner):
        raw = np.asarray(raw)
        learner = np.asarray(learner)
        if raw.shape != learner.shape:
            raise ValueError('Raw and learner rewards must have matching shapes')
        self.raw += float(raw.sum(dtype=np.float64))
        self.learner += float(learner.sum(dtype=np.float64))
        self.count += raw.size

    def summary(self):
        return dict(raw=self.raw/self.count, learner=self.learner/self.count, samples=self.count)
