"""Serializable patience stopping; best selection and meaningful improvement are separate."""
import numpy as np
from experiments.protocol import settings

class Selection:
    def __init__(self, stage, protocol=None, state=None):
        self.p=protocol or settings(); self.stage=stage
        self.sign=-1 if stage=='wce' else 1
        self.history=[]; self.best=None; self.anchor=None; self.anchor_episode=0; self.improved=False
        self.no_improvement_evals=0
        if state:
            if 'no_improvement_evals' not in state:
                raise ValueError('Old stopping state is incompatible with patience_v1')
            for key,value in state.items(): setattr(self,key,value)

    def state(self):
        return {k:getattr(self,k) for k in ('history','best','anchor','anchor_episode','improved','no_improvement_evals')}

    def add(self, episode, score, checkpoint):
        if not np.isfinite(score) or score<0: raise ValueError('Invalid validation score')
        if self.history and episode<=self.history[-1]['episode']: raise ValueError('Duplicate/nonmonotonic validation')
        row=dict(episode=episode,score=float(score),checkpoint=checkpoint)
        better=self.best is None or self.sign*score<self.sign*self.best['score']
        if better: self.best=row.copy()
        if self.anchor is None:
            self.anchor=float(score); self.anchor_episode=episode
        elif self.sign*(self.anchor-score)>=max(self.p['relative_delta']*self.anchor,self.p['absolute_delta']):
            self.anchor=float(score);self.anchor_episode=episode;self.improved=True
            self.no_improvement_evals=0
        else:
            self.no_improvement_evals+=1
        self.history.append(row)
        return better

    def diagnostics(self):
        """Curve statistics are reported, never used as stopping gates."""
        a=np.asarray([r['score'] for r in self.history[-self.p['window']:]],dtype=float)
        if not len(a):return {}
        return dict(no_improvement_evals=self.no_improvement_evals,
                    recent_range=float(np.ptp(a)), recent_mean=float(a.mean()),
                    recent_slope=float(np.polyfit(np.arange(len(a)),a,1)[0]) if len(a)>1 else 0.,
                    current_best_gap=max(0.,self.sign*(a[-1]-self.best['score'])))

    def best_through(self, episode):
        candidates=[r for r in self.history if r['episode']<=episode]
        if not candidates:raise ValueError('No evaluated checkpoint within budget')
        return min(candidates,key=lambda r:(self.sign*r['score'],r['episode'])).copy()

    def candidate(self):
        if not self.history:return False
        return (self.history[-1]['episode']>=self.p['min_episodes'][self.stage] and
                self.no_improvement_evals>=self.p['patience'])

    def terminal(self):
        if not self.history:return None
        if self.candidate():
            return 'no_learning' if self.stage!='continue' and not self.improved else 'early_stopped'
        if self.history[-1]['episode']>=self.p['max_episodes'][self.stage]:
            return 'no_learning' if self.stage!='continue' and not self.improved else 'max_budget_reached'
        return None
