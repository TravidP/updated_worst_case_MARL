"""Serializable empirical-plateau state; selection and patience are separate."""
import numpy as np
from experiments.protocol import settings

class Selection:
    def __init__(self, stage, protocol=None, state=None):
        self.p=protocol or settings(); self.stage=stage
        self.sign=-1 if stage=='wce' else 1
        self.history=[]; self.best=None; self.anchor=None; self.anchor_episode=0; self.improved=False
        if state:
            for key,value in state.items(): setattr(self,key,value)

    def state(self):
        return {k:getattr(self,k) for k in ('history','best','anchor','anchor_episode','improved')}

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
        self.history.append(row)
        return better

    def stable(self, scores, best_score):
        if len(scores)<self.p['window']: return False
        a=np.asarray(scores[-self.p['window']:],dtype=float)
        if not np.isfinite(a).all(): return False
        scale=max(float(a.mean()),1.)
        slope=float(np.polyfit(np.arange(len(a)),a,1)[0])
        gap=max(0.,self.sign*(a[-1]-best_score))
        return (float(np.ptp(a))<=self.p['range_tolerance']*scale and
                abs(slope)<=self.p['slope_tolerance']*scale and
                gap<=self.p['gap_tolerance']*max(best_score,1.))

    def best_through(self, episode):
        candidates=[r for r in self.history if r['episode']<=episode]
        if not candidates:raise ValueError('No evaluated checkpoint within budget')
        return min(candidates,key=lambda r:(self.sign*r['score'],r['episode'])).copy()

    def candidate(self):
        if not self.history: return False
        episode=self.history[-1]['episode']
        return (episode>=self.p['min_episodes'][self.stage] and
                episode-self.anchor_episode>=self.p['patience']*self.p['validation_every'] and
                (self.stage=='continue' or self.improved) and
                self.stable([r['score'] for r in self.history],self.best['score']))

    def terminal(self, confirmed=False):
        if confirmed: return 'plateau_confirmed'
        if self.history[-1]['episode']>=self.p['max_episodes'][self.stage]:
            return 'no_learning' if self.stage!='continue' and not self.improved else 'max_budget_unconfirmed'
        return None
