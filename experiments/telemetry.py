"""Buffered native TensorBoard output; raw records remain the source of truth."""
import json
from pathlib import Path
import numpy as np
from tf_compat import tf


class Telemetry:
    def __init__(self, path):
        self.path = Path(path)
        self.writer = tf.summary.FileWriter(str(self.path / 'tensorboard'), max_queue=1000, flush_secs=30)
        self.learner = (self.path / 'learner_metrics.jsonl').open('a')

    def scalars(self, values, step):
        if not all(np.isfinite(float(v)) for v in values.values()):
            raise FloatingPointError('Nonfinite telemetry value')
        self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=float(v))
                                                  for k,v in values.items()]), int(step))

    def updates(self, model, prefix='learner'):
        records = model.diagnostics
        if not records:
            return
        for row in records:
            self.learner.write(json.dumps(dict(row, role=prefix)) + '\n')
        ignored = {'agent', 'learning_steps', 'macro_steps'}
        fields = sorted(set.intersection(*(set(r) for r in records)) - ignored)
        step = model.learning_steps if prefix == 'learner' else model.macro_steps
        values = {}
        for key in fields:
            a = [r[key] for r in records]
            for suffix, fn in [('mean', np.mean), ('min', np.min), ('max', np.max)]:
                values[prefix+'/'+key+'/'+suffix] = float(fn(a))
        self.scalars(values, step)
        records[:] = []

    def episode(self, env, controller, stage, episode, stage_steps, elapsed, updates):
        q = np.asarray([r['queue'] for r in env.rows])
        raw_rewards = np.asarray([
            r.get('raw_rewards', r['learner_rewards'])
            for r in env.controller_rows]).mean(axis=1)
        learner_rewards = np.asarray(
            [r['learner_rewards'] for r in env.controller_rows]).mean(axis=1)
        clip_fraction = float(np.mean([
            r.get('reward_clipped', np.zeros(len(r['learner_rewards'])))
            for r in env.controller_rows]))
        full = len(q) == 6600
        data = dict(episode=episode, stage=stage, complete_episode=full, duration_seconds=len(q),
                    learning_steps=controller.learning_steps, stage_simulation_steps=stage_steps,
                    mean_total_queue=float(q.mean()), peak_total_queue=float(q.max()),
                    stopped_vehicle_seconds=float(q.sum()),
                    mean_raw_reward=float(raw_rewards.mean()),
                    raw_reward_sum=float(raw_rewards.sum()),
                    mean_learner_reward=float(learner_rewards.mean()),
                    learner_reward_sum=float(learner_rewards.sum()),
                    mean_reward=float(learner_rewards.mean()),
                    reward_sum=float(learner_rewards.sum()),
                    reward_clip_fraction=clip_fraction, scheduled=env.scheduled,
                    pending=env.rows[-1]['pending'], remaining=env.rows[-1]['active'],
                    elapsed_seconds=elapsed, backward_calls=controller.backward_calls,
                    episode_backward_calls=controller.backward_calls-updates)
        for key in ['inserted', 'completed', 'teleports', 'collisions']:
            data[key] = sum(r[key] for r in env.rows)
        with (self.path/'episode_metrics.jsonl').open('a') as f:
            f.write(json.dumps(data)+'\n')
        prefix = ('wce_episode' if stage == 'wce' else 'train/episode') if full else 'train/partial_episode'
        self.scalars({prefix+'/'+k:v for k,v in data.items() if k not in ('stage', 'complete_episode')},
                     stage_steps//120 if stage == 'wce' else controller.learning_steps)
        self.writer.flush()
        self.learner.flush()

    def close(self):
        self.writer.close()
        self.learner.close()
