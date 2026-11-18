"""One complete eleven-block routine validation; patience stopping and final tests stay separate."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from experiments.protocol import settings
from experiments.validation import validation_manifest, load_manifest, worker, seed
from experiments.scenarios import definitions, block_rows
from experiments.metrics import EpisodeMetrics


class ValidationSequenceTests(unittest.TestCase):
    def load(self, manifest, split='validation'):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps(manifest))
            return load_manifest(path, manifest['network'], True, split)

    def test_frozen_design_and_protocol(self):
        p = settings()
        self.assertEqual(p['validation_repeats'],1)
        self.assertEqual(p['wce_validation_repeats'],10)
        self.assertEqual(p['stopping_rule'],'patience_v1')
        self.assertNotIn('confirmation_repeats',p)
        self.assertNotIn('wce_confirmation_repeats',p)
        for network in ('grid', 'monaco'):
            manifest = validation_manifest(network, 41001, True)
            self.assertEqual(manifest, validation_manifest(network, 41001, True))
            _, fingerprint = self.load(manifest)
            scene = manifest['scenarios'][0]
            self.assertEqual(scene['horizon'], 6600)
            self.assertEqual([b['start'] for b in scene['blocks']], list(range(0, 6600, 600)))
            self.assertEqual(sum(b['duration'] for b in scene['blocks']), 6600)
            self.assertEqual(len({seed(fingerprint, scene['id'], i, 'sumo') for i in range(10)}), 10)
            self.assertNotEqual(fingerprint, self.load(validation_manifest(network, 41002, True))[1])

    def test_reject_old_partial_or_misaligned_design(self):
        original = validation_manifest('grid', 41001, True)
        for change in ('old', 'short', 'missing', 'extra_scene', 'order', 'profile', 'gap'):
            m = copy.deepcopy(original)
            scene = m['scenarios'][0]
            if change == 'old':del m['validation_schema']
            if change == 'short':scene['horizon'] = 3600
            if change == 'missing':scene['blocks'].pop()
            if change == 'extra_scene':m['scenarios'].append(copy.deepcopy(scene))
            if change == 'order':scene['profile_names'].reverse()
            if change == 'profile':scene['blocks'][0]['profile'] = 'Uniform'
            if change == 'gap':scene['blocks'][1]['start'] += 5
            with self.subTest(change=change), self.assertRaises(ValueError):self.load(m)

    def test_final_tests_remain_3600_and_weights_bind_names(self):
        scenes = definitions('grid', 'seen') + definitions('grid', 'test')
        self.assertEqual(len(scenes), 23)
        self.load(dict(version='8.1', network='grid', split='test', frozen=True, scenarios=scenes), 'test')
        self.assertTrue(all(s['horizon'] == 3600 for s in scenes))
        groups = [dict(name=str(i), rows=[('o', 'd', float(i + 1))]) for i in range(11)]
        definition = dict(profile_names=[str(i) for i in range(11)])
        self.assertEqual(block_rows(groups[::-1], definition, dict(weights=[1.] + [0.] * 10)), [('o', 'd', 1.)])

    def test_worker_runs_whole_sequence_without_block_resets(self):
        manifest = validation_manifest('grid', 41001, True)
        class Environment:
            def __init__(self):self.reset_horizons=[];self.scheduled=0;self.audit=False
            def reset_episode(self, random_seed, evaluation=True, horizon=6600):
                self.reset_horizons.append(horizon);self.time=0;self.metrics=EpisodeMetrics(['lane'])
                return [[0.]]
            def inject(self, vehicles):pass
            def step(self, actions):
                for _ in range(5):
                    self.time+=1
                    self.metrics.add(self.time,[1+(self.time-1)//600])
                return [[0.]], [0.], self.time == self.reset_horizons[-1]
            def terminate(self):pass
        class Controller:
            def __init__(self):self.rng=SimpleNamespace(rng={});self.resets=0;self.actions=0
            def reset(self):self.resets+=1
            def act(self,*args):self.actions+=1;return [0],[],[]
            def close(self):pass
        env=Environment();ctrl=Controller()
        request=dict(network='grid',controller='dql',seed=9001,checkpoint='fixture',
                     manifest=manifest,manifest_hash='fixed',repeats=1)
        with patch('agents.controller.Controller',return_value=ctrl), \
             patch('envs.experiment_env.make_environment',return_value=env), \
             patch('experiments.checkpoint.load_checkpoint'), \
             patch('experiments.checkpoint.inspect_checkpoint',return_value={'hash':'fixture'}), \
             patch('experiments.validation.traffic',return_value=([], 'traffic')) as traffic:
            result=worker(request)
            self.assertEqual(ctrl.actions, 1320)
            self.assertEqual(ctrl.resets, 1)
            self.assertEqual(len(env.reset_horizons), 2)  # Model setup + one rollout; never per block.
            self.assertEqual(len(result['rollouts']), 1)
            self.assertEqual(result['rollouts'][0]['sample_count'], 6600)
            self.assertEqual([b['mean_queue'] for b in result['rollouts'][0]['blocks']], list(range(1,12)))
            self.assertEqual(result['score'], 6.)
            # A short pilot must still traverse all eleven groups.
            request.update(start=0,repeats=1,pilot_horizon=110,pilot_block_seconds=10)
            result=worker(request)
            scene=traffic.call_args[0][1]
            self.assertEqual([b['start'] for b in scene['blocks']],list(range(0,110,10)))
            self.assertEqual(len(result['rollouts'][0]['blocks']),11)
            self.assertEqual(result['rollouts'][0]['sample_count'],110)
            request['pilot_horizon']=100
            with self.assertRaisesRegex(ValueError, 'complete sequence'):
                worker(request)

if __name__ == '__main__':unittest.main()
