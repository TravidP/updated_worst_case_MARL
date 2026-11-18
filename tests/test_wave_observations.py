"""Wave-only observation contract for both networks, controllers and WCE."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from envs.env import TrafficSimulator
from envs.experiment_env import RevisedMixin
from envs.large_grid_env import STATE_NAMES as GRID_STATES
from envs.real_net_env import STATE_NAMES as MONACO_STATES
from experiments.configuration import load_controller_config
from experiments.checkpoint import save_checkpoint, load_checkpoint
from agents.controller import Controller
from agents.wce import WCE
from test_learning import FakeEnv


class WaveObservationTests(unittest.TestCase):
    def test_configs_only_allow_wave(self):
        self.assertEqual(GRID_STATES, ['wave'])
        self.assertEqual(MONACO_STATES, ['wave'])
        for network in ('grid', 'monaco'):
            for family in ('ma2c', 'dql', 'ppo'):
                config, _ = load_controller_config(network, family)
                self.assertEqual(config['ENV_CONFIG']['observation'], 'wave')
                self.assertNotIn('num_ft', config['MODEL_CONFIG'])
                for key in ('norm_wait', 'clip_wait', 'fast_wait_metric'):
                    self.assertNotIn(key, config['ENV_CONFIG'])
                for key in ('seed', 'test_seeds'):
                    del config['ENV_CONFIG'][key]
                config['ENV_CONFIG']['observation'] = 'wave,wait'
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'invalid.ini'
                    with path.open('w') as stream:
                        config.write(stream)
                    with self.assertRaisesRegex(ValueError, 'wave only'):
                        load_controller_config(network, family, path)

    def test_measurement_and_controller_wce_order_without_wait(self):
        for network in ('grid', 'monaco'):
            for family in ('ma2c', 'dql', 'ppo'):
                # No wait accessors or wait_state: any attempted read fails.
                lane = SimpleNamespace(getLastStepVehicleNumber=lambda name: {'a': 5, 'b': 10}[name])
                nodes = {name: SimpleNamespace(ilds_in=[name], num_state=1,
                         num_fingerprint=1, neighbor=[other], fingerprint=np.array([.4]))
                         for name, other in [('a', 'b'), ('b', 'a')]}
                env = SimpleNamespace(name='large_grid' if network == 'grid' else 'real_net',
                    network=network, agent=family, state_names=['wave'], node_names=['a', 'b'],
                    nodes=nodes, sim=SimpleNamespace(lane=lane, lanearea=lane),
                    norms={'wave': 5.}, clips={'wave': 2.}, record_stats=False,
                    coop_gamma=.5, wce_nodes=['a', 'b'], feature_width=2)
                env._reset_state = lambda: None
                env._norm_clip_state = TrafficSimulator._norm_clip_state
                env._measure_state_step = lambda: TrafficSimulator._measure_state_step(env)
                TrafficSimulator._init_state_space(env)
                self.assertEqual(env.n_w_ls, [0, 0])
                obs = TrafficSimulator._get_state(env)
                expected = [1., 1., .4] if family == 'ma2c' else [1., 2.]
                np.testing.assert_allclose(obs[0], expected)
                self.assertEqual(len(obs[0]), env.n_s_ls[0])
                np.testing.assert_array_equal(RevisedMixin.wce_observation(env), [1., 0., 2., 0.])

    def test_controllers_reject_wait(self):
        for family in ('ma2c', 'dql', 'ppo'):
            env = FakeEnv(family)
            env.n_w_ls = [1, 1]
            with self.assertRaisesRegex(ValueError, 'wave-only'):
                Controller(env, family, 9001)

    def test_cnn_gcn_wce_update_and_checkpoint(self):
        for network, count, width in [('grid', 25, 6), ('monaco', 3, 4)]:
            obs = np.linspace(0., 1., count * width).astype(np.float32)
            env = SimpleNamespace(network=network, wce_nodes=list(range(count)),
                feature_width=width, asset_hashes={'network': 'fixture'},
                adjacency=lambda: np.eye(count, dtype=np.float32),
                wce_observation=lambda: obs)
            model = WCE(env, 9001)
            try:
                self.assertEqual(model.policy.ob_fw.shape.as_list(), [None, count * width])
                if network == 'grid':
                    kernels = [v for v in model.trainable if 'conv2d/kernel' in v.name]
                    self.assertEqual(kernels[0].shape.as_list()[2], 6)
                for _ in range(11):
                    weights, logits, value = model.act(obs)
                    self.assertAlmostEqual(float(weights.sum()), 1.)
                    model.observe(obs, logits, value, 10.)
                self.assertEqual(model.updates, 1)
                self.assertTrue(all(np.isfinite(v) for v in model.diagnostics[0].values()))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'checkpoint'
                    save_checkpoint(path, {'wce': model}, {})
                    restored = WCE(env, 3)
                    try:
                        load_checkpoint(path, {'wce': restored})
                        for actual, expected in zip(restored.act(obs), model.act(obs)):
                            np.testing.assert_array_equal(actual, expected)
                    finally:
                        restored.close()
            finally:
                model.close()


if __name__ == '__main__':
    unittest.main()
