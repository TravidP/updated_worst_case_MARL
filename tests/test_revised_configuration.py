"""Contract tests for the authoritative revised configuration layer."""
import configparser
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.configuration import (controller_config_path, load_controller_config,
                                       load_wce_config)


class RevisedConfiguration(unittest.TestCase):
    def test_all_eight_mappings_and_wce_configs_validate(self):
        seen = set()
        for network in ('grid', 'monaco'):
            for family in ('ia2c', 'ma2c', 'iqll', 'ppo'):
                parser, path = load_controller_config(network, family, seed=9002)
                self.assertEqual(path, controller_config_path(network, family).resolve())
                self.assertEqual(parser['ENV_CONFIG']['agent'], family)
                self.assertNotIn('TRAIN_CONFIG', parser)
                self.assertNotIn('seed', dict(configparser.ConfigParser().defaults()))
                seen.add(path)
            load_wce_config(network)
        self.assertEqual(len(seen), 8)

    def test_unknown_typo_and_wrong_identity_fail(self):
        source = controller_config_path('grid', 'ia2c').read_text()
        cases = (
            source.replace('gamma = 0.99', 'gammma = 0.99'),
            source.replace('agent = ia2c', 'agent = ma2c'),
            source.replace('reward_norm = 3000.0', 'reward_norm = 0'),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, text in enumerate(cases):
                path = Path(directory) / ('bad_{}.ini'.format(index))
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_controller_config('grid', 'ia2c', path, 9002)

    def test_reward_transform_and_schedules_use_ini(self):
        from agents.controller import Controller, scheduled
        controller = Controller.__new__(Controller)
        controller.reward_norm = 100.
        controller.reward_clip = 2.
        learner, clipped = controller.transform_rewards([-300., -50., 50.])
        np.testing.assert_allclose(learner, [-2., -.5, .5])
        np.testing.assert_array_equal(clipped, [True, False, False])

        cfg = configparser.ConfigParser()
        cfg.read_dict({'X': {
            'lr_init': '1', 'lr_decay': 'linear', 'lr_min': '.2',
            'lr_decay_steps': '100', 'entropy_coef_init': '.1',
            'entropy_decay': 'constant',
        }})
        section = cfg['X']
        self.assertAlmostEqual(scheduled(section, 'lr', 50), .6)
        self.assertAlmostEqual(scheduled(section, 'lr', 1000), .2)
        self.assertAlmostEqual(scheduled(section, 'entropy', 50), .1)

    def test_algorithm_specific_values_are_present(self):
        ppo = load_controller_config('grid', 'ppo')[0]['MODEL_CONFIG']
        self.assertGreater(ppo.getint('ppo_n_epoch'), 0)
        self.assertGreater(ppo.getfloat('ppo_clip_ratio'), 0)
        self.assertIn('ppo_adv_norm', ppo)
        iql = load_controller_config('grid', 'iqll')[0]['MODEL_CONFIG']
        for key in ('buffer_size', 'update_interval', 'updates_per_trigger',
                    'epsilon_init', 'epsilon_min', 'epsilon_decay_steps'):
            self.assertIn(key, iql)

    def test_linear_entropy_schedule_schema(self):
        text = controller_config_path('grid', 'ia2c').read_text().replace(
            'entropy_decay = constant',
            'entropy_decay = linear\nentropy_coef_min = 0.001\nentropy_decay_steps = 1000')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'linear.ini'
            path.write_text(text)
            parser, _ = load_controller_config('grid', 'ia2c', path, 9002)
            self.assertEqual(parser['MODEL_CONFIG']['entropy_decay'], 'linear')


if __name__ == '__main__':
    unittest.main()
