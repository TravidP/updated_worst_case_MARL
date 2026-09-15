"""Display selection must not change experiment arguments or reset seeds."""
import os
import unittest
from unittest.mock import patch

from experiments.runner import parser
from envs.experiment_env import RevisedMixin, validate_visualization


class Visualization(unittest.TestCase):
    def test_cli_default_on_off_and_conflict(self):
        base=['--stage','parent','--network','grid','--controller','iqll','--output','unused']
        self.assertFalse(parser().parse_args(base).visualization)
        self.assertTrue(parser().parse_args(base+['--visualization']).visualization)
        self.assertFalse(parser().parse_args(base+['--no-visualization']).visualization)
        with self.assertRaises(SystemExit):
            parser().parse_args(base+['--visualization','--no-visualization'])

    def test_missing_display_or_binary_fail_only_when_enabled(self):
        with patch('envs.experiment_env.shutil.which',return_value=None):
            validate_visualization(False)
            with self.assertRaisesRegex(ValueError,'sumo-gui'):validate_visualization(True)
        with patch('envs.experiment_env.shutil.which',return_value='/usr/bin/sumo-gui'), \
             patch('envs.experiment_env.sys.platform','linux'), patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'DISPLAY'):validate_visualization(True)
            validate_visualization(False)
        for bad in ('false',1,None):
            with self.assertRaises(ValueError):validate_visualization(bad)

    def test_reset_propagates_display_and_preserves_evaluation_seed(self):
        from types import SimpleNamespace
        for enabled in (False,True):
            env=SimpleNamespace(visualization=enabled,init_test_seeds=lambda seeds:None)
            with patch('envs.experiment_env.TrafficSimulator.reset',return_value='observation') as reset:
                self.assertEqual(RevisedMixin.reset_episode(env,61002,True,3600),'observation')
                reset.assert_called_once_with(env,gui=enabled,test_ind=0)
            self.assertEqual(env.seed,61002)
            self.assertFalse(env.train_mode)
            self.assertEqual(env.T,720)

    def test_dashboard_boolean_and_generated_command(self):
        from experiments.dashboard import preflight
        spec=dict(stage='parent',network='grid',controller='iqll',seed=9001,pilot=True,steps=160)
        with patch('envs.experiment_env.shutil.which',return_value='/usr/bin/sumo-gui'), \
             patch.dict(os.environ,{'DISPLAY':':99'}):
            off=preflight(spec)['argv']
            on=preflight(dict(spec,visualization=True))['argv']
            self.assertIn('--no-visualization',off)
            self.assertIn('--visualization',on)
            self.assertEqual(off[3:-2],[('--no-visualization' if x=='--visualization' else x) for x in on[3:-2]])
            with self.assertRaisesRegex(ValueError,'boolean'):preflight(dict(spec,visualization='false'))


if __name__=='__main__':unittest.main(verbosity=2)
