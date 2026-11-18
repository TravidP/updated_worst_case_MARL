"""Verify the actor feed, critic targets and constant-advantage edge case."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from agents.wce import WCE
from experiments.configuration import load_wce_config


class WCEAdvantageTests(unittest.TestCase):
    def test_default_and_boolean_validation(self):
        for network in ('grid', 'monaco'):
            config, _ = load_wce_config(network)
            self.assertTrue(config['WCE_CONFIG'].getboolean('adv_norm'))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'wce.ini'
                for value in ('false', 'invalid'):
                    config['WCE_CONFIG']['adv_norm'] = value
                    with path.open('w') as stream:config.write(stream)
                    if value == 'false':
                        self.assertFalse(load_wce_config(network,path)[0]['WCE_CONFIG'].getboolean('adv_norm'))
                    else:
                        with self.assertRaises(ValueError):load_wce_config(network,path)

    def test_actor_feed_and_unchanged_critic_targets(self):
        for enabled in (True, False):
            for constant in (True, False):
                model = WCE.__new__(WCE)
                model.training=True;model.gamma=1.;model.reward_norm=1.;model.reward_clip=0.
                model.pending=[];model.macro_steps=0;model.updates=0;model.diagnostics=[];model.variables=[]
                model.config=load_wce_config('grid')[0]['WCE_CONFIG']
                model.config['adv_norm']=str(enabled)
                names=['ob_fw','done_fw','A','R','ADV','lr','entropy_coef','metrics','_train']
                model.policy=SimpleNamespace(**{name:name for name in names})
                captured={}
                def run(fetches, feed=None):
                    if feed is None:return []
                    captured.update(feed)
                    return {'loss':0.},None
                model.sess=SimpleNamespace(run=run)
                # Rewards=1 produce returns 11..1. Subtract values for constant A=2.
                expected_returns=np.arange(11,0,-1,dtype=np.float32)
                for i in range(11):
                    model.observe([0.],[0.],expected_returns[i]-2 if constant else 0.,1.)
                raw=np.full(11,2.) if constant else expected_returns
                expected=(raw-raw.mean())/(raw.std()+1e-8) if enabled else raw
                np.testing.assert_allclose(captured['ADV'],expected,atol=1e-7)
                np.testing.assert_array_equal(captured['R'],expected_returns)
                self.assertTrue(np.isfinite(captured['ADV']).all())
                self.assertEqual(model.updates,1)
                self.assertEqual(model.pending,[])
                self.assertAlmostEqual(model.diagnostics[0]['raw_advantage_mean'],float(raw.mean()))
                if enabled:self.assertAlmostEqual(model.diagnostics[0]['advantage_mean'],0.,places=6)

if __name__=='__main__':unittest.main()
