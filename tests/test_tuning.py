"""Fast tests for the guarded tuning campaign gates."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.offline_screen import iql_targets
from experiments.tuning import candidate_scales, final_gate, paired_gate, round_125


class TuningCampaign(unittest.TestCase):
    def test_iql_targets_remain_float32(self):
        targets = iql_targets(
            np.asarray([-1., -2.], dtype=np.float32),
            np.asarray([[1., 3.], [4., 2.]], dtype=np.float32),
            .99, [False, True])
        self.assertEqual(targets.dtype, np.float32)
        np.testing.assert_allclose(targets, [1.97, -2.], rtol=1e-6)

    def test_rounded_center_and_candidates(self):
        self.assertEqual(round_125(1388.28), 1000.)
        self.assertEqual(round_125(390.06), 500.)
        center, values = candidate_scales([10, 20, 30, 40, 50], 30)
        self.assertIn(center, values)
        self.assertIn(30., values)

    def test_pair_and_final_gates_use_traffic_not_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            monitoring = run / 'monitoring'
            for index, (queue, completed, teleports) in enumerate(
                    [(100, 80, 2), (95, 82, 2), (90, 85, 1)]):
                target = monitoring / ('round_{:06}'.format(index))
                target.mkdir(parents=True)
                (target / 'summary.json').write_text(json.dumps({
                    'status': 'complete', 'metrics': {
                        'mean_total_queue': queue, 'completed': completed,
                        'teleports': teleports}}))
            with (run / 'learner_metrics.jsonl').open('w') as stream:
                for step in range(30):
                    stream.write(json.dumps({
                        'role': 'learner', 'clipping_factor': .5,
                        'entropy': 1. - step / 100.,
                        'actor_grad_norm': 2., 'critic_grad_norm': 4.,
                    }) + '\n')
            self.assertTrue(paired_gate(run)['passed'])
            result = final_gate(run)
            self.assertTrue(result['passed'])
            self.assertLessEqual(result['monitor_queue_slope'], 0)


if __name__ == '__main__':
    unittest.main()
