"""Temporal context, replay boundaries, and recurrent Q-learning regressions."""
import unittest
import numpy as np
from tf_compat import tf
from agents.dql import Replay
from agents.controller import Controller
from agents.recurrent import compact_lstm, window_lstm
from test_learning import FakeEnv


class RecurrentDQLTests(unittest.TestCase):
    def test_replay_terminal_padding_and_wraparound(self):
        replay = Replay(5, [1])
        for step in range(8):
            replay.add([[step]], [0], [step], [[step + 1]], step == 4)
        # Retained chronological values: 3, 4 (terminal), 5, 6, 7.
        batch = replay.sequences(0, [4, 0, 2, 3], 4)
        np.testing.assert_array_equal(batch['obs'][:, :, 0],
                                      [[0, 0, 3, 4], [0, 0, 0, 5], [0, 5, 6, 7], [0, 0, 0, 3]])
        np.testing.assert_array_equal(batch['mask'],
                                      [[0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 1, 1], [0, 0, 0, 1]])
        np.testing.assert_array_equal(batch['next_obs'][:, :, 0],
                                      [[0, 3, 4, 5], [0, 0, 5, 6], [5, 6, 7, 8], [0, 0, 3, 4]])
        np.testing.assert_array_equal(batch['terminal'], [True, False, False, False])

    def test_batched_gates_match_existing_lstm_and_ignore_padding(self):
        with tf.Graph().as_default():
            xs = tf.placeholder(tf.float32, [None, 4, 3])
            mask = tf.placeholder(tf.float32, [None, 4])
            result = window_lstm(xs, mask, 2, 'shared')
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                expected, _ = compact_lstm(xs[0, 2:], tf.zeros([2]), tf.zeros([4]), 'shared')
            gradient = tf.gradients(tf.reduce_sum(result), xs)[0]
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                values = np.arange(12, dtype=np.float32).reshape(1, 4, 3) / 12
                actual, legacy, grad = sess.run([result, expected, gradient],
                    {xs: values, mask: [[0, 0, 1, 1]]})
                np.testing.assert_allclose(actual[0], legacy[-1], rtol=1e-6, atol=1e-7)
                np.testing.assert_array_equal(grad[:, :2], 0.)
                self.assertGreater(np.linalg.norm(grad[:, 2]), 0.)

    def test_act_replay_context_reset_and_terminal_target(self):
        env = FakeEnv('dql')
        env.config['MODEL_CONFIG']['sequence_length'] = '3'
        model = Controller(env, 'dql', 9001)
        try:
            for step in range(5):
                obs = [np.full(4, step + 1, np.float32)] * 2
                decision = model.act(obs, env, True)
                model.observe(obs, decision, [-2, -3], obs, step == 4, True)
            p = model.policies[0]
            batch = model.replay.sequences(0, [4], 3)
            np.testing.assert_array_equal(batch['obs'][0], model.history[0])
            feed = {p[k]: v for k, v in batch.items()}
            target = model.sess.run(p['target'], feed)
            np.testing.assert_array_equal(target, batch['reward'])
            self.assertTrue(any('/lstm/wh' in v.name for v in model.online))
            model.assert_finite()
            # A terminal observation clears temporal context before the next action.
            model.act(obs, env, False)
            self.assertEqual(len(model.history[0]), 1)
            model.reset()
            self.assertEqual(model.history, [[], []])
            model.act(obs, env, False)
            self.assertEqual(len(model.history[0]), 1)
        finally:
            model.close()


if __name__ == '__main__':
    unittest.main()
