"""CB-WCE policy and recoverable training state."""
import copy
import numpy as np
from tf_compat import tf
from agents.policies import GaussianCNNACPolicy, GaussianGCNACPolicy
from agents.controller import initialization_stream
from experiments.core import Streams

class WCE:
    def __init__(self, env, seed):
        self.rng = Streams(seed)
        self.pending = []
        self.updates = 0
        self.macro_steps = 0
        self.graph = tf.Graph()
        size = len(env.wce_observation())
        self.signature = {'kind': 'wce', 'network': env.network, 'input_size': size,
                          'nodes': env.wce_nodes, 'features': env.feature_width,
                          'adjacency': env.adjacency().tolist(), 'assets': env.asset_hashes,
                          'architecture_version': 1, 'action_size': 11}
        with self.graph.as_default(), initialization_stream(self.rng['initialization']):
            tf.set_random_seed(int(seed))
            if env.network == 'grid':
                self.policy = GaussianCNNACPolicy(size, 11, 11)
            else:
                self.policy = GaussianGCNACPolicy(size, 11, 11, env.adjacency(), len(env.wce_nodes), env.feature_width)
            self.policy.prepare_loss(.5, 40., .99, 1e-5)
            self.sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,
                                                        inter_op_parallelism_threads=1, device_count={'GPU': 0}))
            self.variables = tf.global_variables()
            self.trainable = tf.trainable_variables()
            self.saver = tf.train.Saver(var_list=self.variables, max_to_keep=0)
            self.sess.run(tf.global_variables_initializer())

    def act(self, obs):
        p = self.policy
        mu, std, value = self.sess.run([p.mu, p.std, p.v], {p.ob_fw: [obs], p.done_fw: [False]})
        # Never execute legacy stateful TensorFlow sampling ops.
        logits = np.asarray(mu).reshape(-1) + self.rng['wce'].normal(size=11) * std
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        return weights, logits, float(value)

    def observe(self, obs, logits, value, reward):
        self.pending.append((np.array(obs), np.array(logits), float(value), float(reward)))
        self.macro_steps += 1
        if len(self.pending) == 11:
            returns = np.cumsum([t[3] for t in self.pending][::-1])[::-1]
            advantages = returns - np.array([t[2] for t in self.pending])
            self.policy.backward(self.sess, np.array([t[0] for t in self.pending]),
                np.array([t[1] for t in self.pending]), np.zeros(11), returns, advantages, .0005, .01)
            self.pending = []
            self.updates += 1

    def state(self):
        return {'pending': copy.deepcopy(self.pending), 'updates': self.updates,
                'macro_steps': self.macro_steps, 'rng': self.rng.state()}

    def restore_state(self, state):
        self.pending = copy.deepcopy(state['pending'])
        self.updates, self.macro_steps = state['updates'], state['macro_steps']
        self.rng.restore(state['rng'])

    def close(self):
        self.sess.close()
