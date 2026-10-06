"""CB-WCE policy and recoverable training state."""
import copy
import numpy as np
from tf_compat import tf
from agents.policies import GaussianCNNACPolicy, GaussianGCNACPolicy
from agents.controller import initialization_stream, scheduled
from experiments.core import Streams

class WCE:
    def __init__(self, env, seed):
        if hasattr(env, 'wce_config'):
            self.config = env.wce_config['WCE_CONFIG']
        else:
            from experiments.configuration import load_wce_config
            self.config = load_wce_config(env.network)[0]['WCE_CONFIG']
        self.gamma = self.config.getfloat('gamma')
        self.reward_norm = self.config.getfloat('reward_norm')
        self.reward_clip = self.config.getfloat('reward_clip')
        self.rng = Streams(seed)
        self.pending = []
        self.diagnostics = []
        self.updates = 0
        self.macro_steps = 0
        self.graph = tf.Graph()
        size = len(env.wce_observation())
        self.signature = {'kind': 'wce', 'network': env.network, 'input_size': size,
                          'nodes': env.wce_nodes, 'features': env.feature_width,
                          'adjacency': env.adjacency().tolist(), 'assets': env.asset_hashes,
                          'config': dict(self.config), 'architecture_version': 1,
                          'protocol_version': 6, 'action_size': 11,
                          'reward': 'mean_queue_learner_boundary_scaled_v3'}
        with self.graph.as_default(), initialization_stream(self.rng['initialization']):
            tf.set_random_seed(int(seed))
            if env.network == 'grid':
                self.policy = GaussianCNNACPolicy(size, 11, 11)
            else:
                self.policy = GaussianGCNACPolicy(size, 11, 11, env.adjacency(), len(env.wce_nodes), env.feature_width)
            self.policy.prepare_loss(
                self.config.getfloat('value_coef'),
                self.config.getfloat('max_grad_norm'),
                self.config.getfloat('rmsp_alpha'),
                self.config.getfloat('rmsp_epsilon'))
            session_config = tf.ConfigProto(
                intra_op_parallelism_threads=self.config.getint('tf_intra_op_threads', fallback=1),
                inter_op_parallelism_threads=self.config.getint('tf_inter_op_threads', fallback=1),
                device_count={'GPU': 0})
            timeout = self.config.getint('tf_operation_timeout_ms', fallback=0)
            if timeout > 0:
                session_config.operation_timeout_in_ms = timeout
            self.sess = tf.Session(config=session_config)
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
        scaled = float(reward) / self.reward_norm
        learner_reward = (float(np.clip(scaled, -self.reward_clip, self.reward_clip))
                          if self.reward_clip > 0 else scaled)
        clipped = learner_reward != scaled
        self.pending.append((np.array(obs), np.array(logits), float(value),
                             float(reward), learner_reward, clipped))
        self.macro_steps += 1
        if len(self.pending) == 11:
            running = 0.
            returns = []
            for transition in reversed(self.pending):
                running = transition[4] + self.gamma * running
                returns.append(running)
            returns = np.asarray(returns[::-1], dtype=np.float32)
            advantages = returns - np.array([t[2] for t in self.pending])
            p = self.policy
            feed = {p.ob_fw: np.array([t[0] for t in self.pending]), p.done_fw: np.zeros(11),
                    p.A: np.array([t[1] for t in self.pending]), p.R: returns, p.ADV: advantages,
                    p.lr: scheduled(self.config, 'lr', self.updates),
                    p.entropy_coef: scheduled(self.config, 'entropy', self.updates)}
            values, _train = self.sess.run([p.metrics, p._train], feed)
            values = {key: float(value) for key, value in values.items()}
            values.update(
                raw_reward_mean=float(np.mean([t[3] for t in self.pending])),
                learner_reward_mean=float(np.mean([t[4] for t in self.pending])),
                reward_clip_fraction=float(np.mean([t[5] for t in self.pending])),
                return_target=float(np.mean(returns)),
                learning_rate=scheduled(self.config, 'lr', self.updates),
                entropy_coef=scheduled(self.config, 'entropy', self.updates),
                macro_steps=self.macro_steps)
            if not all(np.isfinite(v) for v in values.values()):
                raise FloatingPointError('Nonfinite WCE diagnostics')
            if not all(np.all(np.isfinite(v)) for v in self.sess.run(self.variables)):
                raise FloatingPointError('Nonfinite WCE variables')
            self.diagnostics.append(values)
            self.pending = []
            self.updates += 1
        return learner_reward, clipped

    def state(self):
        return {'pending': copy.deepcopy(self.pending), 'updates': self.updates,
                'macro_steps': self.macro_steps, 'rng': self.rng.state()}

    def restore_state(self, state):
        self.pending = copy.deepcopy(state['pending'])
        self.updates, self.macro_steps = state['updates'], state['macro_steps']
        self.rng.restore(state['rng'])

    def close(self):
        self.sess.close()
