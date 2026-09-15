"""Shared learner adapters with explicit RNG, masked batches and complete state."""
import copy
from contextlib import contextmanager
import numpy as np
from tf_compat import tf
from agents.policies import (LstmACPolicy, FPLstmACPolicy, PPOLstmACPolicy, LRQPolicy,
                             GaussianCNNACPolicy, GaussianGCNACPolicy)
from experiments.core import Streams
from agents.recurrent import CompactPolicy, CompactFingerprintPolicy


@contextmanager
def initialization_stream(rng):
    # Legacy orthogonal initializers use global NumPy. Scope only graph construction.
    previous = np.random.get_state()
    np.random.set_state(rng.get_state())
    try:
        yield
    finally:
        rng.set_state(np.random.get_state())
        np.random.set_state(previous)


def masked_loss(policy, config, ppo=False):
    n = policy.n_step
    policy.A = tf.placeholder(tf.int32, [n])
    policy.ADV = tf.placeholder(tf.float32, [n])
    policy.R = tf.placeholder(tf.float32, [n])
    policy.mask = tf.placeholder(tf.float32, [n])
    policy.old_logp = tf.placeholder(tf.float32, [n])
    policy.lr = tf.placeholder(tf.float32, [])
    log_all = tf.log(tf.clip_by_value(policy.pi, 1e-10, 1.))
    logp = tf.reduce_sum(log_all * tf.one_hot(policy.A, policy.n_a), axis=1)
    mean = lambda x: tf.reduce_sum(x * policy.mask) / tf.reduce_sum(policy.mask)
    if ppo:
        ratio = tf.exp(logp - policy.old_logp)
        clipped = tf.clip_by_value(ratio, .8, 1.2)
        actor = -mean(tf.minimum(ratio * policy.ADV, clipped * policy.ADV))
    else:
        actor = -mean(logp * policy.ADV)
    entropy = -tf.reduce_sum(policy.pi * log_all, axis=1)
    policy.loss = actor + .5 * config.getfloat('value_coef') * mean(tf.square(policy.R - policy.v)) - config.getfloat('entropy_coef_init') * mean(entropy)
    variables = tf.trainable_variables(scope=policy.name)
    grads, _ = tf.clip_by_global_norm(tf.gradients(policy.loss, variables), config.getfloat('max_grad_norm'))
    optimizer = (tf.train.AdamOptimizer(policy.lr, epsilon=config.getfloat('rmsp_epsilon')) if ppo
                 else tf.train.RMSPropOptimizer(policy.lr, decay=config.getfloat('rmsp_alpha'),
                                                epsilon=config.getfloat('rmsp_epsilon')))
    policy._train = optimizer.apply_gradients(zip(grads, variables))


class Controller:
    def __init__(self, env, family, seed):
        self.family = family
        self.config = env.config['MODEL_CONFIG']
        self.rng = Streams(seed)
        self.batch = self.config.getint('batch_size')
        self.lr = self.config.getfloat('lr_init')
        self.learning_steps = 0
        self.scheduler_steps = 0
        self.backward_calls = 0
        self.minibatch_updates = 0  # per agent; multiply by agent count for network total
        self.pending = []
        self.replay = []
        self.replay_cursor = 0
        self.previous_done = True
        self.batch_lengths = []
        self.graph = tf.Graph()
        self.policies = []
        self.signature = {'kind': 'controller', 'network': env.network, 'family': family,
                          'n_s': list(map(int, env.n_s_ls)), 'n_a': list(map(int, env.n_a_ls)),
                          'n_w': list(map(int, env.n_w_ls)), 'n_f': list(map(int, env.n_f_ls)),
                          'config': dict(self.config), 'assets': env.asset_hashes,
                          'lanes': env.metric.lanes, 'reward': 'queue_1hz_once_100_v1'}
        with self.graph.as_default(), initialization_stream(self.rng['initialization']):
            tf.set_random_seed(int(seed))
            for i, (ns, na, nw, nf) in enumerate(zip(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls)):
                name = '{}a'.format(i)
                if family == 'iqll':
                    p = LRQPolicy(ns, na, self.batch, name=name)
                    p.prepare_loss(self.config.getfloat('max_grad_norm'), .99)
                else:
                    common = dict(n_fc_wave=self.config.getint('num_fw'),
                                  n_fc_wait=self.config.getint('num_ft'),
                                  n_lstm=self.config.getint('num_lstm'), name=name)
                    if family == 'ma2c':
                        p = CompactFingerprintPolicy(ns - nw - nf, na, nw, nf, self.batch,
                                          n_fc_fp=self.config.getint('num_fp', fallback=64), **common)
                    else:
                        p = CompactPolicy(ns - nw, na, nw, self.batch, **common)
                    masked_loss(p, self.config, family == 'ppo')
                self.policies.append(p)
            self.sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,
                                                         inter_op_parallelism_threads=1, device_count={'GPU': 0}))
            self.variables = tf.global_variables()
            self.trainable = tf.trainable_variables()
            self.saver = tf.train.Saver(var_list=self.variables, max_to_keep=0)
            self.sess.run(tf.global_variables_initializer())

    def reset(self):
        if self.pending:
            raise ValueError('Flush on-policy batch before episode reset')
        for p in self.policies:
            if hasattr(p, '_reset'):
                p._reset()
        self.previous_done = True

    @property
    def epsilon(self):
        return max(.01, 1. - self.scheduler_steps / 500000.)

    def act(self, obs, env, learning):
        actions, values, probabilities = [], [], []
        if self.family == 'iqll' and learning:
            self.scheduler_steps += 1
        feeds = {}
        for p, ob in zip(self.policies, obs):
            if self.family == 'iqll':
                feeds[p.S] = [ob]
            else:
                feeds.update({p.ob_fw: [ob], p.done_fw: [self.previous_done], p.states: p.states_fw})
        outputs = ([p.qvalues for p in self.policies] if self.family == 'iqll'
                   else [[p.pi_fw, p.v_fw, p.new_states] for p in self.policies])
        predictions = self.sess.run(outputs, feeds)
        for p, prediction in zip(self.policies, predictions):
            if self.family == 'iqll':
                q = prediction
                action = (self.rng['policy'].randint(p.n_a) if learning and self.rng['policy'].rand() < self.epsilon
                          else int(np.argmax(q)))
                actions.append(action)
            else:
                pi, value, p.states_fw = prediction
                actions.append(self.rng['policy'].choice(len(pi), p=pi))
                probabilities.append(pi)
                values.append(value)
        if self.family == 'ma2c':
            env.update_fingerprint(probabilities)
        return actions, values, probabilities

    def observe(self, obs, decision, rewards, next_obs, done, learning):
        actions, values, probabilities = decision
        if learning:
            transition = dict(obs=copy.deepcopy(obs), actions=list(actions), rewards=np.array(rewards),
                              next_obs=copy.deepcopy(next_obs), done=bool(done), previous_done=self.previous_done,
                              values=np.array(values), probabilities=copy.deepcopy(probabilities))
            self.learning_steps += 1
            if self.family == 'iqll':
                if len(self.replay) < 1000:
                    self.replay.append(transition)
                else:
                    self.replay[self.replay_cursor % 1000] = transition
                self.replay_cursor += 1
                if self.learning_steps % 20 == 0 and len(self.replay) >= self.batch:
                    self._update_iql()
            else:
                self.pending.append(transition)
                if len(self.pending) == self.batch or done:
                    self.flush(next_obs, done)
        self.previous_done = bool(done)

    def _update_iql(self):
        for i, p in enumerate(self.policies):
            for _ in range(10):
                indices = self.rng['replay'].choice(len(self.replay), self.batch, replace=False)
                batch = [self.replay[j] for j in indices]
                p.backward(self.sess, np.array([t['obs'][i] for t in batch]),
                    np.array([t['actions'][i] for t in batch]),
                    np.array([t['next_obs'][i] for t in batch]),
                    np.array([t['done'] for t in batch]),
                    np.array([t['rewards'][i] for t in batch]), self.lr)
        self.backward_calls += 1
        self.minibatch_updates += 10
        self.batch_lengths.append(self.batch)

    def flush(self, next_obs, terminal=False):
        if not self.pending:
            return
        count = len(self.pending)
        for i, p in enumerate(self.policies):
            bootstrap = 0. if terminal else float(p.forward(self.sess, next_obs[i], False, 'v'))
            returns = []
            for t in reversed(self.pending):
                bootstrap = t['rewards'][i] + .99 * bootstrap * (not t['done'])
                returns.append(bootstrap)
            returns = np.array(returns[::-1], dtype=np.float32)
            advantages = returns - np.array([t['values'][i] for t in self.pending])
            if self.family == 'ppo' and self.config.getboolean('ppo_adv_norm', fallback=True):
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            pad = lambda a: np.pad(np.asarray(a), [(0, self.batch - count)] + [(0, 0)] * (np.asarray(a).ndim - 1), 'constant')
            feed = {p.ob_bw: pad([t['obs'][i] for t in self.pending]),
                    p.done_bw: pad([t['previous_done'] for t in self.pending]), p.states: p.states_bw,
                    p.A: pad([t['actions'][i] for t in self.pending]), p.R: pad(returns),
                    p.ADV: pad(advantages), p.mask: pad(np.ones(count)), p.lr: self.lr,
                    p.old_logp: pad([np.log(max(t['probabilities'][i][t['actions'][i]], 1e-10)) for t in self.pending])}
            # PPO epochs all use the same start-of-rollout recurrent state.
            for _ in range(4 if self.family == 'ppo' else 1):
                loss, _ = self.sess.run([p.loss, p._train], feed)
                if not np.isfinite(loss):
                    raise FloatingPointError('Nonfinite controller loss')
            p.states_bw = p.states_fw.copy()
        self.scheduler_steps += count
        self.backward_calls += 1
        self.minibatch_updates += 4 if self.family == 'ppo' else 1
        self.batch_lengths.append(count)
        self.pending = []

    def state(self):
        names = ('learning_steps', 'scheduler_steps', 'backward_calls', 'minibatch_updates',
                 'pending', 'replay', 'replay_cursor', 'previous_done', 'batch_lengths')
        state = {name: copy.deepcopy(getattr(self, name)) for name in names}
        state['rng'] = self.rng.state()
        state['recurrent'] = [(p.states_fw.copy(), p.states_bw.copy()) if hasattr(p, 'states_fw') else None for p in self.policies]
        return state

    def restore_state(self, state):
        state = copy.deepcopy(state)
        self.rng.restore(state.pop('rng'))
        for p, recurrent in zip(self.policies, state.pop('recurrent')):
            if recurrent is not None:
                p.states_fw, p.states_bw = recurrent
        for name, value in state.items():
            setattr(self, name, value)

    def close(self):
        self.sess.close()


