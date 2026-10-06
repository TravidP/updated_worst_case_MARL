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
    policy.entropy_coef = tf.placeholder_with_default(
        tf.constant(config.getfloat('entropy_coef_init'), dtype=tf.float32), [])
    log_all = tf.log(tf.clip_by_value(policy.pi, 1e-10, 1.))
    logp = tf.reduce_sum(log_all * tf.one_hot(policy.A, policy.n_a), axis=1)
    mean = lambda x: tf.reduce_sum(x * policy.mask) / tf.reduce_sum(policy.mask)
    if ppo:
        ratio = tf.exp(logp - policy.old_logp)
        clip_ratio = config.getfloat('ppo_clip_ratio')
        clipped = tf.clip_by_value(ratio, 1. - clip_ratio, 1. + clip_ratio)
        actor = -mean(tf.minimum(ratio * policy.ADV, clipped * policy.ADV))
    else:
        actor = -mean(logp * policy.ADV)
    entropy = -tf.reduce_sum(policy.pi * log_all, axis=1)
    policy.loss = (actor + .5 * config.getfloat('value_coef') *
                   mean(tf.square(policy.R - policy.v)) -
                   policy.entropy_coef * mean(entropy))
    variables = tf.trainable_variables(scope=policy.name)
    raw_grads = tf.gradients(policy.loss, variables)
    norm = tf.global_norm(raw_grads)
    policy.metrics = dict(actor_loss=actor, value_loss=.5 * config.getfloat('value_coef') * mean(tf.square(policy.R-policy.v)),
        entropy=mean(entropy), predicted_value=mean(policy.v), return_target=mean(policy.R), advantage=mean(policy.ADV),
        actor_grad_norm=tf.global_norm([g for g,v in zip(raw_grads, variables) if '/pi' in v.name]),
        critic_grad_norm=tf.global_norm([g for g,v in zip(raw_grads, variables) if '/v' in v.name]),
        grad_norm=norm, clipping_factor=tf.minimum(1., config.getfloat('max_grad_norm')/tf.maximum(norm, 1e-12)), loss=policy.loss)
    if ppo:
        policy.metrics['clip_fraction'] = mean(
            tf.cast(tf.abs(ratio - 1.) > clip_ratio, tf.float32))
    grads, _ = tf.clip_by_global_norm(raw_grads, config.getfloat('max_grad_norm'))
    optimizer = (tf.train.AdamOptimizer(policy.lr, epsilon=config.getfloat('adam_epsilon')) if ppo
                 else tf.train.RMSPropOptimizer(policy.lr, decay=config.getfloat('rmsp_alpha'),
                                                epsilon=config.getfloat('rmsp_epsilon')))
    with tf.control_dependencies(list(policy.metrics.values())):
        policy._train = optimizer.apply_gradients(zip(grads, variables))


def scheduled(config, prefix, step):
    """Return an INI-controlled constant or linear scheduler value."""
    # Entropy uses entropy_coef_{init,min} but entropy_{decay,decay_steps}.
    value_prefix = prefix + '_coef' if prefix == 'entropy' else prefix
    initial = float(config[value_prefix + '_init'])
    if config[prefix + '_decay'] == 'constant':
        return initial
    minimum = float(config[value_prefix + '_min'])
    duration = max(1, int(config[prefix + '_decay_steps']))
    progress = min(max(float(step) / duration, 0.), 1.)
    return initial + progress * (minimum - initial)


class Controller:
    def __init__(self, env, family, seed):
        self.family = family
        self.config = env.config['MODEL_CONFIG']
        self.rng = Streams(seed)
        self.batch = self.config.getint('batch_size')
        self.gamma = self.config.getfloat('gamma')
        self.reward_norm = self.config.getfloat('reward_norm')
        self.reward_clip = self.config.getfloat('reward_clip')
        self.replay_capacity = self.config.getint('buffer_size') if family == 'iqll' else 0
        self.update_interval = self.config.getint('update_interval') if family == 'iqll' else 0
        self.updates_per_trigger = self.config.getint('updates_per_trigger') if family == 'iqll' else 0
        self.learning_steps = 0
        self.scheduler_steps = 0
        self.backward_calls = 0
        self.minibatch_updates = 0  # per agent; multiply by agent count for network total
        self.pending = []
        self.diagnostics = []
        self.replay = []
        self.replay_cursor = 0
        self.previous_done = True
        self.batch_lengths = []
        self.graph = tf.Graph()
        self.policies = []
        self.signature = {'kind': 'controller', 'network': env.network, 'family': family,
                          'n_s': list(map(int, env.n_s_ls)), 'n_a': list(map(int, env.n_a_ls)),
                          'n_w': list(map(int, env.n_w_ls)), 'n_f': list(map(int, env.n_f_ls)),
                          'config': {section: dict(env.config[section])
                                     for section in env.config.sections()},
                          'assets': env.asset_hashes,
                          'lanes': env.metric.lanes, 'reward': 'learner_boundary_scaled_v3'}
        with self.graph.as_default(), initialization_stream(self.rng['initialization']):
            tf.set_random_seed(int(seed))
            for i, (ns, na, nw, nf) in enumerate(zip(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls)):
                name = '{}a'.format(i)
                if family == 'iqll':
                    p = LRQPolicy(ns, na, self.batch, name=name)
                    p.prepare_loss(self.config.getfloat('max_grad_norm'), self.gamma,
                                   self.config.getfloat('adam_epsilon'))
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
            session_config = tf.ConfigProto(
                intra_op_parallelism_threads=self.config.getint(
                    'tf_intra_op_threads', fallback=1),
                inter_op_parallelism_threads=self.config.getint(
                    'tf_inter_op_threads', fallback=1),
                device_count={'GPU': 0})
            timeout = self.config.getint('tf_operation_timeout_ms', fallback=0)
            if timeout > 0:
                session_config.operation_timeout_in_ms = timeout
            self.sess = tf.Session(config=session_config)
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
        if self.family != 'iqll':
            return 0.
        return scheduled(self.config, 'epsilon', self.scheduler_steps)

    @property
    def lr(self):
        return scheduled(self.config, 'lr', self.scheduler_steps)

    @property
    def entropy_coef(self):
        if self.family == 'iqll':
            return 0.
        return scheduled(self.config, 'entropy', self.scheduler_steps)

    def transform_rewards(self, raw_rewards):
        raw = np.asarray(raw_rewards, dtype=np.float32)
        scaled = raw / self.reward_norm
        if self.reward_clip > 0:
            learner = np.clip(scaled, -self.reward_clip, self.reward_clip)
        else:
            learner = scaled
        clipped = np.not_equal(learner, scaled)
        return learner, clipped

    def act(self, obs, env, learning):
        actions, values, probabilities = [], [], []
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
        if self.family == 'iqll' and learning:
            self.scheduler_steps += 1
        return actions, values, probabilities

    def observe(self, obs, decision, rewards, next_obs, done, learning):
        actions, values, probabilities = decision
        learner_rewards, clipped = self.transform_rewards(rewards)
        if learning:
            transition = dict(obs=copy.deepcopy(obs), actions=list(actions),
                              raw_rewards=np.asarray(rewards, dtype=np.float32),
                              rewards=learner_rewards.copy(),
                              next_obs=copy.deepcopy(next_obs), done=bool(done), previous_done=self.previous_done,
                              values=np.array(values), probabilities=copy.deepcopy(probabilities))
            self.learning_steps += 1
            if self.family == 'iqll':
                if len(self.replay) < self.replay_capacity:
                    self.replay.append(transition)
                else:
                    self.replay[self.replay_cursor % self.replay_capacity] = transition
                self.replay_cursor += 1
                if (self.learning_steps % self.update_interval == 0 and
                        len(self.replay) >= self.batch):
                    self._update_iql()
            else:
                self.pending.append(transition)
                if len(self.pending) == self.batch or done:
                    self.flush(next_obs, done)
        self.previous_done = bool(done)
        return learner_rewards, clipped

    def _update_iql(self):
        for i, p in enumerate(self.policies):
            for _ in range(self.updates_per_trigger):
                indices = self.rng['replay'].choice(len(self.replay), self.batch, replace=False)
                batch = [self.replay[j] for j in indices]
                feed = {p.S: np.array([t['obs'][i] for t in batch]), p.A: np.array([t['actions'][i] for t in batch]),
                        p.S1: np.array([t['next_obs'][i] for t in batch]), p.DONE: np.array([t['done'] for t in batch]),
                        p.R: np.array([t['rewards'][i] for t in batch]), p.lr: self.lr}
                metrics, _train = self.sess.run([p.metrics, p._train], feed)
                self._record_metrics(i, metrics, replay_occupancy=len(self.replay),
                                     epsilon=self.epsilon, learning_rate=self.lr)
        self.backward_calls += 1
        self.minibatch_updates += self.updates_per_trigger
        self.batch_lengths.append(self.batch)

    def flush(self, next_obs, terminal=False):
        if not self.pending:
            return
        count = len(self.pending)
        for i, p in enumerate(self.policies):
            bootstrap = 0. if terminal else float(p.forward(self.sess, next_obs[i], False, 'v'))
            returns = []
            for t in reversed(self.pending):
                bootstrap = t['rewards'][i] + self.gamma * bootstrap * (not t['done'])
                returns.append(bootstrap)
            returns = np.array(returns[::-1], dtype=np.float32)
            advantages = returns - np.array([t['values'][i] for t in self.pending])
            normalize_advantage = (
                self.config.getboolean('ppo_adv_norm') if self.family == 'ppo'
                else self.config.getboolean('adv_norm'))
            if normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            pad = lambda a: np.pad(np.asarray(a), [(0, self.batch - count)] + [(0, 0)] * (np.asarray(a).ndim - 1), 'constant')
            feed = {p.ob_bw: pad([t['obs'][i] for t in self.pending]),
                    p.done_bw: pad([t['previous_done'] for t in self.pending]), p.states: p.states_bw,
                    p.A: pad([t['actions'][i] for t in self.pending]), p.R: pad(returns),
                    p.ADV: pad(advantages), p.mask: pad(np.ones(count)), p.lr: self.lr,
                    p.entropy_coef: self.entropy_coef,
                    p.old_logp: pad([np.log(max(t['probabilities'][i][t['actions'][i]], 1e-10)) for t in self.pending])}
            # PPO epochs all use the same start-of-rollout recurrent state.
            n_epoch = self.config.getint('ppo_n_epoch') if self.family == 'ppo' else 1
            for _ in range(n_epoch):
                metrics, _train = self.sess.run([p.metrics, p._train], feed)
                self._record_metrics(i, metrics, real_batch_size=count,
                                     learning_rate=self.lr,
                                     entropy_coef=self.entropy_coef)
            p.states_bw = p.states_fw.copy()
        self.scheduler_steps += count
        self.backward_calls += 1
        self.minibatch_updates += (
            self.config.getint('ppo_n_epoch') if self.family == 'ppo' else 1)
        self.batch_lengths.append(count)
        self.pending = []

    def _record_metrics(self, agent, metrics, **extra):
        values = {k: float(v) for k,v in metrics.items()}
        if not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError('Nonfinite controller diagnostics')
        self.diagnostics.append(dict(values, agent=agent, learning_steps=self.learning_steps, **extra))

    def assert_finite(self):
        if not all(np.all(np.isfinite(v)) for v in self.sess.run(self.variables)):
            raise FloatingPointError('Nonfinite controller variables')

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
