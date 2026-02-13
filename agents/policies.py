import numpy as np
import tensorflow as tf
from agents.utils import *
import bisect


class ACPolicy:
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name):
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _build_out_net(self, h, out_type):
        if out_type == 'pi':
            pi = fc(h, out_type, self.n_a, act=tf.nn.softmax)
            return tf.squeeze(pi)
        else:
            v = fc(h, out_type, 1, act=lambda x: x)
            return tf.squeeze(v)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi)
        if 'v' in out_type:
            outs.append(self.v)
        return outs

    def _return_forward_outs(self, out_values):
        if len(out_values) == 1:
            return out_values[0]
        return out_values

    def prepare_loss(self, v_coef, max_grad_norm, alpha, epsilon):
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.entropy_coef = tf.placeholder(tf.float32, [])
        A_sparse = tf.one_hot(self.A, self.n_a)
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
        policy_loss = -tf.reduce_mean(tf.reduce_sum(log_pi * A_sparse, axis=1) * self.ADV)
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        self.loss = policy_loss + value_loss + entropy_loss

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                   epsilon=epsilon)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        # monitor training
        if self.name.endswith('_0a'):
            summaries = []
            # summaries.append(tf.summary.scalar('loss/%s_entropy_loss' % self.name, entropy_loss))
            summaries.append(tf.summary.scalar('loss/%s_policy_loss' % self.name, policy_loss))
            summaries.append(tf.summary.scalar('loss/%s_value_loss' % self.name, value_loss))
            summaries.append(tf.summary.scalar('loss/%s_total_loss' % self.name, self.loss))
            # summaries.append(tf.summary.scalar('train/%s_lr' % self.name, self.lr))
            # summaries.append(tf.summary.scalar('train/%s_entropy_beta' % self.name, self.entropy_coef))
            summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
            self.summary = tf.summary.merge(summaries)


class LstmACPolicy(ACPolicy):
    def __init__(self, n_s, n_a, n_w, n_step, n_fc_wave=128, n_fc_wait=32, n_lstm=64, name=None):
        super().__init__(n_a, n_s, n_step, 'lstm', name)
        self.n_lstm = n_lstm
        self.n_fc_wait = n_fc_wait
        self.n_fc_wave = n_fc_wave
        self.n_w = n_w
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s + n_w]) # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s + n_w]) # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net('forward', 'pi')
            self.v_fw, v_state = self._build_net('forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net('backward', 'pi')
            self.v, _ = self._build_net('backward', 'v')
        self._reset()

    def _build_net(self, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        if self.n_w == 0:
            h = fc(ob, out_type + '_fcw', self.n_fc_wave)
        else:
            h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
            h1 = fc(ob[:, self.n_s:], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1], 1)
        h, new_states = lstm(h, done, states, out_type + '_lstm')
        out_val = self._build_out_net(h, out_type)
        return out_val, new_states

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
        self.states_bw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        # update state only when p is called
        if 'p' in out_type:
            outs.append(self.new_states)
        out_values = sess.run(outs, {self.ob_fw:np.array([ob]),
                                     self.done_fw:np.array([done]),
                                     self.states:self.states_fw})
        if 'p' in out_type:
            self.states_fw = out_values[-1]
            out_values = out_values[:-1]
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                        {self.ob_bw: obs,
                         self.done_bw: dones,
                         self.states: self.states_bw,
                         self.A: acts,
                         self.ADV: Advs,
                         self.R: Rs,
                         self.lr: cur_lr,
                         self.entropy_coef: cur_beta})
        self.states_bw = np.copy(self.states_fw)
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi_fw)
        if 'v' in out_type:
            outs.append(self.v_fw)
        return outs


class FPLstmACPolicy(LstmACPolicy):
    def __init__(self, n_s, n_a, n_w, n_f, n_step, n_fc_wave=128, n_fc_wait=32, n_fc_fp=32, n_lstm=64, name=None):
        ACPolicy.__init__(self, n_a, n_s, n_step, 'fplstm', name)
        self.n_lstm = n_lstm
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_fc_fp = n_fc_fp
        self.n_w = n_w
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s + n_w + n_f]) # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s + n_w + n_f]) # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net('forward', 'pi')
            self.v_fw, v_state = self._build_net('forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net('backward', 'pi')
            self.v, _ = self._build_net('backward', 'v')
        self._reset()

    def _build_net(self, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
        h1 = fc(ob[:, (self.n_s + self.n_w):], out_type + '_fcf', self.n_fc_fp)
        if self.n_w == 0:
            h = tf.concat([h0, h1], 1)
        else:
            h2 = fc(ob[:, self.n_s: (self.n_s + self.n_w)], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1, h2], 1)
        h, new_states = lstm(h, done, states, out_type + '_lstm')
        out_val = self._build_out_net(h, out_type)
        return out_val, new_states


class FcACPolicy(ACPolicy):
    def __init__(self, n_s, n_a, n_w, n_step, n_fc_wave=128, n_fc_wait=32, n_lstm=64, name=None):
        super().__init__(n_a, n_s, n_step, 'fc', name)
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_fc = n_lstm
        self.n_w = n_w
        self.obs = tf.placeholder(tf.float32, [None, n_s + n_w])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi = self._build_net('pi')
            self.v = self._build_net('v')

    def _build_net(self, out_type):
        if self.n_w == 0:
            h = fc(self.obs, out_type + '_fcw', self.n_fc_wave)
        else:
            h0 = fc(self.obs[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
            h1 = fc(self.obs[:, self.n_s:], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1], 1)
        h = fc(h, out_type + '_fc', self.n_fc)
        return self._build_out_net(h, out_type)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        out_values = sess.run(outs, {self.obs: np.array([ob])})
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                        {self.obs: obs,
                         self.A: acts,
                         self.ADV: Advs,
                         self.R: Rs,
                         self.lr: cur_lr,
                         self.entropy_coef: cur_beta})
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)


class FPFcACPolicy(FcACPolicy):
    def __init__(self, n_s, n_a, n_w, n_f, n_step, n_fc_wave=128, n_fc_wait=32, n_fc_fp=32, n_lstm=64, name=None):
        ACPolicy.__init__(self, n_a, n_s, n_step, 'fpfc', name)
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_fc_fp = n_fc_fp
        self.n_fc = n_lstm
        self.n_w = n_w
        self.obs = tf.placeholder(tf.float32, [None, n_s + n_w + n_f])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi = self._build_net('pi')
            self.v = self._build_net('v')

    def _build_net(self, out_type):
        h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
        h1 = fc(ob[:, (self.n_s + self.n_w):], out_type + '_fcf', self.n_fc_fp)
        if self.n_w == 0:
            h = tf.concat([h0, h1], 1)
        else:
            h2 = fc(ob[:, self.n_s: (self.n_s + self.n_w)], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1, h2], 1)
        h = fc(h, out_type + '_fc', self.n_fc)
        return self._build_out_net(h, out_type)


class QPolicy:
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name):
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _build_fc_net(self, h, n_fc_ls):
        for i, n_fc in enumerate(n_fc_ls):
            h = fc(h, 'q_fc_%d' % i, n_fc)
        q = fc(h, 'q', self.n_a, act=lambda x: x)
        return tf.squeeze(q)

    def _build_net(self):
        raise NotImplementedError()

    def prepare_loss(self, max_grad_norm, gamma):
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.S1 = tf.placeholder(tf.float32, [self.n_step, self.n_s + self.n_w])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.DONE = tf.placeholder(tf.bool, [self.n_step])
        A_sparse = tf.one_hot(self.A, self.n_a)

        # backward
        with tf.variable_scope(self.name + '_q', reuse=True):
            q0s = self._build_net(self.S)
            q0 = tf.reduce_sum(q0s * A_sparse, axis=1)
        with tf.variable_scope(self.name + '_q', reuse=True):
            q1s = self._build_net(self.S1)
            q1 = tf.reduce_max(q1s, axis=1)
        tq = tf.stop_gradient(tf.where(self.DONE, self.R, self.R + gamma * q1))
        self.loss = tf.reduce_mean(tf.square(q0 - tq))

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        # monitor training
        if self.name.endswith('_0a'):
            summaries = []
            summaries.append(tf.summary.scalar('train/%s_loss' % self.name, self.loss))
            summaries.append(tf.summary.scalar('train/%s_q' % self.name, tf.reduce_mean(q0)))
            summaries.append(tf.summary.scalar('train/%s_tq' % self.name, tf.reduce_mean(tq)))
            summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
            self.summary = tf.summary.merge(summaries)


class DeepQPolicy(QPolicy):
    def __init__(self, n_s, n_a, n_w, n_step, n_fc0=128, n_fc=64, name=None):
        super().__init__(n_a, n_s, n_step, 'dqn', name)
        self.n_fc = n_fc
        self.n_fc0 = n_fc0
        self.n_w = n_w
        self.S = tf.placeholder(tf.float32, [None, n_s + n_w])
        with tf.variable_scope(self.name + '_q'):
            self.qvalues = self._build_net(self.S)

    def _build_net(self, S):
        if self.n_w == 0:
            h = fc(S, 'q_fcw', self.n_fc0)
        else:
            h0 = fc(S[:, :self.n_s], 'q_fcw', self.n_fc0)
            h1 = fc(S[:, self.n_s:], 'q_fct', self.n_fc0 / 4)
            h = tf.concat([h0, h1], 1)
        return self._build_fc_net(h, [self.n_fc])

    def forward(self, sess, ob):
        return sess.run(self.qvalues, {self.S: np.array([ob])})

    def backward(self, sess, obs, acts, next_obs, dones, rs, cur_lr,
                 summary_writer=None, global_step=None):
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                        {self.S: obs,
                         self.A: acts,
                         self.S1: next_obs,
                         self.DONE: dones,
                         self.R: rs,
                         self.lr: cur_lr})
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)


class LRQPolicy(DeepQPolicy):
    def __init__(self, n_s, n_a, n_step, name=None):
        QPolicy.__init__(self, n_a, n_s, n_step, 'lr', name)
        self.S = tf.placeholder(tf.float32, [None, n_s])
        self.n_w = 0
        with tf.variable_scope(self.name + '_q'):
            self.qvalues = self._build_net(self.S)

    def _build_net(self, S):
        return self._build_fc_net(S, [])



# Adapt this structure based on agents/policies.py (which likely contains LstmACPolicy)
class CNNACPolicy(ACPolicy):
    def __init__(self, n_s, n_a, n_step, name=None):
        super().__init__(n_a, n_s, n_step, 'cnn', name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step
        
        # Define single placeholder for both inference (batch=1) and training (batch=N)
        # We replace ob_fw/done_fw with a unified 'obs' to handle variable batch sizes automatically.
        self.obs = tf.placeholder(tf.float32, [None, n_s])
        
        with tf.variable_scope(self.name):
            # pi and v use separate nets (consistent with other policies in this repo)
            self.pi = self._build_net('pi')
            self.v = self._build_net('v')

    def _build_net(self, out_type):
        # 1. Reshape for CNN [Batch, 5, 5, Channels]
        grid_size = 5
        channels = int(self.n_s / (grid_size * grid_size))
        
        # Ensure input is reshaped correctly
        x = tf.reshape(self.obs, [-1, grid_size, grid_size, channels])

        # 2. Build CNN Layers (Scoped by out_type to separate Actor and Critic nets)
        # Using tf.layers for consistency with your snippet
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=3,
            padding='same',
            activation=tf.nn.relu,
            name=out_type + '_conv1'
        )
        
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            kernel_size=3,
            padding='valid',
            activation=tf.nn.relu,
            name=out_type + '_conv2'
        )
        
        flat = tf.layers.flatten(conv2)
        
        # 3. Dense Layer
        h = tf.layers.dense(flat, 128, activation=tf.nn.relu, name=out_type + '_fc')
        
        # 4. Output Head (using base class helper)
        return self._build_out_net(h, out_type)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        
        # Feed the single observation as a batch of 1
        feed_dict = {self.obs: np.array([ob])}
        
        out_values = sess.run(outs, feed_dict=feed_dict)
        
        # Squeeze logic to handle scalar returns
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        # Maps to ACPolicy training ops
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
            
        outs = sess.run(ops,
                        {self.obs: obs,
                         self.A: acts,
                         self.ADV: Advs,
                         self.R: Rs,
                         self.lr: cur_lr,
                         self.entropy_coef: cur_beta})
                         
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)



# --- ADD THIS TO agents/policies.py ---
class GaussianCNNACPolicy(ACPolicy):
    def __init__(self, n_s, n_a, n_step, name='gaussian_cnn_policy'):
        super().__init__(n_a, n_s, n_step, 'gaussian_cnn', name)
        
        with tf.variable_scope(self.name):
            # 1. Inputs
            self.ob_fw = tf.placeholder(tf.float32, [None, n_s], name='ob')
            self.done_fw = tf.placeholder(tf.float32, [None], name='done')
            
            # 2. CNN Architecture (Same as before)
            grid_size = 5
            channels = int(n_s / (grid_size * grid_size))
            ob_reshaped = tf.reshape(self.ob_fw, [-1, grid_size, grid_size, channels])

            conv1 = tf.layers.conv2d(ob_reshaped, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3, padding='valid', activation=tf.nn.relu)
            flat = tf.layers.flatten(conv2)
            fc = tf.layers.dense(flat, 128, activation=tf.nn.relu)

            # 3. Continuous Output Heads
            # mu: The mean of the action distribution (unbounded logits)
            self.mu = tf.layers.dense(fc, n_a, activation=None) 
            
            # log_std: Learnable log standard deviation (parameter)
            self.log_std = tf.get_variable(name='log_std', initializer=-0.5 * np.ones(n_a, dtype=np.float32))
            self.std = tf.exp(self.log_std)

            # 4. Action Sampling (Reparameterization Trick)
            # self.pi is now the actual SAMPLED action vector, not probabilities
            self.pi = self.mu + tf.random_normal(tf.shape(self.mu)) * self.std
            
            # Value Head
            self.v = tf.layers.dense(fc, 1, activation=None)
            self.v = tf.squeeze(self.v)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi)
        if 'v' in out_type:
            outs.append(self.v)
        
        # FIX: Wrap 'ob' and 'done' in np.array([]) to add the batch dimension (Batch Size = 1)
        # Input 'ob' shape: (300,) -> Becomes (1, 300)
        feed_dict = {
            self.ob_fw: np.array([ob]), 
            self.done_fw: np.array([done])
        }
        
        out_values = sess.run(outs, feed_dict=feed_dict)
        
        # Use the existing helper to return single value or tuple
        return self._return_forward_outs(out_values)

    # def prepare_loss(self, v_coef, max_grad_norm, alpha, epsilon):
    #     # FIX: Use 'None' for the batch dimension to allow variable batch sizes 
    #     # (e.g., when an episode ends early with fewer steps than batch_size).
        
    #     self.A = tf.placeholder(tf.float32, [None, self.n_a], name='action_ph')
    #     self.ADV = tf.placeholder(tf.float32, [None], name='advantage_ph')
    #     self.R = tf.placeholder(tf.float32, [None], name='return_ph')
    #     self.entropy_coef = tf.placeholder(tf.float32, [], name='entropy_coef_ph')

    #     # 1. Gaussian Log Likelihood Loss
    #     # log_prob = -0.5 * sum(( (x-mu)/std )^2 + 2*log_std + log(2pi) )
    #     log_prob = -0.5 * tf.reduce_sum(tf.square((self.A - self.mu) / self.std), axis=1) \
    #                - 0.5 * tf.reduce_sum(2.0 * self.log_std, axis=0) \
    #                - 0.5 * self.n_a * np.log(2 * np.pi)
        
    #     # 2. Policy Loss (Maximize log_prob * Advantage)
    #     policy_loss = -tf.reduce_mean(log_prob * self.ADV)

    #     # 3. Entropy (for Gaussian: 0.5 + 0.5 * log(2pi) + log_std)
    #     entropy = tf.reduce_sum(self.log_std + 0.5 + 0.5 * np.log(2 * np.pi), axis=0)
    #     entropy_loss = -entropy * self.entropy_coef # Maximize entropy

    #     # 4. Value Loss
    #     value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef

    #     self.loss = policy_loss + value_loss + entropy_loss
class GaussianCNNACPolicy(ACPolicy):
    def __init__(self, n_s, n_a, n_step, name='gaussian_cnn_policy'):
        super().__init__(n_a, n_s, n_step, 'gaussian_cnn', name)
        
        with tf.variable_scope(self.name):
            # 1. Inputs
            self.ob_fw = tf.placeholder(tf.float32, [None, n_s], name='ob')
            self.done_fw = tf.placeholder(tf.float32, [None], name='done')
            
            # 2. CNN Architecture
            # We assume n_s corresponds to a 5x5 grid with C channels flattened
            grid_size = 5
            channels = int(n_s / (grid_size * grid_size))
            ob_reshaped = tf.reshape(self.ob_fw, [-1, grid_size, grid_size, channels])

            conv1 = tf.layers.conv2d(ob_reshaped, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3, padding='valid', activation=tf.nn.relu)
            flat = tf.layers.flatten(conv2)
            fc = tf.layers.dense(flat, 128, activation=tf.nn.relu)

            # 3. Continuous Output Heads
            # mu: The mean of the action distribution
            self.mu = tf.layers.dense(fc, n_a, activation=None) 
            
            # log_std: Learnable log standard deviation (starting at -0.5)
            self.log_std = tf.get_variable(name='log_std', initializer=-0.5 * np.ones(n_a, dtype=np.float32))
            self.std = tf.exp(self.log_std)

            # 4. Action Sampling (Reparameterization Trick)
            self.pi = self.mu + tf.random_normal(tf.shape(self.mu)) * self.std
            
            # Value Head
            self.v = tf.layers.dense(fc, 1, activation=None)
            self.v = tf.squeeze(self.v)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi)
        if 'v' in out_type:
            outs.append(self.v)
        
        # Batch size of 1 for inference
        feed_dict = {
            self.ob_fw: np.array([ob]), 
            self.done_fw: np.array([done])
        }
        
        out_values = sess.run(outs, feed_dict=feed_dict)
        return self._return_forward_outs(out_values)

    def prepare_loss(self, v_coef, max_grad_norm, alpha, epsilon):
        # Placeholders with 'None' to accept variable batch sizes
        self.A = tf.placeholder(tf.float32, [None, self.n_a], name='action_ph')
        self.ADV = tf.placeholder(tf.float32, [None], name='advantage_ph')
        self.R = tf.placeholder(tf.float32, [None], name='return_ph')
        self.entropy_coef = tf.placeholder(tf.float32, [], name='entropy_coef_ph')

        # 1. Gaussian Log Likelihood
        # log_prob = -0.5 * sum(( (x-mu)/std )^2 + 2*log_std + log(2pi) )
        log_prob = -0.5 * tf.reduce_sum(tf.square((self.A - self.mu) / self.std), axis=1) \
                   - 0.5 * tf.reduce_sum(2.0 * self.log_std, axis=0) \
                   - 0.5 * self.n_a * np.log(2 * np.pi)
        
        # 2. Policy Loss
        policy_loss = -tf.reduce_mean(log_prob * self.ADV)
        
        # 3. Entropy (Maximize entropy to encourage exploration)
        entropy = tf.reduce_sum(self.log_std + 0.5 + 0.5 * np.log(2 * np.pi), axis=0)
        entropy_loss = -entropy * self.entropy_coef 

        # 4. Value Loss
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        
        # 5. Total Loss
        self.loss = policy_loss + value_loss + entropy_loss

        # 6. Optimization & Gradient Norm
        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        else:
            self.grad_norm = tf.global_norm(grads) # Just monitor it if not clipping
        
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha, epsilon=epsilon)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        
        # 7. TensorBoard Summaries
        self.summary = tf.summary.merge([
            tf.summary.scalar('Loss/Policy_Loss', policy_loss),
            tf.summary.scalar('Loss/Value_Loss', value_loss),
            tf.summary.scalar('Loss/Entropy_Loss', entropy_loss),
            tf.summary.scalar('Loss/Total_Loss', self.loss),
            tf.summary.scalar('Train/Policy_Entropy', entropy),
            tf.summary.scalar('Train/Grad_Norm', self.grad_norm),
            tf.summary.scalar('Train/Sigma_Mean', tf.reduce_mean(self.std)) # Critical for debugging exploration
        ])

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta, summary_writer=None, global_step=None):
        feed_dict = {
            self.ob_fw: obs, 
            self.done_fw: dones,
            self.A: acts,     
            self.ADV: Advs,
            self.R: Rs,
            self.lr: cur_lr,
            self.entropy_coef: cur_beta
        }
        
        # Only run summary op if we have a writer (saves time)
        if summary_writer is not None:
            ops = [self.summary, self._train]
            outs = sess.run(ops, feed_dict=feed_dict)
            summary_writer.add_summary(outs[0], global_step=global_step)
        else:
            sess.run(self._train, feed_dict=feed_dict)


class GaussianGCNACPolicy(ACPolicy):
    def __init__(self, n_s, n_a, n_step, adj_matrix, num_nodes, features_per_node, name='gaussian_gcn'):
        super().__init__(n_a, n_s, n_step, 'gaussian_gcn', name)
        self.adj = tf.constant(adj_matrix, dtype=tf.float32)
        self.num_nodes = num_nodes
        self.features_per_node = features_per_node
        
        with tf.variable_scope(self.name):
            self.ob_fw = tf.placeholder(tf.float32, [None, n_s], name='ob') # Flattened inputs
            self.done_fw = tf.placeholder(tf.float32, [None], name='done')
            
            # --- GCN Forward Pass ---
            # 1. Reshape Input: [Batch, Nodes * Features] -> [Batch, Nodes, Features]
            h = tf.reshape(self.ob_fw, [-1, self.num_nodes, self.features_per_node])
            
            # 2. Graph Convolution Layer 1
            # AXW
            h = self._gcn_layer(h, 64, 'gcn1', activation=tf.nn.relu)
            
            # 3. Graph Convolution Layer 2
            h = self._gcn_layer(h, 64, 'gcn2', activation=tf.nn.relu)
            
            # 4. Global Pooling (Readout)
            # Max pooling over nodes to get graph embedding
            graph_embedding = tf.reduce_max(h, axis=1) 
            
            # 5. Dense Layers
            fc = tf.layers.dense(graph_embedding, 128, activation=tf.nn.relu)
            
            # 6. Heads
            self.mu = tf.layers.dense(fc, n_a)
            self.log_std = tf.get_variable('log_std', initializer=-0.5 * np.ones(n_a, dtype=np.float32))
            self.std = tf.exp(self.log_std)
            self.pi = self.mu + tf.random_normal(tf.shape(self.mu)) * self.std
            
            self.v = tf.layers.dense(fc, 1)
            self.v = tf.squeeze(self.v)

    def _gcn_layer(self, x, filters, name, activation=None):
        with tf.variable_scope(name):
            # x shape: [Batch, Nodes, In_Feat]
            # Weights: [In_Feat, Out_Feat]
            in_feat = x.get_shape().as_list()[-1]
            W = tf.get_variable('W', [in_feat, filters], initializer=tf.glorot_uniform_initializer())
            
            # Linear Transform: XW
            # Reshape to 2D for matmul: [Batch*Nodes, In_Feat]
            x_flat = tf.reshape(x, [-1, in_feat])
            xw_flat = tf.matmul(x_flat, W)
            xw = tf.reshape(xw_flat, [-1, self.num_nodes, filters])
            
            # Graph Propagation: A * XW
            # Use scan or permute for batch matmul
            # A shape: [Nodes, Nodes], xw shape: [Batch, Nodes, Filters]
            # Transpose xw to [Batch, Filters, Nodes] for easier matmul or use tensordot
            axw = tf.tensordot(xw, self.adj, axes=[[1], [1]]) # [Batch, Filters, Nodes]
            axw = tf.transpose(axw, [0, 2, 1]) # Back to [Batch, Nodes, Filters]
            
            if activation:
                return activation(axw)
            return axw

    def forward(self, sess, ob, done, out_type='pv'):
        outs = []
        if 'p' in out_type: outs.append(self.pi)
        if 'v' in out_type: outs.append(self.v)
        feed_dict = {self.ob_fw: np.array([ob]), self.done_fw: np.array([done])}
        return self._return_forward_outs(sess.run(outs, feed_dict=feed_dict))
    
    # ... (Copy backward/prepare_loss from GaussianCNNACPolicy) ...
    def prepare_loss(self, v_coef, max_grad_norm, alpha, epsilon):
        # Placeholders with 'None' to accept variable batch sizes
        self.A = tf.placeholder(tf.float32, [None, self.n_a], name='action_ph')
        self.ADV = tf.placeholder(tf.float32, [None], name='advantage_ph')
        self.R = tf.placeholder(tf.float32, [None], name='return_ph')
        self.entropy_coef = tf.placeholder(tf.float32, [], name='entropy_coef_ph')

        # 1. Gaussian Log Likelihood
        # log_prob = -0.5 * sum(( (x-mu)/std )^2 + 2*log_std + log(2pi) )
        log_prob = -0.5 * tf.reduce_sum(tf.square((self.A - self.mu) / self.std), axis=1) \
                   - 0.5 * tf.reduce_sum(2.0 * self.log_std, axis=0) \
                   - 0.5 * self.n_a * np.log(2 * np.pi)
        
        # 2. Policy Loss
        policy_loss = -tf.reduce_mean(log_prob * self.ADV)
        
        # 3. Entropy (Maximize entropy to encourage exploration)
        entropy = tf.reduce_sum(self.log_std + 0.5 + 0.5 * np.log(2 * np.pi), axis=0)
        entropy_loss = -entropy * self.entropy_coef 

        # 4. Value Loss
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        
        # 5. Total Loss
        self.loss = policy_loss + value_loss + entropy_loss

        # 6. Optimization & Gradient Norm
        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        else:
            self.grad_norm = tf.global_norm(grads) # Just monitor it if not clipping
        
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha, epsilon=epsilon)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        
        # 7. TensorBoard Summaries
        self.summary = tf.summary.merge([
            tf.summary.scalar('Loss/Policy_Loss', policy_loss),
            tf.summary.scalar('Loss/Value_Loss', value_loss),
            tf.summary.scalar('Loss/Entropy_Loss', entropy_loss),
            tf.summary.scalar('Loss/Total_Loss', self.loss),
            tf.summary.scalar('Train/Policy_Entropy', entropy),
            tf.summary.scalar('Train/Grad_Norm', self.grad_norm),
            tf.summary.scalar('Train/Sigma_Mean', tf.reduce_mean(self.std)) # Critical for debugging exploration
        ])

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta, summary_writer=None, global_step=None):
        feed_dict = {
            self.ob_fw: obs, 
            self.done_fw: dones,
            self.A: acts,     
            self.ADV: Advs,
            self.R: Rs,
            self.lr: cur_lr,
            self.entropy_coef: cur_beta
        }
        
        # Only run summary op if we have a writer (saves time)
        if summary_writer is not None:
            ops = [self.summary, self._train]
            outs = sess.run(ops, feed_dict=feed_dict)
            summary_writer.add_summary(outs[0], global_step=global_step)
        else:
            sess.run(self._train, feed_dict=feed_dict)
