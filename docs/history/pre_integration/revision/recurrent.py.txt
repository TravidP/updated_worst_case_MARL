"""Same legacy LSTM equations/variables, compact training graph instead of unrolling.

Inference retains the original one-step implementation. Numerical equivalence of the
training recurrence and its gradients is checked against the legacy implementation.
"""
from tf_compat import tf
from agents.utils import fc, lstm, DEFAULT_METHOD, DEFAULT_SCALE, DEFAULT_MODE
from agents.policies import LstmACPolicy, FPLstmACPolicy


def compact_lstm(xs, dones, state, scope):
    width = int(state.shape[0]) // 2
    with tf.variable_scope(scope):
        wx = tf.get_variable('wx', [int(xs.shape[1]), width * 4], initializer=DEFAULT_METHOD(DEFAULT_SCALE, DEFAULT_MODE))
        wh = tf.get_variable('wh', [width, width * 4], initializer=DEFAULT_METHOD(DEFAULT_SCALE, DEFAULT_MODE))
        bias = tf.get_variable('b', [width * 4], initializer=tf.zeros_initializer())
    c, h = tf.split(tf.expand_dims(state, 0), 2, axis=1)
    length = tf.shape(xs)[0]
    output = tf.TensorArray(tf.float32, size=length)

    def body(index, c, h, output):
        c = c * (1. - dones[index])
        h = h * (1. - dones[index])
        z = tf.matmul(tf.expand_dims(xs[index], 0), wx) + tf.matmul(h, wh) + bias
        i, f, o, u = tf.split(z, 4, axis=1)
        c = tf.sigmoid(f) * c + tf.sigmoid(i) * tf.tanh(u)
        h = tf.sigmoid(o) * tf.tanh(c)
        return index + 1, c, h, output.write(index, h[0])

    _, c, h, output = tf.while_loop(lambda index, *_: index < length, body,
                                   (tf.constant(0), c, h, output), parallel_iterations=1)
    values = output.stack()
    values.set_shape([xs.shape[0], width])
    return values, tf.squeeze(tf.concat([c, h], axis=1))


class CompactNetwork:
    def _build_net(self, in_type, out_type):
        ob = self.ob_fw if in_type == 'forward' else self.ob_bw
        done = self.done_fw if in_type == 'forward' else self.done_bw
        state = self.states[0 if out_type == 'pi' else 1]
        if hasattr(self, 'n_fc_fp'):
            parts = [fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave),
                     fc(ob[:, self.n_s + self.n_w:], out_type + '_fcf', self.n_fc_fp)]
            if self.n_w:
                parts.append(fc(ob[:, self.n_s:self.n_s + self.n_w], out_type + '_fct', self.n_fc_wait))
            h = tf.concat(parts, 1)
        elif self.n_w:
            h = tf.concat([fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave),
                           fc(ob[:, self.n_s:], out_type + '_fct', self.n_fc_wait)], 1)
        else:
            h = fc(ob, out_type + '_fcw', self.n_fc_wave)
        recurrence = lstm if in_type == 'forward' else compact_lstm
        h, new_state = recurrence(h, done, state, out_type + '_lstm')
        return self._build_out_net(h, out_type), new_state


class CompactPolicy(CompactNetwork, LstmACPolicy):
    pass


class CompactFingerprintPolicy(CompactNetwork, FPLstmACPolicy):
    pass
