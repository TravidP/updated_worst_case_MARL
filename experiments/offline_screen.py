"""Frozen-trajectory, fixed-batch screening for revised reward scales."""
import argparse
import copy
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np

from agents.controller import Controller
from envs.experiment_env import make_environment
from experiments.core import Streams, write_json
from experiments.demand import materialize, mixture
from tf_compat import tf


def tensor_hash(arrays):
    value = hashlib.sha256()
    for array in arrays:
        value.update(np.asarray(array).tobytes())
    return value.hexdigest()


def collect(network, controller, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = model = None
    try:
        env = make_environment(network, controller, output / 'runtime', 9002)
        streams = Streams(9002)
        obs = env.reset_episode(int(streams['sumo'].randint(1, 2147483647)))
        env.prepare_routes()
        model = Controller(env, controller, 9002)
        model.reset()
        initial_hash = tensor_hash(model.sess.run(model.trainable))
        transitions = []
        for block in range(11):
            vehicles = materialize(
                mixture(env.groups, np.eye(11)[block]), block * 600, 600,
                streams['demand'], env.resolve_route, 'frozen_b{}'.format(block))
            env.inject(vehicles)
            for _ in range(120):
                states = ([p.states_fw.copy() for p in model.policies]
                          if controller != 'iqll' else None)
                previous_done = model.previous_done
                decision = model.act(obs, env, False)
                nxt, rewards, done = env.step(decision[0])
                learner, clipped = model.observe(obs, decision, rewards, nxt, done, False)
                env.controller_rows[-1].update(
                    learner_rewards=learner.tolist(),
                    reward_clipped=clipped.astype(int).tolist())
                transitions.append({
                    'obs': copy.deepcopy(obs), 'next_obs': copy.deepcopy(nxt),
                    'actions': list(decision[0]), 'raw_rewards': np.asarray(rewards),
                    'done': bool(done), 'previous_done': previous_done,
                    'states': states,
                })
                obs = nxt
        if tensor_hash(model.sess.run(model.trainable)) != initial_hash:
            raise AssertionError('Frozen collection changed model parameters')
        batch_size = model.batch
        if controller == 'iqll':
            indices = np.random.RandomState(9002).choice(
                len(transitions), batch_size, replace=False).tolist()
        else:
            indices = list(range(len(transitions) - batch_size, len(transitions)))
        with (output / 'batch.pkl').open('xb') as stream:
            pickle.dump({'batch': [transitions[i] for i in indices],
                         'indices': indices, 'initial_hash': initial_hash},
                        stream, protocol=4)
        env.export_episode(output / 'frozen_episode')
        write_json(output / 'summary.json', {
            'network': network, 'controller': controller, 'seed': 9002,
            'simulation_steps': len(transitions), 'learning_steps': 0,
            'batch_size': batch_size, 'indices': indices,
            'initial_hash': initial_hash,
        })
    finally:
        if model is not None:
            model.close()
        if env is not None:
            env.terminate()


def learner_rewards(raw, model):
    scaled = np.asarray(raw, dtype=np.float32) / model.reward_norm
    return (np.clip(scaled, -model.reward_clip, model.reward_clip)
            if model.reward_clip > 0 else scaled)


def iql_targets(rewards, next_q, gamma, dones):
    """Build Q-learning targets without NumPy promoting them to float64."""
    rewards = np.asarray(rewards, dtype=np.float32)
    next_q = np.asarray(next_q, dtype=np.float32)
    not_done = 1. - np.asarray(dones, dtype=np.float32)
    return np.asarray(
        rewards + np.float32(gamma) * np.max(next_q, axis=1) * not_done,
        dtype=np.float32)


def fit(network, controller, trajectory, config, output):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = pickle.loads((Path(trajectory) / 'batch.pkl').read_bytes())
    batch = data['batch']
    env = model = None
    try:
        env = make_environment(network, controller, output / 'runtime', 9002,
                               config_path=config)
        model = Controller(env, controller, 9002)
        if tensor_hash(model.sess.run(model.trainable)) != data['initial_hash']:
            raise AssertionError('Candidate initialization differs from frozen trajectory')
        feeds, mses, norms, factors, operations, variables = {}, [], [], [], [], []
        initial_values = []
        with model.graph.as_default():
            before = set(tf.global_variables())
            for agent, policy in enumerate(model.policies):
                selected = [v for v in model.trainable
                            if (v.name.startswith(policy.name + '/') or
                                v.name.startswith(policy.name + '_')) and
                            (controller == 'iqll' or '/v' in v.name)]
                if not selected:
                    raise AssertionError('No critic/Q variables selected')
                rewards = learner_rewards([t['raw_rewards'][agent] for t in batch], model)
                if controller == 'iqll':
                    next_q = model.sess.run(
                        policy.qvalues,
                        {policy.S: np.asarray([t['next_obs'][agent] for t in batch])})
                    targets = iql_targets(
                        rewards, next_q, model.gamma,
                        [t['done'] for t in batch])
                    prediction = tf.reduce_sum(
                        policy.qvalues * tf.one_hot(policy.A, policy.n_a), axis=1)
                    feeds.update({
                        policy.S: np.asarray([t['obs'][agent] for t in batch]),
                        policy.A: np.asarray([t['actions'][agent] for t in batch]),
                    })
                    coefficient = 1.
                else:
                    running, values = 0., []
                    for reward, transition in zip(reversed(rewards), reversed(batch)):
                        running = reward + model.gamma * running * (not transition['done'])
                        values.append(running)
                    targets = np.asarray(values[::-1], dtype=np.float32)
                    prediction = policy.v
                    feeds.update({
                        policy.ob_bw: np.asarray([t['obs'][agent] for t in batch]),
                        policy.done_bw: np.asarray([t['previous_done'] for t in batch]),
                        policy.states: batch[0]['states'][agent],
                    })
                    coefficient = .5 * model.config.getfloat('value_coef')
                mse = tf.reduce_mean(tf.square(prediction - tf.constant(targets)))
                gradients = tf.gradients(coefficient * mse, selected)
                norm = tf.global_norm(gradients)
                clipped, _ = tf.clip_by_global_norm(
                    gradients, model.config.getfloat('max_grad_norm'))
                with tf.variable_scope('offline_screen_{}'.format(agent)):
                    if controller in ('ppo', 'iqll'):
                        epsilon_key = 'adam_epsilon'
                        optimizer = tf.train.AdamOptimizer(
                            model.lr, epsilon=model.config.getfloat(epsilon_key))
                    else:
                        optimizer = tf.train.RMSPropOptimizer(
                            model.lr, decay=model.config.getfloat('rmsp_alpha'),
                            epsilon=model.config.getfloat('rmsp_epsilon'))
                    operations.append(optimizer.apply_gradients(zip(clipped, selected)))
                variables.append(selected)
                mses.append(mse)
                norms.append(norm)
                factors.append(tf.minimum(
                    1., model.config.getfloat('max_grad_norm') /
                    tf.maximum(norm, 1e-12)))
            created = [v for v in tf.global_variables() if v not in before]
            model.sess.run(tf.variables_initializer(created))
        initial_mse = np.asarray(model.sess.run(mses, feeds))
        initial_values = [model.sess.run(group) for group in variables]
        # The clipping gate applies to the complete frozen trajectory, not only
        # the fixed optimization minibatch.
        controls = Path(trajectory) / 'frozen_episode.controls.jsonl'
        full_raw = []
        with controls.open() as stream:
            for line in stream:
                full_raw.append(json.loads(line)['raw_rewards'])
        raw_matrix = np.asarray(full_raw, dtype=np.float32)
        scaled_matrix = raw_matrix / model.reward_norm
        reward_clip_fraction = (float(np.mean(np.abs(scaled_matrix) > model.reward_clip))
                                if model.reward_clip > 0 else 0.)
        clipping_samples = []
        for update in range(200):
            current_factor = model.sess.run(factors, feeds)
            clipping_samples.extend(float(x) for x in current_factor)
            model.sess.run(operations, feeds)
        final_mse = np.asarray(model.sess.run(mses, feeds))
        changed = []
        for group, original in zip(variables, initial_values):
            current = model.sess.run(group)
            changed.append(any(not np.array_equal(a, b)
                               for a, b in zip(current, original)))
        finite = all(np.all(np.isfinite(x)) for x in model.sess.run(model.variables))
        ratios = final_mse / np.maximum(initial_mse, 1e-12)
        improving = ratios <= .9
        median_factor = float(np.median(clipping_samples))
        passed = (finite and all(changed) and reward_clip_fraction <= .05 and
                  np.mean(improving) >= .8 and
                  median_factor >= .01)
        result = {
            'network': network, 'controller': controller,
            'config': str(Path(config).resolve()),
            'reward_norm': model.reward_norm, 'reward_clip': model.reward_clip,
            'updates': 200, 'agents': len(model.policies),
            'finite': bool(finite), 'all_parameters_changed': bool(all(changed)),
            'improving_agents': int(np.sum(improving)),
            'median_relative_mse': float(np.median(ratios)),
            'median_clipping_factor': median_factor,
            'reward_clip_fraction': reward_clip_fraction, 'passed': bool(passed),
            'wall_seconds': time.monotonic() - started,
        }
        write_json(output / 'summary.json', result)
        return result
    finally:
        if model is not None:
            model.close()
        if env is not None:
            env.terminate()


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    # ``required=`` for subparsers was added in Python 3.7.  The project still
    # supports the Python 3.6 ``deeprlsc`` environment, where passing that
    # keyword raises TypeError before any command can start.  Assigning the
    # Action attribute gives the same argparse validation on both versions.
    sub = value.add_subparsers(dest='command')
    sub.required = True
    collect_parser = sub.add_parser('collect')
    collect_parser.add_argument('--network', choices=['grid', 'monaco'], required=True)
    collect_parser.add_argument('--controller', choices=['ia2c', 'ma2c', 'iqll', 'ppo'], required=True)
    collect_parser.add_argument('--output', required=True)
    fit_parser = sub.add_parser('fit')
    fit_parser.add_argument('--network', choices=['grid', 'monaco'], required=True)
    fit_parser.add_argument('--controller', choices=['ia2c', 'ma2c', 'iqll', 'ppo'], required=True)
    fit_parser.add_argument('--trajectory', required=True)
    fit_parser.add_argument('--config', required=True)
    fit_parser.add_argument('--output', required=True)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'collect':
        collect(args.network, args.controller, args.output)
    else:
        print(json.dumps(fit(args.network, args.controller, args.trajectory,
                             args.config, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
