import itertools
import logging
import numpy as np
from tf_compat import tf
import time
import os
import pandas as pd
import subprocess
import threading
import faulthandler
import signal


def is_policy_gradient_agent(agent):
    agent = str(agent).strip().lower()
    return agent.endswith('a2c') or agent == 'ppo'


def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    cmd = 'cp %s %s' % (src_dir, tar_dir)
    subprocess.check_call(cmd, shell=True)


def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False


def plot_train(data_dirs, labels):
    pass

def plot_evaluation(data_dirs, labels):
    pass


class Counter:
    def __init__(self, total_step, test_step, log_step, start_step=0):
        start_step = int(start_step)
        self.counter = itertools.count(start_step + 1)
        self.cur_step = start_step
        self.cur_test_step = start_step
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False
        # self.init_test = True

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        # if self.init_test:
        #     self.init_test = False
        #     return True
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    # def update_test(self, reward):
    #     if self.prev_reward is not None:
    #         if abs(self.prev_reward - reward) <= self.delta_reward:
    #             self.stop = True
    #     self.prev_reward = reward

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop

class Trainer():
    # Added model_path=None to the arguments
    def __init__(self, env, model, global_counter, summary_writer, run_test, output_path=None, model_path=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent
        self.model = model
        self.sess = self.model.sess
        self.n_step = self.model.n_step
        self.summary_writer = summary_writer
        self.run_test = run_test
        self.model_path = model_path  # Store the path
        assert self.env.T % self.n_step == 0
        self.data = []
        self.output_path = output_path
        self.stage_warn_sec = float(getattr(self.env, 'trainer_stage_warn_sec', 60.0))
        self.stall_watchdog_sec = int(getattr(self.env, 'stall_watchdog_sec', 180))
        self.stall_watchdog_poll_sec = int(getattr(self.env, 'stall_watchdog_poll_sec', 10))
        self._watchdog_lock = threading.Lock()
        self._watchdog_last_touch = time.time()
        self._watchdog_label = 'init'
        self._watchdog_last_dump = 0.0
        self._watchdog_running = False
        self._watchdog_thread = None
        dump_dir = self.output_path if self.output_path else '.'
        self._watchdog_dump_path = os.path.join(dump_dir, 'stall_watchdog_stacks.log')
        self._signal_dump_path = os.path.join(dump_dir, 'manual_signal_stacks.log')
        self._signal_dump_fh = None
        self._signal_dump_registered = False
        if run_test:
            self.test_num = self.env.test_num
            logging.info('Testing: total test num: %d' % self.test_num)
        self._init_summary()
        self._setup_manual_stack_dump_signal()
        self._start_watchdog()
    def _init_summary(self):
        self.train_reward = tf.placeholder(tf.float32, [])
        self.train_summary = tf.summary.scalar('train_reward', self.train_reward)
        self.test_reward = tf.placeholder(tf.float32, [])
        self.test_summary = tf.summary.scalar('test_reward', self.test_reward)

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            summ = self.sess.run(self.train_summary, {self.train_reward: reward})
        else:
            summ = self.sess.run(self.test_summary, {self.test_reward: reward})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def _touch_watchdog(self, label):
        if not self._watchdog_running:
            return
        with self._watchdog_lock:
            self._watchdog_last_touch = time.time()
            self._watchdog_label = label

    def _watchdog_loop(self):
        while self._watchdog_running:
            time.sleep(max(1, self.stall_watchdog_poll_sec))
            now = time.time()
            with self._watchdog_lock:
                last_touch = self._watchdog_last_touch
                label = self._watchdog_label
            stalled_for = now - last_touch
            if stalled_for < self.stall_watchdog_sec:
                continue
            # Throttle dumps to avoid log storms.
            if (now - self._watchdog_last_dump) < max(30, self.stall_watchdog_sec // 2):
                continue
            self._watchdog_last_dump = now
            logging.warning("Watchdog: no training progress for %.1fs (stage=%s). Dumping stacks...",
                            stalled_for, label)
            try:
                with open(self._watchdog_dump_path, 'a') as fh:
                    fh.write(
                        '\n=== %s stalled_for=%.1fs stage=%s ===\n'
                        % (time.strftime('%Y-%m-%d %H:%M:%S'), stalled_for, label)
                    )
                    faulthandler.dump_traceback(file=fh, all_threads=True)
            except Exception as e:
                logging.warning('Watchdog: failed to write stack dump: %s', e)

    def _start_watchdog(self):
        if self.stall_watchdog_sec <= 0:
            logging.info('Watchdog disabled (stall_watchdog_sec=%d).', self.stall_watchdog_sec)
            return
        if self._watchdog_running:
            return
        self._watchdog_running = True
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()
        logging.info(
            'Watchdog enabled: stall=%ss poll=%ss dump_file=%s',
            self.stall_watchdog_sec, self.stall_watchdog_poll_sec, self._watchdog_dump_path
        )

    def _setup_manual_stack_dump_signal(self):
        if not hasattr(signal, 'SIGUSR1'):
            return
        try:
            self._signal_dump_fh = open(self._signal_dump_path, 'a')
            faulthandler.register(signal.SIGUSR1, file=self._signal_dump_fh, all_threads=True)
            self._signal_dump_registered = True
            logging.info(
                'Manual stack dump enabled: send `kill -USR1 %d` to write %s',
                os.getpid(), self._signal_dump_path
            )
        except Exception as e:
            self._signal_dump_registered = False
            if self._signal_dump_fh is not None:
                try:
                    self._signal_dump_fh.close()
                except Exception:
                    pass
                self._signal_dump_fh = None
            logging.warning('Failed to enable SIGUSR1 stack dumps: %s', e)

    def _teardown_manual_stack_dump_signal(self):
        if self._signal_dump_registered and hasattr(signal, 'SIGUSR1'):
            try:
                faulthandler.unregister(signal.SIGUSR1)
            except Exception:
                pass
            self._signal_dump_registered = False
        if self._signal_dump_fh is not None:
            try:
                self._signal_dump_fh.close()
            except Exception:
                pass
            self._signal_dump_fh = None

    def _stop_watchdog(self):
        if not self._watchdog_running:
            self._teardown_manual_stack_dump_signal()
            return
        self._watchdog_running = False
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=1.0)
            self._watchdog_thread = None
        self._teardown_manual_stack_dump_signal()

    def explore(self, prev_ob, prev_done):
        ob = prev_ob
        done = prev_done
        rewards = []
        for _ in range(self.n_step):
            self._touch_watchdog('explore:forward')
            forward_start = time.time()
            if is_policy_gradient_agent(self.agent):
                policy, value = self.model.forward(ob, done)
                # need to update fingerprint before calling step
                if self.agent == 'ma2c':
                    self.env.update_fingerprint(policy)
                if self.agent == 'a2c':
                    action = np.random.choice(np.arange(len(policy)), p=policy)
                else:
                    action = []
                    for pi in policy:
                        action.append(np.random.choice(np.arange(len(pi)), p=pi))
            else:
                action, policy = self.model.forward(ob, mode='explore')
            forward_elapsed = time.time() - forward_start
            if forward_elapsed > self.stage_warn_sec:
                logging.warning('Training: slow forward %.2fs at global step %d',
                                forward_elapsed, self.global_counter.cur_step)

            self._touch_watchdog('explore:env_step')
            env_step_start = time.time()
            next_ob, reward, done, global_reward = self.env.step(action)
            env_step_elapsed = time.time() - env_step_start
            if env_step_elapsed > self.stage_warn_sec:
                logging.warning('Training: slow env.step %.2fs at global step %d',
                                env_step_elapsed, self.global_counter.cur_step)

            rewards.append(global_reward)
            global_step = self.global_counter.next()
            self.cur_step += 1

            self._touch_watchdog('explore:add_transition')
            add_start = time.time()
            if is_policy_gradient_agent(self.agent):
                if self.agent == 'ppo':
                    self.model.add_transition(ob, action, reward, value, done, policy)
                else:
                    self.model.add_transition(ob, action, reward, value, done)
            else:
                self.model.add_transition(ob, action, reward, next_ob, done)
            add_elapsed = time.time() - add_start
            if add_elapsed > self.stage_warn_sec:
                logging.warning('Training: slow add_transition %.2fs at global step %d',
                                add_elapsed, global_step)
            # logging
            if self.global_counter.should_log():
                logging.info('''Training: global step %d, episode step %d,
                                   ob: %s, a: %s, pi: %s, r: %.2f, train r: %.2f, done: %r''' %
                             (global_step, self.cur_step,
                              str(ob), str(action), str(policy), global_reward, np.mean(reward), done))
            # # termination
            # if done:
            #     self.env.terminate()
            #     time.sleep(2)
            #     ob = self.env.reset()
            #     self._add_summary(cum_reward / float(self.cur_step), global_step)
            #     cum_reward = 0
            #     self.cur_step = 0
            # else:
            if done:
                break
            ob = next_ob
            self._touch_watchdog('explore:loop_continue')
        if is_policy_gradient_agent(self.agent):
            if done:
                R = 0 if self.agent == 'a2c' else [0] * self.model.n_agent
            else:
                self._touch_watchdog('explore:value_bootstrap')
                R = self.model.forward(ob, False, 'v')
        else:
            R = 0
        return ob, done, R, rewards

    def perform(self, test_ind, demo=False, policy_type='default'):
        ob = self.env.reset(gui=demo, test_ind=test_ind)
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model.reset()
        rewards = []
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            elif is_policy_gradient_agent(self.agent):
                # policy-based on-poicy learning
                policy = self.model.forward(ob, done, 'p')
                if self.agent == 'ma2c':
                    self.env.update_fingerprint(policy)
                if self.agent == 'a2c':
                    if policy_type != 'deterministic':
                        action = np.random.choice(np.arange(len(policy)), p=policy)
                    else:
                        action = np.argmax(np.array(policy))
                else:
                    action = []
                    for pi in policy:
                        if policy_type != 'deterministic':
                            action.append(np.random.choice(np.arange(len(pi)), p=pi))
                        else:
                            action.append(np.argmax(np.array(pi)))
            else:
                # value-based off-policy learning
                if policy_type != 'stochastic':
                    action, _ = self.model.forward(ob)
                else:
                    action, _ = self.model.forward(ob, stochastic=True)
            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    def run_thread(self, coord):
        '''Multi-threading is disabled'''
        ob = self.env.reset()
        done = False
        cum_reward = 0
        while not coord.should_stop():
            ob, done, R, cum_reward = self.explore(ob, done, cum_reward)
            global_step = self.global_counter.cur_step
            if is_policy_gradient_agent(self.agent):
                self.model.backward(R, self.summary_writer, global_step)
            else:
                self.model.backward(self.summary_writer, global_step)
            self.summary_writer.flush()
            if (self.global_counter.should_stop()) and (not coord.should_stop()):
                self.env.terminate()
                coord.request_stop()
                logging.info('Training: stop condition reached!')
                return

    def run(self):
        episode_idx = 0
        try:
            self._touch_watchdog('run:start')
            while not self.global_counter.should_stop():
                episode_idx += 1
                episode_start = time.time()
                # test
                if self.run_test and self.global_counter.should_test():
                    rewards = []
                    global_step = self.global_counter.cur_step
                    self.env.train_mode = False
                    for test_ind in range(self.test_num):
                        self._touch_watchdog('run:test_perform')
                        mean_reward, std_reward = self.perform(test_ind)
                        self._touch_watchdog('run:test_terminate')
                        self.env.terminate()
                        rewards.append(mean_reward)
                        log = {'agent': self.agent,
                               'step': global_step,
                               'test_id': test_ind,
                               'avg_reward': mean_reward,
                               'std_reward': std_reward}
                        self.data.append(log)
                    avg_reward = np.mean(np.array(rewards))
                    self._touch_watchdog('run:test_summary')
                    self._add_summary(avg_reward, global_step, is_train=False)
                    logging.info('Testing: global step %d, avg R: %.2f' %
                                 (global_step, avg_reward))
                # train
                self.env.train_mode = True
                self._touch_watchdog('run:env_reset')
                ob = self.env.reset()
                done = True
                self.model.reset()
                self.cur_step = 0
                rewards = []
                logging.info('Training: episode %d started at global step %d',
                             episode_idx, self.global_counter.cur_step)
                while True:
                    self._touch_watchdog('run:explore')
                    explore_start = time.time()
                    ob, done, R, cur_rewards = self.explore(ob, done)
                    explore_elapsed = time.time() - explore_start
                    if explore_elapsed > 120:
                        logging.warning('Training: slow explore chunk %.2fs at global step %d',
                                        explore_elapsed, self.global_counter.cur_step)
                    rewards += cur_rewards
                    global_step = self.global_counter.cur_step

                    # --- NEW CODE START ---
                    # Checkpoint saving every 10000 steps
                    if self.model_path and global_step % 10000 == 0:
                        self._touch_watchdog('run:checkpoint_save')
                        logging.info('Training: checkpoint model at step %d ...' % global_step)
                        save_start = time.time()
                        self.model.save(self.model_path, global_step)
                        logging.info('Training: checkpoint finished in %.2fs',
                                     time.time() - save_start)
                    # --- NEW CODE END ---

                    self._touch_watchdog('run:backward')
                    backward_start = time.time()
                    backward_failed = False
                    try:
                        if is_policy_gradient_agent(self.agent):
                            self.model.backward(R, self.summary_writer, global_step)
                        else:
                            self.model.backward(self.summary_writer, global_step)
                    except tf.errors.DeadlineExceededError:
                        backward_failed = True
                        logging.error(
                            'Training: backward timed out at global step %d. '
                            'Terminating current episode and continuing.',
                            global_step
                        )
                    except Exception as e:
                        backward_failed = True
                        logging.error('Training: backward failed at global step %d: %s',
                                      global_step, e)
                    backward_elapsed = time.time() - backward_start
                    if backward_elapsed > 60:
                        logging.warning('Training: slow backward %.2fs at global step %d',
                                        backward_elapsed, global_step)
                    if backward_failed:
                        self._touch_watchdog('run:backward_failed_terminate')
                        terminate_start = time.time()
                        try:
                            self.env.terminate()
                        except Exception as e:
                            logging.warning('Training: env termination after backward failure raised: %s', e)
                        logging.info('Training: env termination after backward failure finished in %.2fs',
                                     time.time() - terminate_start)
                        break
                    # termination
                    if done:
                        logging.info('Training: episode %d done at global step %d, terminating env ...',
                                     episode_idx, global_step)
                        self._touch_watchdog('run:episode_terminate')
                        terminate_start = time.time()
                        self.env.terminate()
                        logging.info('Training: env termination finished in %.2fs',
                                     time.time() - terminate_start)
                        break
                rewards = np.array(rewards)
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                log = {'agent': self.agent,
                       'step': global_step,
                       'test_id': -1,
                       'avg_reward': mean_reward,
                       'std_reward': std_reward}
                self.data.append(log)
                self._touch_watchdog('run:summary_flush')
                summary_start = time.time()
                self._add_summary(mean_reward, global_step)
                self.summary_writer.flush()
                summary_elapsed = time.time() - summary_start
                if summary_elapsed > 30:
                    logging.warning('Training: slow summary write %.2fs at global step %d',
                                    summary_elapsed, global_step)
                logging.info('Training: episode %d finished in %.2fs (global step %d, avg R %.2f)',
                             episode_idx, time.time() - episode_start, global_step, mean_reward)
            df = pd.DataFrame(self.data)
            df.to_csv(self.output_path + 'train_reward.csv')
        finally:
            self._stop_watchdog()


class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def _init_summary(self):
        self.reward = tf.placeholder(tf.float32, [])
        self.summary = tf.summary.scalar('test_reward', self.reward)

    def run_offline(self):
        # enable traffic measurments for offline test
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.perform(test_ind))
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()

    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Evaluator(Tester):
    def __init__(self, env, model, output_path, demo=False, policy_type='default'):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.demo = demo
        self.policy_type = policy_type

    def run(self):
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward, _ = self.perform(test_ind, demo=self.demo, policy_type=self.policy_type)
            self.env.terminate()
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()
