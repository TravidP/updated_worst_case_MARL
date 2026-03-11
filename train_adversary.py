import os
import logging
import configparser
import numpy as np
from tf_compat import tf
import random
import sys
import time
import json
import threading
import faulthandler
import signal
import glob
import shutil
import argparse
from envs.adversarial_large_grid_env import AdversarialLargeGridEnv
from agents.models import CNNA2C

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _script_rel(*parts):
    return os.path.join(SCRIPT_DIR, *parts)

def _has_model_checkpoint(run_dir):
    model_dir = os.path.join(run_dir, 'model')
    if not os.path.isdir(model_dir):
        return False
    if os.path.exists(os.path.join(model_dir, 'checkpoint')):
        return True
    return bool(glob.glob(os.path.join(model_dir, 'checkpoint-*.index')))

def _pick_existing_with_checkpoint(candidates):
    for path in candidates:
        if _has_model_checkpoint(path):
            return path
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]

def _resolve_default_frozen_paths(agent):
    agent = str(agent).strip().lower()
    if agent == 'ia2c':
        frozen_model_dir = _script_rel('runs', 'ia2c_large')
        frozen_config_path = _script_rel('config', 'config_ia2c_large.ini')
        base_dir = _script_rel('output_adversary', 'ia2c_large')
        return frozen_config_path, frozen_model_dir, base_dir

    if agent == 'ma2c':
        run_candidates = [
            _script_rel('runs', 'ma2c_large'),
            _script_rel('runs', 'ma2c_large_baseline'),
        ]
        frozen_model_dir = _pick_existing_with_checkpoint(run_candidates)
        run_name = os.path.basename(os.path.normpath(frozen_model_dir))
        if run_name == 'ma2c_large_baseline':
            frozen_config_path = _script_rel('config', 'config_ma2c_large_baseline.ini')
        else:
            frozen_config_path = _script_rel('config', 'config_ma2c_large.ini')
        base_dir = _script_rel('output_adversary', run_name)
        return frozen_config_path, frozen_model_dir, base_dir

    if agent == 'iqll':
        frozen_model_dir = _script_rel('runs', 'iqll_large')
        frozen_config_path = _script_rel('config', 'config_iqll_large.ini')
        base_dir = _script_rel('output_adversary', 'iqll_large')
        return frozen_config_path, frozen_model_dir, base_dir

    if agent == 'ppo':
        frozen_model_dir = _script_rel('runs', 'ppo_large')
        frozen_config_path = _script_rel('config', 'config_ppo_large.ini')
        base_dir = _script_rel('output_adversary', 'ppo_large')
        return frozen_config_path, frozen_model_dir, base_dir

    raise ValueError("Unsupported --agent '%s'. Use ma2c, ia2c, iqll, or ppo." % agent)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train adversary (WCE) against a frozen traffic controller.'
    )
    parser.add_argument(
        '--agent',
        type=str,
        default='iqll',
        choices=['ma2c', 'ia2c', 'iqll', 'ppo'],
        help='Frozen traffic controller type used to select default paths.'
    )
    parser.add_argument(
        '--frozen-config',
        type=str,
        default=None,
        help='Path to frozen controller config .ini. Overrides --agent default.'
    )
    parser.add_argument(
        '--frozen-model-dir',
        type=str,
        default=None,
        help='Path to frozen controller run directory (expects data/ and model/).'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Adversary output directory.'
    )
    parser.add_argument(
        '--total-episodes',
        type=int,
        default=500,
        help='Number of adversary training episodes.'
    )
    parser.add_argument(
        '--checkpoint-interval-ep',
        type=int,
        default=5,
        help='Save adversary checkpoint every N episodes.'
    )
    parser.add_argument(
        '--resume-checkpoint',
        type=str,
        default='latest',
        help="Episode index to resume from, or 'latest'."
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable adversary checkpoint resume.'
    )
    return parser.parse_args()

ADVERSARY_REWARD_NORM = 500.0
ADVERSARY_REWARD_CLIP = 5.0
ADVERSARY_TARGET_BATCH_EPISODES = 4


def _prepare_adversary_model_config(model_config_dict, macro_steps_per_episode):
    tuned = dict(model_config_dict)
    configured_batch = max(1, int(float(tuned.get('batch_size', 120))))
    target_batch = max(
        int(macro_steps_per_episode),
        min(configured_batch, int(macro_steps_per_episode) * ADVERSARY_TARGET_BATCH_EPISODES)
    )
    tuned['batch_size'] = str(int(target_batch))
    tuned['reward_norm'] = str(ADVERSARY_REWARD_NORM)
    tuned['reward_clip'] = str(ADVERSARY_REWARD_CLIP)
    return tuned, configured_batch, target_batch

def log_episode_stats(writer, episode, reward, steps, action_history, n_actions):
    """
    Manually logs episode-level statistics to TensorBoard.
    """
    if writer is None:
        return
        
    summary = tf.Summary()
    
    # 1. Performance Metrics
    summary.value.add(tag='Episode/Total_Congestion_Reward', simple_value=reward)
    summary.value.add(tag='Episode/Steps', simple_value=steps)
    
    # 2. Action Distribution (dominant selected scenario weight index)
    if action_history:
        counts = np.bincount(action_history, minlength=n_actions)
        probs = counts / np.sum(counts)
        for act_idx, prob in enumerate(probs):
            summary.value.add(tag=f'Adversary/Action_{act_idx}_Freq', simple_value=prob)

    writer.add_summary(summary, episode)
    writer.flush()

# --- HELPER CLASSES ---
class ConfigWrapper:
    """
    Wraps a dictionary to mimic configparser behavior.
    Provides default values for A2C parameters if they are missing 
    (e.g., when loading an IQL or Greedy config).
    """
    def __init__(self, config_dict):
        self.config = config_dict
        
        # Default parameters for CNNA2C (based on config_ia2c_large.ini)
        self.defaults = {
            # Network Structure
            'num_lstm': 64,
            'num_fw': 128,
            'num_ft': 32,
            
            # Learning Hyperparameters
            'batch_size': 120,
            'gamma': 0.99,
            'lr_init': 5e-4,
            'lr_decay': 'constant',
            'lr_min': 1e-5,
            
            # A2C Specific (Entropy & Value)
            'entropy_coef_init': 0.01,
            'entropy_coef_min': 0.01,
            'entropy_decay': 'constant',
            'entropy_ratio': 0.5,
            'value_coef': 0.5,
            
            # Optimization
            'max_grad_norm': 40.0,
            'rmsp_alpha': 0.99,
            'rmsp_epsilon': 1e-5,
            
            # Rewards
            'reward_norm': 2000.0,
            'reward_clip': 2.0,
        }

    _MISSING = object()

    def _get_val(self, key, fallback=_MISSING):
        if key in self.config:
            return self.config[key]
        elif key in self.defaults:
            # logging.warning(f"Config key '{key}' not found. Using default: {self.defaults[key]}")
            return self.defaults[key]
        elif fallback is not self._MISSING:
            return fallback
        else:
            raise KeyError(f"Key '{key}' not found in config or defaults.")

    def getint(self, key, fallback=_MISSING, **kwargs):
        return int(self._get_val(key, fallback=fallback))
        
    def getfloat(self, key, fallback=_MISSING, **kwargs):
        return float(self._get_val(key, fallback=fallback))
        
    def getboolean(self, key, fallback=_MISSING, **kwargs):
        val = self._get_val(key, fallback=fallback)
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        sval = str(val).strip().lower()
        if sval in ('1', 'yes', 'true', 'on'):
            return True
        if sval in ('0', 'no', 'false', 'off'):
            return False
        return False
        
    def get(self, key, fallback=_MISSING, **kwargs):
        return self._get_val(key, fallback=fallback)
        
    def __getitem__(self, key): 
        return self._get_val(key)

    def __setitem__(self, key, value):
        self.config[key] = value
        
    def __contains__(self, key): 
        return key in self.config or key in self.defaults
def make_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def get_logger(log_dir):
    logger = logging.getLogger('adversary_train')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

# --- MAIN TRAINING LOOP ---
def train_adversary(args=None):
    if args is None:
        args = parse_args()

    tf.reset_default_graph()

    default_config_path, default_model_dir, default_base_dir = _resolve_default_frozen_paths(args.agent)
    base_dir = os.path.abspath(args.base_dir) if args.base_dir else default_base_dir
    FROZEN_CONFIG_PATH = os.path.abspath(args.frozen_config) if args.frozen_config else default_config_path
    FROZEN_MODEL_DIR = os.path.abspath(args.frozen_model_dir) if args.frozen_model_dir else default_model_dir
    RESUME = not args.no_resume
    RESUME_CHECKPOINT = args.resume_checkpoint
    CHECKPOINT_INTERVAL_EP = int(args.checkpoint_interval_ep)
    total_episodes = int(args.total_episodes)

    model_dir = os.path.join(base_dir, 'model')
    make_dir(base_dir)
    make_dir(model_dir)
    logger = get_logger(base_dir)

    # Backward compatibility: older runs saved checkpoints to
    # './output_adversary/<run_name>checkpoint-*.{index,data,meta}' (missing '/').
    legacy_prefix = os.path.basename(base_dir) + 'checkpoint-'
    legacy_files = glob.glob(base_dir + 'checkpoint-*')
    migrated = 0
    for src in legacy_files:
        fname = os.path.basename(src)
        if not fname.startswith(legacy_prefix):
            continue
        dst_name = 'checkpoint-' + fname.split('checkpoint-', 1)[1]
        dst = os.path.join(model_dir, dst_name)
        if os.path.exists(dst):
            continue
        try:
            shutil.copy2(src, dst)
            migrated += 1
        except Exception as e:
            logger.warning("Failed to migrate legacy checkpoint file %s -> %s: %s", src, dst, e)
    if migrated:
        logger.info("Migrated %d legacy checkpoint files into %s", migrated, model_dir)

    logger.info(f"Reading configuration from: {FROZEN_CONFIG_PATH}")
    logger.info(f"Using frozen controller run dir: {FROZEN_MODEL_DIR}")
    logger.info(f"Adversary output dir: {base_dir}")

    if not os.path.exists(FROZEN_CONFIG_PATH):
        raise FileNotFoundError("Frozen config not found: %s" % FROZEN_CONFIG_PATH)
    if not os.path.isdir(FROZEN_MODEL_DIR):
        raise FileNotFoundError("Frozen model dir not found: %s" % FROZEN_MODEL_DIR)
    if not _has_model_checkpoint(FROZEN_MODEL_DIR):
        logger.warning("No frozen checkpoint detected under %s/model/. Load may fail.", FROZEN_MODEL_DIR)
    
    parser = configparser.ConfigParser()
    parser.read(FROZEN_CONFIG_PATH)
    
    # Config Patching
    env_config = dict(parser['ENV_CONFIG'])
    env_config['data_path'] = _script_rel('large_grid', 'data') + os.sep
    env_config['mode'] = 'train'
    env_config['seed'] = 42
    env_config['algo'] = 'adversary'
    
    env_config_wrapped = ConfigWrapper(env_config)

    logger.info("Initializing Adversarial Environment...")
    env = AdversarialLargeGridEnv(
        env_config_wrapped, 
        output_path=base_dir,
        frozen_model_dir=FROZEN_MODEL_DIR
    )
    
    set_seed(int(env_config['seed']))
    env.init_test_seeds([int(env_config['seed'])])

    macro_steps_per_episode = int(env.episode_length_sec / env.adversary_step_duration)
    if env.episode_length_sec % env.adversary_step_duration != 0:
        macro_steps_per_episode += 1

    model_config, configured_batch_size, batch_size = _prepare_adversary_model_config(
        dict(parser['MODEL_CONFIG']),
        macro_steps_per_episode
    )
    model_config_wrapped = ConfigWrapper(model_config)
    logger.info(
        "WCE training overrides: reward_norm=%.1f, reward_clip=%.1f, batch_size=%d "
        "(base_config_batch=%d, ~%.1f episodes/update)",
        ADVERSARY_REWARD_NORM,
        ADVERSARY_REWARD_CLIP,
        batch_size,
        configured_batch_size,
        float(batch_size) / float(macro_steps_per_episode)
    )

    logger.info(f"Initializing CNN Adversary...")
    adversary = CNNA2C(
        n_s=env.n_s_adversary,       
        n_a=env.n_adversary_action,  
        total_step=20000,            
        model_config=model_config_wrapped,
        seed=int(env_config['seed'])
    )

    # --- TENSORBOARD SETUP ---
    summary_writer = tf.summary.FileWriter(base_dir, adversary.sess.graph)

    global_step = 0
    expected_total_macro_steps = total_episodes * macro_steps_per_episode
    model_dir_slash = model_dir if model_dir.endswith('/') else model_dir + '/'
    checkpoint_meta_path = os.path.join(model_dir, 'checkpoint_meta.json')

    def _save_checkpoint(ep, gstep):
        adversary.save(model_dir_slash, global_step=ep)
        meta = {'checkpoint_episode': int(ep), 'global_step': int(gstep)}
        try:
            with open(checkpoint_meta_path, 'w') as fh:
                json.dump(meta, fh)
        except Exception as e:
            logger.warning("Failed to write checkpoint metadata: %s", e)
        logger.info("Saved checkpoint at episode=%d (global_step=%d) to %s", ep, gstep, model_dir)

    start_episode = 0
    if RESUME:
        checkpoint_arg = None
        if str(RESUME_CHECKPOINT).strip().lower() not in ['', 'latest', 'auto']:
            checkpoint_arg = int(RESUME_CHECKPOINT)
        loaded = adversary.load(model_dir_slash, checkpoint=checkpoint_arg)
        if loaded:
            loaded_ep = int(getattr(adversary, 'loaded_checkpoint_step', 0))
            start_episode = loaded_ep + 1
            global_step = start_episode * macro_steps_per_episode
            if os.path.exists(checkpoint_meta_path):
                try:
                    with open(checkpoint_meta_path, 'r') as fh:
                        meta = json.load(fh)
                    if int(meta.get('checkpoint_episode', -1)) == loaded_ep:
                        global_step = int(meta.get('global_step', global_step))
                except Exception as e:
                    logger.warning("Failed to read checkpoint metadata: %s", e)
            if hasattr(adversary, 'set_train_step'):
                adversary.set_train_step(global_step)
            logger.info(
                "Resumed WCE from checkpoint episode=%d, start_episode=%d, global_step=%d",
                loaded_ep, start_episode, global_step
            )
        else:
            logger.info("Resume requested but no checkpoint found in %s. Starting fresh.", model_dir)

    if start_episode >= total_episodes:
        logger.info("No training needed: start_episode=%d >= total_episodes=%d", start_episode, total_episodes)
        env.terminate()
        summary_writer.flush()
        summary_writer.close()
        return

    logger.info(
        "WCE setup: scenarios=%d, episode_length_sec=%d, adversary_step_duration=%d, "
        "macro_steps_per_episode~%d, total_episodes=%d, expected_total_macro_steps~%d, "
        "batch_size=%d, checkpoint_interval=%d, resume=%r, model_dir=%s",
        len(env.group_names), env.episode_length_sec, env.adversary_step_duration,
        macro_steps_per_episode, total_episodes, expected_total_macro_steps,
        batch_size, CHECKPOINT_INTERVAL_EP, RESUME, model_dir
    )

    stall_watchdog_sec = int(env_config_wrapped.getint('stall_watchdog_sec', fallback=180))
    stall_watchdog_poll_sec = int(env_config_wrapped.getint('stall_watchdog_poll_sec', fallback=10))
    watchdog_dump_path = os.path.join(base_dir, 'stall_watchdog_stacks.log')
    signal_dump_path = os.path.join(base_dir, 'manual_signal_stacks.log')
    watchdog_state = {
        'running': True,
        'last_touch': time.time(),
        'label': 'init',
        'last_dump': 0.0
    }
    watchdog_lock = threading.Lock()

    def _touch_watchdog(label):
        with watchdog_lock:
            watchdog_state['last_touch'] = time.time()
            watchdog_state['label'] = label

    def _watchdog_loop():
        while watchdog_state['running']:
            time.sleep(max(1, stall_watchdog_poll_sec))
            now = time.time()
            with watchdog_lock:
                last_touch = watchdog_state['last_touch']
                label = watchdog_state['label']
                last_dump = watchdog_state['last_dump']
            stalled_for = now - last_touch
            if stalled_for < stall_watchdog_sec:
                continue
            if (now - last_dump) < max(30, stall_watchdog_sec // 2):
                continue
            with watchdog_lock:
                watchdog_state['last_dump'] = now
            logger.warning(
                "Watchdog: no WCE progress for %.1fs (stage=%s). Dumping stacks...",
                stalled_for, label
            )
            try:
                with open(watchdog_dump_path, 'a') as fh:
                    fh.write(
                        '\n=== %s stalled_for=%.1fs stage=%s ===\n'
                        % (time.strftime('%Y-%m-%d %H:%M:%S'), stalled_for, label)
                    )
                    faulthandler.dump_traceback(file=fh, all_threads=True)
            except Exception as e:
                logger.warning("Watchdog: failed to write stack dump: %s", e)

    signal_dump_fh = None
    signal_dump_registered = False
    if hasattr(signal, 'SIGUSR1'):
        try:
            signal_dump_fh = open(signal_dump_path, 'a')
            faulthandler.register(signal.SIGUSR1, file=signal_dump_fh, all_threads=True)
            signal_dump_registered = True
            logger.info("Manual stack dump enabled: `kill -USR1 %d` -> %s",
                        os.getpid(), signal_dump_path)
        except Exception as e:
            logger.warning("Failed to enable SIGUSR1 stack dumps: %s", e)
            if signal_dump_fh is not None:
                try:
                    signal_dump_fh.close()
                except Exception:
                    pass
                signal_dump_fh = None

    watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
    watchdog_thread.start()

    logger.info("Start Training from episode %d ...", start_episode)

    try:
        for ep in range(start_episode, total_episodes):
            _touch_watchdog('episode_reset')
            obs = env.reset()
            done = False
            ep_reward = 0
            ep_steps = 0
            
            # Track actions taken this episode
            ep_actions = []

            while not done:
                _touch_watchdog('forward')
                action_logits, value = adversary.forward(obs, done, out_type='pv')
                action_logits = np.squeeze(action_logits)

                _touch_watchdog('env_step')
                scenario_weights = env._normalize_action_weights(action_logits)
                ep_actions.append(int(np.argmax(scenario_weights)))
                next_obs, reward, done, info = env.step(action_logits)

                _touch_watchdog('add_transition')
                adversary.add_transition(obs, action_logits, reward, value, done)
                
                if adversary.trans_buffer.size >= batch_size:
                    if done:
                        bootstrap_R = 0
                    else:
                        _, next_value = adversary.forward(next_obs, done=False, out_type='pv')
                        bootstrap_R = np.squeeze(next_value)
                    _touch_watchdog('backward')
                    adversary.backward(
                        R=bootstrap_R,
                        summary_writer=summary_writer,
                        global_step=global_step
                    )
                
                obs = next_obs
                ep_reward += reward
                ep_steps += 1
                global_step += 1
                
            if (ep == total_episodes - 1) and adversary.trans_buffer.size:
                _touch_watchdog('backward_final_flush')
                adversary.backward(R=0, summary_writer=summary_writer, global_step=global_step)

            logger.info("Episode %d: Steps=%d, Reward=%.2f", ep, ep_steps, ep_reward)
            
            log_episode_stats(summary_writer, ep, ep_reward, ep_steps, ep_actions, env.n_adversary_action)

            if ((ep + 1) % CHECKPOINT_INTERVAL_EP == 0) or (ep == total_episodes - 1):
                _touch_watchdog('checkpoint_save')
                _save_checkpoint(ep, global_step)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Saving emergency checkpoint before exit.")
        if 'ep' in locals():
            _save_checkpoint(ep, global_step)
        raise
    except Exception:
        logger.exception("WCE training crashed. Saving emergency checkpoint.")
        if 'ep' in locals():
            _save_checkpoint(ep, global_step)
        raise
    finally:
        watchdog_state['running'] = False
        watchdog_thread.join(timeout=1.0)
        if signal_dump_registered and hasattr(signal, 'SIGUSR1'):
            try:
                faulthandler.unregister(signal.SIGUSR1)
            except Exception:
                pass
        if signal_dump_fh is not None:
            try:
                signal_dump_fh.close()
            except Exception:
                pass
        env.terminate()
        summary_writer.flush()
        summary_writer.close()
        logger.info("Training Finished.")

if __name__ == '__main__':
    train_adversary()
