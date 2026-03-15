import os
import argparse
import logging
import configparser
import numpy as np
from tf_compat import tf
import json
import time
import threading
import faulthandler
import signal
import shutil
from utils import init_dir, init_log
from envs.coevolution_env import CoevolutionLargeGridEnv
# Updated imports to include all agent types
from agents.models import A2C, IA2C, MA2C, IQL, CNNA2C, PPO


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _script_rel(*parts):
    return os.path.join(SCRIPT_DIR, *parts)


def _resolve_default_coev_paths(agent):
    agent = str(agent).strip().lower()
    if agent == 'ia2c':
        return {
            'base_dir': _script_rel('output_coevolution', 'ia2c_large'),
            'adversary_ckpt_dir': _script_rel('output_adversary', 'ia2c_large', 'model'),
            'traffic_ckpt_dir': _script_rel('runs', 'ia2c_large', 'model'),
            'config_file': _script_rel('config', 'config_ia2c_large.ini'),
        }
    if agent == 'iqll':
        return {
            'base_dir': _script_rel('output_coevolution', 'iqll_large'),
            'adversary_ckpt_dir': _script_rel('output_adversary', 'iqll_large', 'model'),
            'traffic_ckpt_dir': _script_rel('runs', 'iqll_large', 'model'),
            'config_file': _script_rel('config', 'config_iqll_large.ini'),
        }
    if agent == 'ppo':
        return {
            'base_dir': _script_rel('output_coevolution', 'ppo_large'),
            'adversary_ckpt_dir': _script_rel('output_adversary', 'ppo_large', 'model'),
            'traffic_ckpt_dir': _script_rel('runs', 'ppo_large', 'model'),
            'config_file': _script_rel('config', 'config_ppo_large.ini'),
        }
    return {
        'base_dir': _script_rel('output_coevolution', 'ma2c_large'),
        'adversary_ckpt_dir': _script_rel('output_adversary', 'ma2c_large', 'model'),
        'traffic_ckpt_dir': _script_rel('runs', 'ma2c_large', 'model'),
        'config_file': _script_rel('config', 'config_ma2c_large.ini'),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run co-evolution training for the large-grid controller and WCE.'
    )
    parser.add_argument(
        '--agent',
        type=str,
        default='ma2c',
        choices=['ma2c', 'ia2c', 'iqll', 'ppo'],
        help='Traffic controller family used for config selection and training.'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Co-evolution output directory.'
    )
    parser.add_argument(
        '--total-episodes',
        type=int,
        default=1000,
        help='Number of co-evolution episodes.'
    )
    parser.add_argument(
        '--adv-interval-sec',
        type=int,
        default=600,
        help='Adversary action interval in simulation seconds.'
    )
    parser.add_argument(
        '--checkpoint-interval-ep',
        type=int,
        default=10,
        help='Save co-evolution checkpoints every N episodes.'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from prior co-evolution checkpoints in base-dir.'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from prior co-evolution checkpoints (default behavior).'
    )
    args = parser.parse_args()
    if args.resume and args.no_resume:
        parser.error('--resume and --no-resume are mutually exclusive.')
    return args

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

def train_coevolution(args=None):
    if args is None:
        args = parse_args()
    tf.reset_default_graph()
    default_paths = _resolve_default_coev_paths(args.agent)
    base_dir = os.path.abspath(args.base_dir) if args.base_dir else default_paths['base_dir']
    dirs = init_dir(base_dir, pathes=['log', 'data', 'model_traffic', 'model_adversary'])
    init_log(dirs['log'])
    
    # --- 1. CONFIGURATION ---
    ADVERSARY_CHECKPOINT_DIR = default_paths['adversary_ckpt_dir']
    TRAFFIC_CHECKPOINT_DIR = default_paths['traffic_ckpt_dir']
    CONFIG_FILE = default_paths['config_file']
    TOTAL_EPISODES = int(args.total_episodes)
    ADV_ACTION_INTERVAL = int(args.adv_interval_sec)
    CHECKPOINT_INTERVAL_EP = int(args.checkpoint_interval_ep)
    RESUME_COEVOLUTION = bool(args.resume) and not bool(args.no_resume)
    # Force resume from known good co-evolution checkpoints.
    # Override without code edits:
    #   COEV_FORCE_RESUME=1 COEV_FORCE_TRAFFIC_CKPT=... COEV_FORCE_ADV_CKPT=...
    FORCE_RESUME_FROM_CHECKPOINT = os.environ.get('COEV_FORCE_RESUME', '0') == '1'
    FORCE_TRAFFIC_CHECKPOINT_STEP = int(os.environ.get('COEV_FORCE_TRAFFIC_CKPT', '1172520'))
    FORCE_ADVERSARY_CHECKPOINT_EP = int(os.environ.get('COEV_FORCE_ADV_CKPT', '129'))

    logging.info(f"Loading Config: {CONFIG_FILE}")
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    
    env_config_dict = dict(config['ENV_CONFIG'])
    env_config_dict['data_path'] = _script_rel('large_grid', 'data') + os.sep
    env_config_dict['mode'] = 'train'
    # Ensure runtime agent always follows the CLI selection.
    env_config_dict['agent'] = str(args.agent).strip().lower()
    env_config_wrapped = ConfigWrapper(env_config_dict)
    
    # --- 2. INIT ENVIRONMENT ---
    # Will now pass 'dummy_path' internally to avoid ValueError
    env = CoevolutionLargeGridEnv(env_config_wrapped, port=0, output_path=dirs['data'])

    # Co-evolution traffic learner must be created in training mode (non-zero total_step)
    # so that replay/on-policy buffers and schedulers are initialized.
    base_total_step = int(float(config.get('TRAIN_CONFIG', 'total_step')))
    traffic_steps_per_episode = int(np.ceil(float(env.episode_length_sec) /
                                            float(env.control_interval_sec)))
    traffic_total_step = base_total_step + (TOTAL_EPISODES * traffic_steps_per_episode)

    macro_steps_per_episode = int(np.ceil(float(env.episode_length_sec) /
                                          float(ADV_ACTION_INTERVAL)))
    adversary_total_step = max(100000, TOTAL_EPISODES * macro_steps_per_episode)
    
    # --- 3. INIT AGENTS ---
    # A) Adversary (Worst-Case Estimator) in its own graph/session
    adv_model_config_dict, configured_adv_batch_size, batch_size = _prepare_adversary_model_config(
        dict(config['MODEL_CONFIG']),
        macro_steps_per_episode
    )
    adv_model_config = ConfigWrapper(adv_model_config_dict)
    logging.info(
        "WCE training overrides: reward_norm=%.1f, reward_clip=%.1f, batch_size=%d "
        "(base_config_batch=%d, ~%.1f episodes/update)",
        ADVERSARY_REWARD_NORM,
        ADVERSARY_REWARD_CLIP,
        batch_size,
        configured_adv_batch_size,
        float(batch_size) / float(macro_steps_per_episode)
    )
    adversary_graph = tf.Graph()
    with adversary_graph.as_default():
        adversary = CNNA2C(
            n_s=env.n_s_adversary,
            n_a=env.n_adversary_action,
            total_step=adversary_total_step,
            model_config=adv_model_config,
            seed=42
        )
    adversary.graph = adversary_graph
    
    # B) Traffic Controller (Robust Agent)
    # Modified to dynamically load agent based on config 'agent' type
    traffic_model_config = ConfigWrapper(dict(config['MODEL_CONFIG']))
    agent_type = env_config_dict.get('agent', 'ma2c')  # Default to ma2c if not specified
    
    logging.info(f"Initializing Traffic Agent of type: {agent_type}")

    # B) Traffic Controller in its own graph/session
    traffic_graph = tf.Graph()
    with traffic_graph.as_default():
        if agent_type == 'a2c':
            traffic_agent = A2C(env.n_s, env.n_a, traffic_total_step, traffic_model_config, seed=42)
        elif agent_type == 'ia2c':
            traffic_agent = IA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, traffic_total_step, traffic_model_config, seed=42)
        elif agent_type == 'ma2c':
            traffic_agent = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, traffic_total_step, traffic_model_config, seed=42)
        elif agent_type == 'ppo':
            traffic_agent = PPO(env.n_s_ls, env.n_a_ls, env.n_w_ls, traffic_total_step, traffic_model_config, seed=42)
        elif agent_type == 'iqld':
            traffic_agent = IQL(env.n_s_ls, env.n_a_ls, env.n_w_ls, traffic_total_step, traffic_model_config, seed=42, model_type='dqn')
        elif agent_type == 'iqll':
            # model_type='lr' triggers LRQPolicy in agents/models.py
            traffic_agent = IQL(env.n_s_ls, env.n_a_ls, env.n_w_ls, traffic_total_step, traffic_model_config, seed=42, model_type='lr')
        else:
            # Fallback
            logging.warning(f"Unknown agent type '{agent_type}', falling back to MA2C")
            traffic_agent = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, traffic_total_step, traffic_model_config, seed=42)
    traffic_agent.graph = traffic_graph

    
    # --- 4. LOAD PRE-TRAINED WEIGHTS ---
    logging.info(">>> Loading Pre-trained Weights...")
    with adversary.sess.graph.as_default():
        if adversary.load(ADVERSARY_CHECKPOINT_DIR):
            logging.info("Loaded Adversary.")
        else:
            raise RuntimeError("Could not load adversary checkpoint from %s" % ADVERSARY_CHECKPOINT_DIR)
    
    # Load traffic agent weights
    with traffic_agent.sess.graph.as_default():
        if traffic_agent.load(TRAFFIC_CHECKPOINT_DIR):
            logging.info(f"Loaded Traffic Agent ({agent_type}).")
        else:
            raise RuntimeError("Could not load traffic checkpoint from %s" % TRAFFIC_CHECKPOINT_DIR)

    summary_writer = tf.summary.FileWriter(dirs['log'], traffic_agent.sess.graph)

    def _log_scalar(tag, value, step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=float(value))
        summary_writer.add_summary(summary, int(step))

    # Co-evolution watchdog: dump stacks and exit on long stalls so runs can be resumed
    # from checkpoints instead of hanging forever.
    configured_watchdog_sec = int(env_config_wrapped.getint('stall_watchdog_sec', fallback=180))
    # Co-evolution segments can include long traffic backward calls; keep timeout generous by default.
    watchdog_timeout_sec = int(os.environ.get(
        'COEV_WATCHDOG_TIMEOUT_SEC',
        str(max(600, configured_watchdog_sec))
    ))
    watchdog_poll_sec = int(env_config_wrapped.getint('stall_watchdog_poll_sec', fallback=10))
    watchdog_timeout_sec = max(180, watchdog_timeout_sec)
    watchdog_dump_path = os.path.join(dirs['data'], 'coev_stall_watchdog_stacks.log')
    watchdog_lock = threading.Lock()
    watchdog_state = {'last_touch': time.time(), 'label': 'init', 'running': True}

    def _watchdog_touch(label):
        with watchdog_lock:
            watchdog_state['last_touch'] = time.time()
            watchdog_state['label'] = label

    def _watchdog_loop():
        while True:
            time.sleep(max(1, watchdog_poll_sec))
            with watchdog_lock:
                running = watchdog_state['running']
                last_touch = watchdog_state['last_touch']
                label = watchdog_state['label']
            if not running:
                return
            stalled_for = time.time() - last_touch
            if stalled_for < watchdog_timeout_sec:
                continue
            try:
                with open(watchdog_dump_path, 'a') as fh:
                    fh.write(
                        '\n=== %s stalled_for=%.1fs stage=%s ===\n'
                        % (time.strftime('%Y-%m-%d %H:%M:%S'), stalled_for, label)
                    )
                    faulthandler.dump_traceback(file=fh, all_threads=True)
                    fh.flush()
            except Exception as e:
                logging.error("Watchdog: failed to write stack dump: %s", e)
            logging.error(
                "Watchdog: co-evolution stalled for %.1fs at stage '%s'. "
                "Exiting so training can be resumed from checkpoints.",
                stalled_for, label
            )
            os._exit(2)

    watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
    watchdog_thread.start()

    # Avoid silent hangs when disk is near exhaustion.
    min_free_gb = 2.0

    def _check_disk_space():
        free_gb = shutil.disk_usage(base_dir).free / float(1024 ** 3)
        if free_gb < min_free_gb:
            raise RuntimeError(
                "Low disk space: %.2f GB free (minimum %.2f GB required). "
                "Free space and resume from checkpoint."
                % (free_gb, min_free_gb)
            )

    # --- 5. CO-EVOLUTION LOOP ---
    # Sync env duration
    env.adversary_step_duration = ADV_ACTION_INTERVAL
    macro_steps_per_episode = int(env.episode_length_sec / env.adversary_step_duration)
    if env.episode_length_sec % env.adversary_step_duration != 0:
        macro_steps_per_episode += 1

    traffic_ckpt_dir = dirs['model_traffic']
    adversary_ckpt_dir = dirs['model_adversary']
    traffic_ckpt_dir_slash = traffic_ckpt_dir if traffic_ckpt_dir.endswith('/') else traffic_ckpt_dir + '/'
    adversary_ckpt_dir_slash = adversary_ckpt_dir if adversary_ckpt_dir.endswith('/') else adversary_ckpt_dir + '/'
    coev_meta_path = os.path.join(base_dir, 'coev_meta.json')

    def _has_checkpoint(model_dir):
        if not os.path.isdir(model_dir):
            return False
        for file_name in os.listdir(model_dir):
            if file_name.startswith('checkpoint-') and file_name.endswith('.index'):
                return True
        return False

    # Initialize counters from loaded pre-trained checkpoints.
    global_traffic_step = int(getattr(traffic_agent, 'loaded_checkpoint_step', 0))
    loaded_adv_ep = int(getattr(adversary, 'loaded_checkpoint_step', 0))
    global_adv_step = max(0, loaded_adv_ep * macro_steps_per_episode)
    start_episode = 0

    # Keep schedulers consistent with loaded checkpoints.
    if hasattr(traffic_agent, 'set_train_step'):
        traffic_agent.set_train_step(global_traffic_step)
    if hasattr(adversary, 'set_train_step'):
        adversary.set_train_step(global_adv_step)

    # Resume previous co-evolution run if checkpoints exist.
    if RESUME_COEVOLUTION:
        traffic_resumed = False
        adversary_resumed = False

        if FORCE_RESUME_FROM_CHECKPOINT:
            with traffic_agent.sess.graph.as_default():
                traffic_resumed = traffic_agent.load(
                    traffic_ckpt_dir_slash,
                    checkpoint=FORCE_TRAFFIC_CHECKPOINT_STEP
                )
            with adversary.sess.graph.as_default():
                adversary_resumed = adversary.load(
                    adversary_ckpt_dir_slash,
                    checkpoint=FORCE_ADVERSARY_CHECKPOINT_EP
                )
            if not traffic_resumed:
                raise RuntimeError(
                    "Forced traffic checkpoint load failed: %s checkpoint-%d"
                    % (traffic_ckpt_dir_slash, FORCE_TRAFFIC_CHECKPOINT_STEP)
                )
            if not adversary_resumed:
                raise RuntimeError(
                    "Forced adversary checkpoint load failed: %s checkpoint-%d"
                    % (adversary_ckpt_dir_slash, FORCE_ADVERSARY_CHECKPOINT_EP)
                )

            global_traffic_step = int(FORCE_TRAFFIC_CHECKPOINT_STEP)
            start_episode = int(FORCE_ADVERSARY_CHECKPOINT_EP) + 1
            global_adv_step = int(
                (loaded_adv_ep * macro_steps_per_episode) +
                ((FORCE_ADVERSARY_CHECKPOINT_EP + 1) * macro_steps_per_episode)
            )

            # If metadata matches forced checkpoint, prefer exact persisted counters.
            if os.path.exists(coev_meta_path):
                try:
                    with open(coev_meta_path, 'r') as fh:
                        meta = json.load(fh)
                    if int(meta.get('episode', -1)) == int(FORCE_ADVERSARY_CHECKPOINT_EP):
                        meta_traffic_step = int(meta.get('global_traffic_step', global_traffic_step))
                        if meta_traffic_step == int(FORCE_TRAFFIC_CHECKPOINT_STEP):
                            global_adv_step = int(meta.get('global_adv_step', global_adv_step))
                except Exception as e:
                    logging.warning("Failed to read co-evolution metadata for forced resume: %s", e)

            if hasattr(traffic_agent, 'set_train_step'):
                traffic_agent.set_train_step(global_traffic_step)
            if hasattr(adversary, 'set_train_step'):
                adversary.set_train_step(global_adv_step)
            logging.info(
                "Force-resumed co-evolution at episode=%d, traffic_step=%d, adversary_step=%d "
                "(traffic_ckpt=%d, adversary_ckpt=%d)",
                start_episode, global_traffic_step, global_adv_step,
                FORCE_TRAFFIC_CHECKPOINT_STEP, FORCE_ADVERSARY_CHECKPOINT_EP
            )
        else:
            if _has_checkpoint(traffic_ckpt_dir_slash):
                with traffic_agent.sess.graph.as_default():
                    traffic_resumed = traffic_agent.load(traffic_ckpt_dir_slash)
            else:
                logging.info("No co-evolution traffic checkpoint found in %s; starting from pre-trained traffic model.",
                             traffic_ckpt_dir_slash)

            if _has_checkpoint(adversary_ckpt_dir_slash):
                with adversary.sess.graph.as_default():
                    adversary_resumed = adversary.load(adversary_ckpt_dir_slash)
            else:
                logging.info("No co-evolution adversary checkpoint found in %s; starting from pre-trained adversary model.",
                             adversary_ckpt_dir_slash)

            if traffic_resumed or adversary_resumed:
                if traffic_resumed:
                    global_traffic_step = int(getattr(traffic_agent, 'loaded_checkpoint_step', global_traffic_step))
                if adversary_resumed:
                    loaded_ep = int(getattr(adversary, 'loaded_checkpoint_step', 0))
                    start_episode = max(start_episode, loaded_ep + 1)
                    global_adv_step = int(
                        (loaded_adv_ep * macro_steps_per_episode) +
                        ((loaded_ep + 1) * macro_steps_per_episode)
                    )
                if os.path.exists(coev_meta_path):
                    try:
                        with open(coev_meta_path, 'r') as fh:
                            meta = json.load(fh)
                        start_episode = max(start_episode, int(meta.get('episode', -1)) + 1)
                        global_traffic_step = int(meta.get('global_traffic_step', global_traffic_step))
                        global_adv_step = int(meta.get('global_adv_step', global_adv_step))
                    except Exception as e:
                        logging.warning("Failed to read co-evolution metadata: %s", e)

                if hasattr(traffic_agent, 'set_train_step'):
                    traffic_agent.set_train_step(global_traffic_step)
                if hasattr(adversary, 'set_train_step'):
                    adversary.set_train_step(global_adv_step)
                logging.info(
                    "Resumed co-evolution at episode=%d, traffic_step=%d, adversary_step=%d",
                    start_episode, global_traffic_step, global_adv_step
                )

    if start_episode >= TOTAL_EPISODES:
        logging.info("No co-evolution training needed: start_episode=%d >= total=%d",
                     start_episode, TOTAL_EPISODES)
        with watchdog_lock:
            watchdog_state['running'] = False
        watchdog_thread.join(timeout=2)
        env.terminate()
        summary_writer.flush()
        summary_writer.close()
        return

    logging.info(
        "Starting Co-Evolution Training: start_episode=%d total_episodes=%d "
        "macro_steps_per_episode=%d wce_batch_size=%d checkpoint_interval=%d",
        start_episode, TOTAL_EPISODES, macro_steps_per_episode, batch_size, CHECKPOINT_INTERVAL_EP
    )
    _watchdog_touch('coev:loop_start')

    segment_summary_flush_every = 5
    segment_since_flush = 0

    for ep in range(start_episode, TOTAL_EPISODES):
        _check_disk_space()
        _watchdog_touch('coev:episode_reset')
        # Reset
        adv_obs = env.reset()  # Adversarial env reset already returns adversary observation.
        traffic_agent.reset()
        
        done = False
        episode_traffic_reward = 0.0
        episode_wce_reward = 0.0
        episode_macro_steps = 0
        
        logging.info(f"Episode {ep} Start.")
        
        while not done:
            _watchdog_touch('coev:adversary_forward')
            # 1. Adversary Action (Weights)
            adv_forward_start = time.time()
            with adversary.sess.graph.as_default():
                adv_action_logits, adv_value = adversary.forward(adv_obs, done=False, out_type='pv')
            adv_forward_elapsed = time.time() - adv_forward_start
            if adv_forward_elapsed > 30:
                logging.warning("Slow adversary forward: %.2fs at adv_step=%d", adv_forward_elapsed, global_adv_step)
            action_to_store = np.squeeze(adv_action_logits)
            value_to_store = np.squeeze(adv_value)
            
            # 2. Environment Step (Runs Inner Loop for 10 mins)
            #    We pass the traffic_agent so the env can drive the inner simulation
            _watchdog_touch('coev:env_step')
            env_step_start = time.time()
            next_adv_obs, adv_reward, done, info = env.step(
                adversary_action=adv_action_logits,
                traffic_agent=traffic_agent,
                summary_writer=summary_writer,
                global_traffic_step=global_traffic_step,
                watchdog_touch=_watchdog_touch
            )
            _watchdog_touch('coev:env_step_done')
            env_step_elapsed = time.time() - env_step_start
            if env_step_elapsed > 120:
                logging.warning("Slow coevolution env.step: %.2fs at adv_step=%d", env_step_elapsed, global_adv_step)
            
            # Sync global step from inner loop
            global_traffic_step = info['global_traffic_step']
            segment_traffic_reward = float(info.get('segment_traffic_reward', -100.0 * adv_reward))
            segment_wce_reward = float(info.get('segment_wce_reward', adv_reward))
            segment_control_steps = int(info.get('segment_control_steps', 0))
            episode_traffic_reward += segment_traffic_reward
            episode_wce_reward += segment_wce_reward
            episode_macro_steps += 1
            if segment_control_steps > 0:
                seg_mean_traffic_reward = segment_traffic_reward / float(segment_control_steps)
            else:
                seg_mean_traffic_reward = 0.0

            _log_scalar('Segment/Traffic_Total_Reward', segment_traffic_reward, global_traffic_step)
            _log_scalar('Segment/WCE_Total_Reward', segment_wce_reward, global_traffic_step)
            _log_scalar('Segment/Traffic_Mean_Reward_Per_ControlStep',
                        seg_mean_traffic_reward, global_traffic_step)
            segment_since_flush += 1
            if segment_since_flush >= segment_summary_flush_every:
                summary_writer.flush()
                segment_since_flush = 0
            
            # 3. Train Adversary (Once per 10 mins)
            adversary.add_transition(adv_obs, action_to_store, adv_reward, value_to_store, done)
            if adversary.trans_buffer.size >= batch_size:
                if done:
                    R = 0
                else:
                    # If not done, we bootstrap with the value of the NEXT state
                    with adversary.sess.graph.as_default():
                        _, next_val = adversary.forward(next_adv_obs, done=False, out_type='pv')
                    R = np.squeeze(next_val)
                
                # Perform update
                _watchdog_touch('coev:adversary_backward')
                adv_backward_start = time.time()
                adversary.backward(R=R, summary_writer=summary_writer, global_step=global_adv_step)
                _watchdog_touch('coev:adversary_backward_done')
                adv_backward_elapsed = time.time() - adv_backward_start
                if adv_backward_elapsed > 60:
                    logging.warning("Slow adversary backward: %.2fs at adv_step=%d", adv_backward_elapsed, global_adv_step)
            
            # Update Adversary State
            adv_obs = next_adv_obs
            global_adv_step += 1
            
            logging.info(
                f"   Adv Step {global_adv_step} | WCE Reward: {segment_wce_reward:.4f} | "
                f"Traffic Segment Reward: {segment_traffic_reward:.2f}"
            )

        if (ep == TOTAL_EPISODES - 1) and adversary.trans_buffer.size:
            _watchdog_touch('coev:adversary_backward_final_flush')
            adv_backward_start = time.time()
            adversary.backward(R=0, summary_writer=summary_writer, global_step=global_adv_step)
            _watchdog_touch('coev:adversary_backward_final_flush_done')
            adv_backward_elapsed = time.time() - adv_backward_start
            if adv_backward_elapsed > 60:
                logging.warning("Slow final adversary backward: %.2fs at adv_step=%d", adv_backward_elapsed, global_adv_step)

        _log_scalar('Episode/Traffic_Total_Reward', episode_traffic_reward, ep)
        _log_scalar('Episode/WCE_Total_Reward', episode_wce_reward, ep)
        _log_scalar('Episode/Macro_Steps', episode_macro_steps, ep)
        summary_writer.flush()
        segment_since_flush = 0

        logging.info(
            "Episode %d Finished | Traffic_Total_Reward=%.2f | WCE_Total_Reward=%.4f | macro_steps=%d",
            ep, episode_traffic_reward, episode_wce_reward, episode_macro_steps
        )

        if ((ep + 1) % CHECKPOINT_INTERVAL_EP == 0) or (ep == TOTAL_EPISODES - 1):
            _watchdog_touch('coev:checkpoint_save')
            traffic_agent.save(traffic_ckpt_dir_slash, global_traffic_step)
            adversary.save(adversary_ckpt_dir_slash, ep)
            try:
                with open(coev_meta_path, 'w') as fh:
                    json.dump({
                        'episode': int(ep),
                        'global_traffic_step': int(global_traffic_step),
                        'global_adv_step': int(global_adv_step)
                    }, fh)
            except Exception as e:
                logging.warning("Failed to write co-evolution metadata: %s", e)
            logging.info(
                "Saved co-evolution checkpoints at episode=%d (traffic_step=%d, adversary_step=%d)",
                ep, global_traffic_step, global_adv_step
            )
            _watchdog_touch('coev:checkpoint_saved')

    with watchdog_lock:
        watchdog_state['running'] = False
    watchdog_thread.join(timeout=2)
    env.terminate()
    summary_writer.flush()
    summary_writer.close()

if __name__ == '__main__':
    train_coevolution()
