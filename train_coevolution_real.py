import os
import argparse
import logging
import configparser
import numpy as np
from tf_compat import tf
import sys
import time

sys.path.append(os.getcwd())

from utils import init_dir, init_log
from train_adversary import ConfigWrapper
from envs.coevolution_real_net_env import CoevolutionRealNetEnv
from agents.models import A2C, IA2C, MA2C, IQL, GCNA2C, PPO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _script_rel(*parts):
    return os.path.join(SCRIPT_DIR, *parts)


def _normalize_agent_name(agent):
    normalized = str(agent).strip().lower()
    if normalized in ('iql-lr', 'iql_lr', 'iqll'):
        return 'iqll'
    return normalized


def _resolve_default_paths(agent):
    if agent == 'ppo':
        return {
            'config_path': _script_rel('config', 'config_ppo_real.ini'),
            'frozen_model_dir': _script_rel('runs', 'ppo_real'),
            'wce_dir': _script_rel('output_adversary_monaco', 'ppo_real'),
            'base_dir': _script_rel('output_coevolution_real', 'ppo'),
        }
    if agent == 'ia2c':
        return {
            'config_path': _script_rel('config', 'config_ia2c_real.ini'),
            'frozen_model_dir': _script_rel('runs', 'ia2c_real'),
            'wce_dir': _script_rel('output_adversary_monaco', 'ia2c_real'),
            'base_dir': _script_rel('output_coevolution_real', 'ia2c'),
        }
    if agent == 'ma2c':
        return {
            'config_path': _script_rel('config', 'config_ma2c_real.ini'),
            'frozen_model_dir': _script_rel('runs', 'ma2c_real'),
            'wce_dir': _script_rel('output_adversary_monaco', 'ma2c_real'),
            'base_dir': _script_rel('output_coevolution_real', 'ma2c'),
        }
    if agent == 'iqll':
        return {
            'config_path': _script_rel('config', 'config_iqll_real.ini'),
            'frozen_model_dir': _script_rel('runs', 'iqll_real'),
            'wce_dir': _script_rel('output_adversary_monaco', 'iqll_real'),
            'base_dir': _script_rel('output_coevolution_real', 'iql-lr'),
        }
    raise ValueError(
        "Unsupported --agent '%s'. Use ppo, ia2c, ma2c, or iql-lr/iqll." % agent
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run co-evolution training on the real network.'
    )
    parser.add_argument(
        '--agent',
        type=str,
        default=None,
        choices=['ppo', 'ia2c', 'ma2c', 'iqll', 'iql-lr', 'iql_lr'],
        help='Traffic controller family used for default path selection.',
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Co-evolution output directory. Defaults to output_coevolution_real/<agent>.',
    )
    parser.add_argument(
        '--frozen-config',
        type=str,
        default=None,
        help='Path to frozen controller config .ini.',
    )
    parser.add_argument(
        '--frozen-model-dir',
        type=str,
        default=None,
        help='Path to frozen controller directory or model checkpoint directory.',
    )
    parser.add_argument(
        '--wce-dir',
        type=str,
        default=None,
        help='Path to adversary checkpoint directory (run dir or model dir).',
    )
    parser.add_argument(
        '--total-episodes',
        type=int,
        default=1000,
        help='Number of co-evolution episodes.',
    )
    parser.add_argument(
        '--checkpoint-interval-ep',
        type=int,
        default=10,
        help='Save co-evolution checkpoints every N episodes.',
    )
    return parser.parse_args()


def _resolve_checkpoint_dir(path):
    resolved = os.path.abspath(path)
    if os.path.isfile(resolved):
        resolved = os.path.dirname(resolved)
    if os.path.basename(os.path.normpath(resolved)).startswith('checkpoint'):
        resolved = os.path.dirname(resolved)
    if os.path.isdir(os.path.join(resolved, 'model')):
        resolved = os.path.join(resolved, 'model')
    return resolved


def _infer_agent_from_config(config_path, fallback='ia2c'):
    if not os.path.exists(config_path):
        return fallback
    parser = configparser.ConfigParser()
    parser.read(config_path)
    if 'ENV_CONFIG' not in parser:
        return fallback
    inferred = _normalize_agent_name(parser['ENV_CONFIG'].get('agent', fallback))
    if inferred in ('ppo', 'ia2c', 'ma2c', 'iqll'):
        return inferred
    return fallback


def load_traffic_agent(env, env_config, traffic_model_config, model_dir, total_step):
    agent_type = _normalize_agent_name(env_config.get('agent', 'ia2c'))
    logging.info(f"Initializing Traffic Agent ({agent_type}) for {env.node_names}...")

    if agent_type == 'a2c':
        traffic_agent = A2C(env.n_s, env.n_a, total_step, traffic_model_config, seed=42)
    elif agent_type == 'ia2c':
        traffic_agent = IA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, total_step, traffic_model_config, seed=42)
    elif agent_type == 'ma2c':
        traffic_agent = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, total_step, traffic_model_config, seed=42)
    elif agent_type == 'ppo':
        traffic_agent = PPO(env.n_s_ls, env.n_a_ls, env.n_w_ls, total_step, traffic_model_config, seed=42)
    elif agent_type == 'iqld':
        traffic_agent = IQL(env.n_s_ls, env.n_a_ls, env.n_w_ls, total_step, traffic_model_config, seed=42, model_type='dqn')
    elif agent_type == 'iqll':
        traffic_agent = IQL(env.n_s_ls, env.n_a_ls, env.n_w_ls, total_step, traffic_model_config, seed=42, model_type='lr')
    else:
        logging.warning(f"Unknown agent type '{agent_type}', falling back to IA2C")
        traffic_agent = IA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, total_step, traffic_model_config, seed=42)
    
    load_path = _resolve_checkpoint_dir(model_dir)
    if os.path.exists(load_path):
        logging.info(f"Loading Traffic Agent weights from: {load_path}")
        load_path_slash = load_path if load_path.endswith(os.sep) else load_path + os.sep
        loaded = traffic_agent.load(load_path_slash)
        if not loaded:
            logging.warning("Traffic checkpoint load failed for %s; continuing from random/init weights.", load_path)
    else:
        logging.warning(f"WARNING: No model found at {load_path}. Random initialization.")

    return traffic_agent

def train_coevolution(args=None):
    if args is None:
        args = parse_args()

    tf.reset_default_graph()
    requested_agent = _normalize_agent_name(args.agent) if args.agent else None
    bootstrap_agent = requested_agent if requested_agent else 'ia2c'
    bootstrap_defaults = _resolve_default_paths(bootstrap_agent)
    bootstrap_config_path = (
        os.path.abspath(args.frozen_config)
        if args.frozen_config
        else bootstrap_defaults['config_path']
    )
    agent_name = requested_agent or _infer_agent_from_config(bootstrap_config_path, fallback='ia2c')
    defaults = _resolve_default_paths(agent_name)

    frozen_config_path = os.path.abspath(args.frozen_config) if args.frozen_config else defaults['config_path']
    frozen_model_dir = os.path.abspath(args.frozen_model_dir) if args.frozen_model_dir else defaults['frozen_model_dir']
    wce_dir = os.path.abspath(args.wce_dir) if args.wce_dir else defaults['wce_dir']
    default_base_dir = defaults['base_dir']
    base_dir = os.path.abspath(args.base_dir) if args.base_dir else default_base_dir
    total_episodes = int(args.total_episodes)
    checkpoint_interval_ep = int(args.checkpoint_interval_ep)

    # 1. Setup
    dirs = init_dir(base_dir, pathes=['log', 'data', 'model_traffic', 'model_adversary'])
    init_log(dirs['log'])

    logging.info("Agent family: %s", agent_name)
    logging.info("Frozen config: %s", frozen_config_path)
    logging.info("Frozen traffic model dir: %s", frozen_model_dir)
    logging.info("Frozen adversary dir: %s", wce_dir)
    logging.info("Co-evolution output dir: %s", base_dir)
    
    config = configparser.ConfigParser()
    config.read(frozen_config_path)
    env_config = dict(config['ENV_CONFIG'])
    env_config['mode'] = 'train'
    env_config['agent'] = agent_name
    model_config = ConfigWrapper(dict(config['MODEL_CONFIG']))
    
    # 2. Init Environment (Builds Graph)
    env = CoevolutionRealNetEnv(ConfigWrapper(env_config), output_path=dirs['data'])

    base_total_step = int(float(config.get('TRAIN_CONFIG', 'total_step', fallback='1000000')))
    traffic_steps_per_episode = int(np.ceil(float(env.episode_length_sec) /
                                            float(env.control_interval_sec)))
    traffic_total_step = base_total_step + (total_episodes * traffic_steps_per_episode)
    
    # 3. Load Traffic Agent
    traffic_agent = load_traffic_agent(
        env, env_config, model_config, frozen_model_dir, total_step=traffic_total_step
    )
    global_traffic_step = int(getattr(traffic_agent, 'loaded_checkpoint_step', 0))
    if hasattr(traffic_agent, 'set_train_step'):
        traffic_agent.set_train_step(global_traffic_step)
    
    # 4. Initialize Adversary (GCNA2C)
    logging.info("Initializing Graph Adversary (GCNA2C)...")
    
    # Extract Graph Params from Env
    global_state_dim = env.n_s_adversary  # Total flattened size (Nodes * Feat)
    feat_dim = int(getattr(env, 'adversary_feat_dim', 0))
    if feat_dim <= 0:
        # Fallback for compatibility with older env variants.
        feat_dim = int(global_state_dim / max(1, int(getattr(env, 'num_nodes', 1))))
    
    adversary = GCNA2C(
        n_s=global_state_dim,
        n_a=env.n_adversary_action,
        total_step=20000,
        model_config=model_config,
        adj_matrix=env.adj_matrix,
        num_nodes=env.num_nodes,
        feat_dim=feat_dim,
        seed=42
    )
    
    # Load Adversary weights if available
    adv_model_dir = _resolve_checkpoint_dir(wce_dir)
    adversary_loaded = False
    if os.path.exists(adv_model_dir):
        logging.info(f"Loading Adversary weights from: {adv_model_dir}")
        try:
            adv_model_dir_slash = adv_model_dir if adv_model_dir.endswith(os.sep) else adv_model_dir + os.sep
            adversary_loaded = adversary.load(adv_model_dir_slash)
            if not adversary_loaded:
                logging.warning("No adversary checkpoint found under %s; training from scratch.", adv_model_dir)
        except Exception as e:
            logging.warning(f"Failed to load adversary weights (Architecture mismatch?): {e}")
    
    # 5. Training Loop
    summary_writer = tf.summary.FileWriter(dirs['log'])
    loaded_adv_episode = int(getattr(adversary, 'loaded_checkpoint_step', 0)) if adversary_loaded else 0
    start_episode = loaded_adv_episode + 1 if adversary_loaded else 0
    global_adv_step = loaded_adv_episode
    if hasattr(adversary, 'set_train_step'):
        adversary.set_train_step(global_adv_step)

    batch_size = int(model_config.get('batch_size', fallback=120))
    
    if start_episode >= total_episodes:
        logging.info(
            "Loaded adversary episode %d and total_episodes=%d. Nothing to train.",
            loaded_adv_episode, total_episodes
        )
        env.terminate()
        return

    logging.info(
        "Starting GCN-based Co-Evolution Training at episode %d/%d...",
        start_episode, total_episodes - 1
    )
    
    for ep in range(start_episode, total_episodes):
        logging.info(
            "Episode %d start (global_traffic_step=%d, global_adv_step=%d)",
            ep, global_traffic_step, global_adv_step
        )
        adv_obs = env.reset()
        traffic_agent.reset() 
        done = False
        ep_adv_reward = 0
        
        while not done:
            # Adversary Step
            adv_forward_start = time.time()
            adv_policy, adv_value = adversary.forward(adv_obs, done, 'pv')
            action_to_store = np.squeeze(adv_policy)
            value_to_store = np.squeeze(adv_value)
            adv_forward_elapsed = time.time() - adv_forward_start
            
            # Environment Step
            env_step_start = time.time()
            next_adv_obs, adv_reward, done, info = env.step(
                adversary_action=action_to_store,
                traffic_agent=traffic_agent,
                summary_writer=summary_writer,
                global_traffic_step=global_traffic_step
            )
            env_step_elapsed = time.time() - env_step_start
            
            global_traffic_step = info['global_traffic_step']
            
            # Adversary Training
            adversary.add_transition(adv_obs, action_to_store, adv_reward, value_to_store, done)
            
            if adversary.trans_buffer.size >= batch_size or done:
                if done:
                    R = 0
                else:
                    _, next_val = adversary.forward(next_adv_obs, False, 'pv')
                    R = np.squeeze(next_val)
                adv_backward_start = time.time()
                adversary.backward(R, summary_writer, global_adv_step)
                adv_backward_elapsed = time.time() - adv_backward_start
            else:
                adv_backward_elapsed = 0.0
            
            adv_obs = next_adv_obs
            ep_adv_reward += adv_reward
            global_adv_step += 1
            logging.info(
                "Adv step %d | wce_reward=%.4f | done=%s | forward=%.2fs | env_step=%.2fs | backward=%.2fs | traffic_step=%d",
                global_adv_step, float(adv_reward), done, adv_forward_elapsed,
                env_step_elapsed, adv_backward_elapsed, global_traffic_step
            )
            
        logging.info(f"Ep {ep}: Adv Reward={ep_adv_reward:.2f}, Global Steps={global_traffic_step}")
        
        if ep > 0 and ep % checkpoint_interval_ep == 0:
            traffic_agent.save(dirs['model_traffic'], global_traffic_step)
            adversary.save(dirs['model_adversary'], ep)

    env.terminate()

if __name__ == '__main__':
    train_coevolution()
