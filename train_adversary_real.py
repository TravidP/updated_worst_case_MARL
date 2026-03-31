import os
import logging
import configparser
import numpy as np
from tf_compat import tf
import sys
import argparse
from envs.adversarial_real_net_env import AdversarialRealNetEnv
from agents.models import GCNA2C
from train_adversary import ConfigWrapper, set_seed, get_logger

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _script_rel(*parts):
    return os.path.join(SCRIPT_DIR, *parts)


def _has_model_checkpoint(run_dir):
    model_dir = os.path.join(run_dir, 'model')
    if not os.path.isdir(model_dir):
        return False
    if os.path.exists(os.path.join(model_dir, 'checkpoint')):
        return True
    for file_name in os.listdir(model_dir):
        if file_name.startswith('checkpoint-') and file_name.endswith('.index'):
            return True
    return False


def _resolve_default_frozen_paths_real(agent):
    agent = str(agent).strip().lower()
    if agent == 'ia2c':
        return (
            _script_rel('config', 'config_ia2c_real.ini'),
            _script_rel('runs', 'ia2c_real'),
            _script_rel('output_adversary_monaco', 'ia2c_real'),
        )
    if agent == 'ma2c':
        return (
            _script_rel('config', 'config_ma2c_real.ini'),
            _script_rel('runs', 'ma2c_real'),
            _script_rel('output_adversary_monaco', 'ma2c_real'),
        )
    if agent == 'iqll':
        return (
            _script_rel('config', 'config_iqll_real.ini'),
            _script_rel('runs', 'iqll_real'),
            _script_rel('output_adversary_monaco', 'iqll_real'),
        )
    if agent == 'iqld':
        return (
            _script_rel('config', 'config_iqld_real.ini'),
            _script_rel('runs', 'iqld_real'),
            _script_rel('output_adversary_monaco', 'iqld_real'),
        )
    if agent == 'ppo':
        return (
            _script_rel('config', 'config_ppo_real.ini'),
            _script_rel('runs', 'ppo_real'),
            _script_rel('output_adversary_monaco', 'ppo_real'),
        )
    raise ValueError(
        "Unsupported --agent '%s'. Use ia2c, ma2c, iqll, iqld, or ppo." % agent
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train real-grid adversary (GCN-A2C) against a frozen controller.'
    )
    parser.add_argument(
        '--agent',
        type=str,
        default='ppo',
        choices=['ia2c', 'ma2c', 'iqll', 'iqld', 'ppo'],
        help='Frozen traffic controller type used to select default paths.',
    )
    parser.add_argument(
        '--frozen-config',
        type=str,
        default=None,
        help='Path to frozen controller config .ini. Overrides --agent default.',
    )
    parser.add_argument(
        '--frozen-model-dir',
        type=str,
        default=None,
        help='Path to frozen controller run directory (expects data/ and model/).',
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Adversary output directory.',
    )
    parser.add_argument(
        '--total-episodes',
        type=int,
        default=500,
        help='Number of adversary training episodes.',
    )
    parser.add_argument(
        '--checkpoint-interval-ep',
        type=int,
        default=10,
        help='Save adversary checkpoint every N episodes.',
    )
    parser.add_argument(
        '--resume-checkpoint',
        type=str,
        default='latest',
        help="Episode index to resume from, or 'latest'.",
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable adversary checkpoint resume.',
    )
    return parser.parse_args()

def log_episode_stats(writer, episode, reward, steps, action_history, n_actions):
    """
    Manually logs episode-level statistics to TensorBoard.
    Includes Action Frequency analysis to detect mode collapse.
    """
    if writer is None:
        return
        
    summary = tf.Summary()
    
    # 1. Performance Metrics
    summary.value.add(tag='Episode/Total_Congestion_Reward', simple_value=reward)
    summary.value.add(tag='Episode/Steps', simple_value=steps)
    
    # 2. Action Distribution (Which scenarios did the adversary pick?)
    if action_history:
        # Calculate frequency of each action index
        counts = np.bincount(action_history, minlength=n_actions) 
        probs = counts / np.sum(counts)
        for act_idx, prob in enumerate(probs):
            summary.value.add(tag=f'Adversary/Action_{act_idx}_Freq', simple_value=prob)

    writer.add_summary(summary, episode)
    writer.flush()

def train_adversary_real(args=None):
    if args is None:
        args = parse_args()

    tf.reset_default_graph()

    default_config_path, default_model_dir, default_base_dir = _resolve_default_frozen_paths_real(args.agent)
    base_dir = os.path.abspath(args.base_dir) if args.base_dir else default_base_dir
    FROZEN_CONFIG_PATH = os.path.abspath(args.frozen_config) if args.frozen_config else default_config_path
    FROZEN_MODEL_DIR = os.path.abspath(args.frozen_model_dir) if args.frozen_model_dir else default_model_dir
    total_episodes = int(args.total_episodes)
    checkpoint_interval_ep = int(args.checkpoint_interval_ep)
    should_resume = not args.no_resume

    os.makedirs(base_dir, exist_ok=True)
    model_dir = os.path.join(base_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    logger = get_logger(base_dir)

    # 1. Configuration
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
    env_config = dict(parser['ENV_CONFIG'])
    env_config['mode'] = 'train'
    env_config.setdefault('seed', 42)
    env_config.setdefault('algo', 'adversary')
    model_config = ConfigWrapper(dict(parser['MODEL_CONFIG']))

    # 2. Init Environment
    logger.info("Initializing Adversarial Environment...")
    env = AdversarialRealNetEnv(
        ConfigWrapper(env_config), 
        output_path=base_dir,
        frozen_model_dir=FROZEN_MODEL_DIR
    )
    set_seed(int(env_config['seed']))
    
    # 3. Calculate Dimensions for GCN
    # s_dim_ls contains specific dims, we took max in Env to pad features
    max_feat_dim = max(env.n_s_ls) 
    global_state_dim = env.num_nodes * max_feat_dim

    # 4. Init GCN Agent
    logger.info("Initializing GCN Adversary...")
    adversary = GCNA2C(
        n_s=global_state_dim,
        n_a=env.n_adversary_action,
        total_step=20000,
        model_config=model_config,
        adj_matrix=env.adj_matrix,
        num_nodes=env.num_nodes,
        feat_dim=max_feat_dim,
        seed=int(env_config['seed'])
    )
    
    # Use adversary.sess.graph to ensure correct graph context
    summary_writer = tf.summary.FileWriter(base_dir, adversary.sess.graph)
    
    # 5. Training Loop
    global_step = 0
    batch_size = model_config.getint('batch_size')
    model_dir_slash = model_dir if model_dir.endswith('/') else model_dir + '/'

    start_episode = 0
    if should_resume:
        checkpoint_arg = None
        if str(args.resume_checkpoint).strip().lower() not in ['', 'latest', 'auto']:
            checkpoint_arg = int(args.resume_checkpoint)
        loaded = adversary.load(model_dir_slash, checkpoint=checkpoint_arg)
        if loaded:
            loaded_ep = int(getattr(adversary, 'loaded_checkpoint_step', 0))
            start_episode = loaded_ep + 1
            logger.info("Resumed adversary from checkpoint episode=%d", loaded_ep)
        else:
            logger.info("Resume requested but no adversary checkpoint found in %s. Starting fresh.", model_dir)
    
    logger.info("Start Training...")
    
    try:
        for ep in range(start_episode, total_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0
            ep_steps = 0
            ep_actions = [] # Track actions for this episode
            
            while not done:
                # Get Action (Weights)
                # 'out_type' must be 'pv' so the Critic calculates the value of the current state.
                action_logits, value = adversary.forward(obs, done, out_type='pv')
                action_logits = np.squeeze(action_logits)
                
                # Store dominant action index for logging
                scenario_weights = env._normalize_action_weights(action_logits)
                ep_actions.append(int(np.argmax(scenario_weights)))

                # Step Env
                next_obs, reward, done, _ = env.step(action_logits)
                
                # Store Transition
                adversary.add_transition(obs, action_logits, reward, value, done)
                
                # Backward pass: Check buffer size OR if episode is done to flush buffer
                if adversary.trans_buffer.size >= batch_size or done:
                    if done:
                        bootstrap_R = 0
                    else:
                        _, next_value = adversary.forward(next_obs, done=False, out_type='pv')
                        bootstrap_R = np.squeeze(next_value)
                    adversary.backward(R=bootstrap_R, summary_writer=summary_writer, global_step=global_step)
                    
                obs = next_obs
                ep_reward += reward
                ep_steps += 1
                global_step += 1
                
            logger.info(f"Episode {ep}: Steps={ep_steps}, Reward={ep_reward:.2f}")
            
            # --- LOG EPISODE STATS ---
            log_episode_stats(summary_writer, ep, ep_reward, ep_steps, ep_actions, env.n_adversary_action)

            if ((ep + 1) % checkpoint_interval_ep == 0) or (ep == total_episodes - 1):
                adversary.save(model_dir_slash, global_step=ep)
                logger.info("Saved checkpoint at episode=%d to %s", ep, model_dir)
    finally:
        env.terminate()
        summary_writer.flush()
        summary_writer.close()
        logger.info("Training Finished.")

if __name__ == '__main__':
    train_adversary_real()
