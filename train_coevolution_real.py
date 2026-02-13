import os
import logging
import configparser
import numpy as np
import tensorflow as tf
import shutil
import sys

sys.path.append(os.getcwd())

from utils import init_dir, init_log
from train_adversary import ConfigWrapper, set_seed, get_logger
from envs.coevolution_real_net_env import CoevolutionRealNetEnv
# Assuming GCNA2C is available in agents.models (based on user request)
from agents.models import A2C, IA2C, MA2C, IQL, CNNA2C, GCNA2C

# --- PATH CONFIGURATION ---
FROZEN_CONFIG_PATH = './config/config_ia2c_real.ini'
FROZEN_MODEL_DIR = './runs/ia2c_real'
WORST_CASE_ESTIMATOR_DIR = './output_adversary_monaco/'

def load_traffic_agent(env, env_config, traffic_model_config, model_dir):
    agent_type = env_config.get('agent', 'ia2c')
    logging.info(f"Initializing Traffic Agent ({agent_type}) for {env.node_names}...")

    if agent_type == 'a2c':
        traffic_agent = A2C(env.n_s, env.n_a, 0, traffic_model_config, seed=42)
    elif agent_type == 'ia2c':
        traffic_agent = IA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, 0, traffic_model_config, seed=42)
    elif agent_type == 'ma2c':
        traffic_agent = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, 0, traffic_model_config, seed=42)
    elif agent_type == 'iqld':
        traffic_agent = IQL(env.n_s_ls, env.n_a_ls, env.n_w_ls, 0, traffic_model_config, seed=42, model_type='dqn')
    elif agent_type == 'iqll':
        traffic_agent = IQL(env.n_s_ls, env.n_a_ls, env.n_w_ls, 0, traffic_model_config, seed=42, model_type='lr')
    else:
        logging.warning(f"Unknown agent type '{agent_type}', falling back to IA2C")
        traffic_agent = IA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, 0, traffic_model_config, seed=42)
    
    load_path = os.path.join(model_dir, 'model')
    if os.path.exists(load_path) or os.path.exists(load_path + '.index'):
        logging.info(f"Loading Traffic Agent weights from: {load_path}")
        traffic_agent.load(load_path + '/')
    else:
        logging.warning(f"WARNING: No model found at {load_path}. Random initialization.")

    return traffic_agent

def train_coevolution():
    # 1. Setup
    base_dir = './output_coevolution_real/'
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    
    config = configparser.ConfigParser()
    config.read(FROZEN_CONFIG_PATH)
    env_config = dict(config['ENV_CONFIG'])
    env_config['mode'] = 'train'
    model_config = dict(config['MODEL_CONFIG'])
    
    # 2. Init Environment (Builds Graph)
    env = CoevolutionRealNetEnv(ConfigWrapper(env_config), output_path=dirs['data'])
    
    # 3. Load Traffic Agent
    traffic_agent = load_traffic_agent(env, env_config, model_config, FROZEN_MODEL_DIR)
    
    # 4. Initialize Adversary (GCNA2C)
    logging.info("Initializing Graph Adversary (GCNA2C)...")
    
    # Extract Graph Params from Env
    global_state_dim = env.n_s_adversary # Total flattened size (Nodes * Feat)
    max_feat_dim = env.max_feat_dim      # Feature size per node
    
    adversary = GCNA2C(
        n_s=global_state_dim,
        n_a=env.n_adversary_action,
        total_step=20000,
        model_config=ConfigWrapper(model_config),
        adj_matrix=env.adj_matrix,
        num_nodes=env.num_nodes,
        feat_dim=max_feat_dim,
        seed=42
    )
    
    # Load Adversary weights if available
    adv_load_path = os.path.join(WORST_CASE_ESTIMATOR_DIR, 'checkpoint')
    if os.path.exists(WORST_CASE_ESTIMATOR_DIR):
        logging.info(f"Loading Adversary weights from {adv_load_path}")
        try:
            adversary.load(adv_load_path)
        except Exception as e:
            logging.warning(f"Failed to load adversary weights (Architecture mismatch?): {e}")
    
    # 5. Training Loop
    summary_writer = tf.summary.FileWriter(dirs['log'])
    total_episodes = 1000
    global_traffic_step = 0
    global_adv_step = 0
    batch_size = int(model_config.get('batch_size', 120))
    
    logging.info("Starting GCN-based Co-Evolution Training...")
    
    for ep in range(total_episodes):
        adv_obs = env.reset()
        traffic_agent.reset() 
        done = False
        ep_adv_reward = 0
        
        while not done:
            # Adversary Step
            adv_policy, adv_value = adversary.forward(adv_obs, done, 'pv')
            
            # Environment Step
            next_adv_obs, adv_reward, done, info = env.step(
                adversary_action=adv_policy, 
                traffic_agent=traffic_agent,
                summary_writer=summary_writer,
                global_traffic_step=global_traffic_step
            )
            
            global_traffic_step = info['global_traffic_step']
            
            # Adversary Training
            adversary.add_transition(adv_obs, adv_policy, adv_reward, adv_value, done)
            
            if adversary.trans_buffer.size >= batch_size or done:
                if done:
                    R = 0
                else:
                    _, next_val = adversary.forward(next_adv_obs, False, 'v')
                    R = np.squeeze(next_val)
                adversary.backward(R, summary_writer, global_adv_step)
            
            adv_obs = next_adv_obs
            ep_adv_reward += adv_reward
            global_adv_step += 1
            
        logging.info(f"Ep {ep}: Adv Reward={ep_adv_reward:.2f}, Global Steps={global_traffic_step}")
        
        if ep > 0 and ep % 10 == 0:
            traffic_agent.save(dirs['model'], global_traffic_step)
            adversary.save(dirs['model'], ep)

    env.terminate()

if __name__ == '__main__':
    train_coevolution()