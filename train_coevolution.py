import os
import logging
import configparser
import numpy as np
import tensorflow as tf
from utils import init_dir, init_log
from envs.coevolution_env import CoevolutionLargeGridEnv
# Updated imports to include all agent types
from agents.models import A2C, IA2C, MA2C, IQL, CNNA2C

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

    def _get_val(self, key):
        if key in self.config:
            return self.config[key]
        elif key in self.defaults:
            # logging.warning(f"Config key '{key}' not found. Using default: {self.defaults[key]}")
            return self.defaults[key]
        else:
            raise KeyError(f"Key '{key}' not found in config or defaults.")

    def getint(self, key): 
        return int(self._get_val(key))
        
    def getfloat(self, key): 
        return float(self._get_val(key))
        
    def getboolean(self, key): 
        val = str(self._get_val(key)).lower()
        return val == 'true'
        
    def get(self, key): 
        return self._get_val(key)
        
    def __getitem__(self, key): 
        return self._get_val(key)
        
    def __contains__(self, key): 
        return key in self.config or key in self.defaults

def train_coevolution():
    base_dir = './output_coevolution/'
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    
    # --- 1. CONFIGURATION ---
    ADVERSARY_CHECKPOINT_DIR = './output_adversary/'     
    TRAFFIC_CHECKPOINT_DIR = './runs/ia2c_large/model/'  
    CONFIG_FILE = './config/config_ia2c_large.ini'

    logging.info(f"Loading Config: {CONFIG_FILE}")
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    
    env_config_dict = dict(config['ENV_CONFIG'])
    env_config_dict['mode'] = 'train'
    env_config_wrapped = ConfigWrapper(env_config_dict)
    
    # --- 2. INIT ENVIRONMENT ---
    # Will now pass 'dummy_path' internally to avoid ValueError
    env = CoevolutionLargeGridEnv(env_config_wrapped, port=0, output_path=dirs['data'])
    
    # --- 3. INIT AGENTS ---
    # A) Adversary (Worst-Case Estimator)
    adv_model_config = ConfigWrapper(dict(config['MODEL_CONFIG']))
    batch_size = int(adv_model_config['batch_size']) 
    adversary = CNNA2C(
        n_s=env.n_s_adversary,
        n_a=env.n_adversary_action,
        total_step=100000,
        model_config=adv_model_config,
        seed=42
    )
    
    # B) Traffic Controller (Robust Agent)
    # Modified to dynamically load agent based on config 'agent' type
    traffic_model_config = ConfigWrapper(dict(config['MODEL_CONFIG']))
    agent_type = env_config_dict.get('agent', 'ia2c')  # Default to ia2c if not specified
    
    logging.info(f"Initializing Traffic Agent of type: {agent_type}")

    if agent_type == 'a2c':
        traffic_agent = A2C(env.n_s, env.n_a, 0, traffic_model_config, seed=42)
    elif agent_type == 'ia2c':
        traffic_agent = IA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, 0, traffic_model_config, seed=42)
    elif agent_type == 'ma2c':
        traffic_agent = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, 0, traffic_model_config, seed=42)
    elif agent_type == 'iqld':
        traffic_agent = IQL(env.n_s_ls, env.n_a_ls, env.n_w_ls, 0, traffic_model_config, seed=42, model_type='dqn')
    elif agent_type == 'iqll':
        # model_type='lr' triggers LRQPolicy in agents/models.py
        traffic_agent = IQL(env.n_s_ls, env.n_a_ls, env.n_w_ls, 0, traffic_model_config, seed=42, model_type='lr')
    else:
        # Fallback
        logging.warning(f"Unknown agent type '{agent_type}', falling back to IA2C")
        traffic_agent = IA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, 0, traffic_model_config, seed=42)

    
    # --- 4. LOAD PRE-TRAINED WEIGHTS ---
    logging.info(">>> Loading Pre-trained Weights...")
    if adversary.load(ADVERSARY_CHECKPOINT_DIR):
        logging.info("Loaded Adversary.")
    
    # Load traffic agent weights
    if traffic_agent.load(TRAFFIC_CHECKPOINT_DIR):
        logging.info(f"Loaded Traffic Agent ({agent_type}).")
    else:
        logging.warning(f"Failed to load Traffic Agent from {TRAFFIC_CHECKPOINT_DIR}")

    summary_writer = tf.summary.FileWriter(dirs['log'], traffic_agent.sess.graph)

    # --- 5. CO-EVOLUTION LOOP ---
    TOTAL_EPISODES = 1000
    ADV_ACTION_INTERVAL = 600 # 10 minutes
    
    # Sync env duration
    env.adversary_step_duration = ADV_ACTION_INTERVAL
    
    global_traffic_step = 0
    global_adv_step = 0
    
    logging.info("Starting Co-Evolution Training...")

    for ep in range(TOTAL_EPISODES):
        # Reset
        env.reset()
        traffic_agent.reset()
        adv_obs = env.get_adversary_state() # Initial Adversary State
        
        done = False
        
        logging.info(f"Episode {ep} Start.")
        
        while not done:
            # 1. Adversary Action (Weights)
            with adversary.graph.as_default():
                adv_action_logits, adv_value = adversary.forward(adv_obs, done=False, out_type='pv')
            action_to_store = np.squeeze(adv_action_logits)
            value_to_store = np.squeeze(adv_value)
            
            # 2. Environment Step (Runs Inner Loop for 10 mins)
            #    We pass the traffic_agent so the env can drive the inner simulation
            next_adv_obs, adv_reward, done, info = env.step(
                adversary_action=adv_action_logits,
                traffic_agent=traffic_agent,
                summary_writer=summary_writer,
                global_traffic_step=global_traffic_step
            )
            
            # Sync global step from inner loop
            global_traffic_step = info['global_traffic_step']
            
            # 3. Train Adversary (Once per 10 mins)
            adversary.add_transition(adv_obs, action_to_store, value_to_store, adv_value, done)
            if adversary.trans_buffer.size >= batch_size or done:
                if done:
                    R = 0
                else:
                    # If not done, we bootstrap with the value of the NEXT state
                    with adversary.graph.as_default():
                        _, next_val = adversary.forward(next_adv_obs, done=False, out_type='pv')
                    R = np.squeeze(next_val)
                
                # Perform update
                adversary.backward(R=R, summary_writer=summary_writer, global_step=global_adv_step)
            
            # Update Adversary State
            adv_obs = next_adv_obs
            global_adv_step += 1
            
            logging.info(f"   Adv Step {global_adv_step} | Reward: {adv_reward:.4f}")

        logging.info(f"Episode {ep} Finished.")
        
        if ep % 10 == 0:
            traffic_agent.save(dirs['model'], global_traffic_step)
            adversary.save(dirs['model'], ep)

    env.terminate()

if __name__ == '__main__':
    train_coevolution()