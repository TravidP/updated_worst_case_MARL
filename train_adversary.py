import os
import logging
import configparser
import numpy as np
import tensorflow as tf
import random
import sys
from envs.adversarial_large_grid_env import AdversarialLargeGridEnv
from agents.models import CNNA2C

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def log_episode_stats(writer, episode, reward, steps, action_history):
    """
    Manually logs episode-level statistics to TensorBoard.
    """
    if writer is None:
        return
        
    summary = tf.Summary()
    
    # 1. Performance Metrics
    summary.value.add(tag='Episode/Total_Congestion_Reward', simple_value=reward)
    summary.value.add(tag='Episode/Steps', simple_value=steps)
    
    # 2. Action Distribution (Which scenarios did the adversary pick?)
    # Calculate frequency of each action index
    if action_history:
        # Assuming 4 actions/scenarios, adjust range if needed
        counts = np.bincount(action_history, minlength=4) 
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
def make_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def get_logger(log_dir):
    logger = logging.getLogger('adversary_train')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    return logger

# --- MAIN TRAINING LOOP ---
def train_adversary():
    tf.reset_default_graph()
    base_dir = './output_adversary/'
    make_dir(base_dir)
    logger = get_logger(base_dir)

    # Path setup
    FROZEN_CONFIG_PATH = './config/config_iqll_large.ini'
    FROZEN_MODEL_DIR = './runs/iqll_large' 
    
    logger.info(f"Reading configuration from: {FROZEN_CONFIG_PATH}")
    
    parser = configparser.ConfigParser()
    parser.read(FROZEN_CONFIG_PATH)
    
    # Config Patching
    env_config = dict(parser['ENV_CONFIG'])
    env_config['data_path'] = './large_grid/data/' 
    env_config['mode'] = 'train'
    env_config['seed'] = 42
    env_config['algo'] = 'adversary'
    
    model_config = dict(parser['MODEL_CONFIG'])
    
    env_config_wrapped = ConfigWrapper(env_config)
    model_config_wrapped = ConfigWrapper(model_config)

    logger.info("Initializing Adversarial Environment...")
    env = AdversarialLargeGridEnv(
        env_config_wrapped, 
        output_path=base_dir,
        frozen_model_dir=FROZEN_MODEL_DIR
    )
    
    set_seed(int(env_config['seed']))
    env.init_test_seeds([int(env_config['seed'])])

    logger.info(f"Initializing CNN Adversary...")
    adversary = CNNA2C(
        n_s=env.n_s_adversary,       
        n_a=env.n_adversary_action,  
        total_step=20000,            
        model_config=model_config_wrapped,
        seed=int(env_config['seed'])
    )

    # --- TENSORBOARD SETUP ---
    # FIX: Use adversary.sess.graph, not undefined 'sess'
    summary_writer = tf.summary.FileWriter(base_dir, adversary.sess.graph)

    total_episodes = 500
    batch_size = int(model_config['batch_size']) 
    global_step = 0
    
    logger.info("Start Training...")
    
    for ep in range(total_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        
        # Track actions taken this episode
        ep_actions = []

        while not done:
            # 1. Adversary Pick: Request BOTH Policy (p) and Value (v)
            # 'out_type' must be 'pv' so the Critic calculates the value of the current state.
            action_logits, value = adversary.forward(obs, done, out_type='pv')

            # Since action_logits is likely [1, n_a], squeeze it
            action_logits = np.squeeze(action_logits)

            # 2. Convert to Weights for Environment (Softmax)
            scenario_weights = softmax(action_logits)

            # 3. Environment Step
            next_obs, reward, done, info = env.step(scenario_weights)

            # 4. Learn
            # CRITICAL FIX: Pass 'value' as the 4th argument, NOT 'next_obs'.
            # The signature is: add_transition(ob, action, reward, value, done)
            adversary.add_transition(obs, action_logits, reward, value, done)
            
            if adversary.trans_buffer.size >= batch_size or done:
                adversary.backward(R=0, summary_writer=summary_writer, global_step=global_step)
            
            obs = next_obs
            ep_reward += reward
            ep_steps += 1
            global_step += 1
            
        logger.info(f"Episode {ep}: Steps={ep_steps}, Reward={ep_reward:.2f}")
        
        # --- LOG EPISODE STATS ---
        log_episode_stats(summary_writer, ep, ep_reward, ep_steps, ep_actions)

        if ep > 0 and ep % 10 == 0:
            adversary.save(base_dir, global_step=ep)

    env.terminate()
    logger.info("Training Finished.")

if __name__ == '__main__':
    train_adversary()