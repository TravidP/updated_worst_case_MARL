import os
import logging
import configparser
import numpy as np
import tensorflow as tf
import sys
from envs.adversarial_real_net_env import AdversarialRealNetEnv
from agents.models import GCNA2C
from train_adversary import ConfigWrapper, set_seed, get_logger

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

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

def train_adversary_real():
    base_dir = './output_adversary_monaco/'
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    logger = get_logger(base_dir)

    # 1. Configuration
    FROZEN_CONFIG_PATH = './config/config_ia2c_real.ini' # Use Monaco Config
    FROZEN_MODEL_DIR = './runs/ia2c_real' # Path to trained IA2C on Monaco
    
    logger.info(f"Reading configuration from: {FROZEN_CONFIG_PATH}")
    
    parser = configparser.ConfigParser()
    parser.read(FROZEN_CONFIG_PATH)
    env_config = dict(parser['ENV_CONFIG'])
    env_config['mode'] = 'train'
    model_config = ConfigWrapper(dict(parser['MODEL_CONFIG']))

    # 2. Init Environment
    env = AdversarialRealNetEnv(
        ConfigWrapper(env_config), 
        output_path=base_dir,
        frozen_model_dir=FROZEN_MODEL_DIR
    )
    
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
        seed=42
    )
    
    # Use adversary.sess.graph to ensure correct graph context
    summary_writer = tf.summary.FileWriter(base_dir, adversary.sess.graph)
    
    # 5. Training Loop
    total_episodes = 500
    global_step = 0
    batch_size = model_config.getint('batch_size')
    
    logger.info("Start Training...")
    
    for ep in range(total_episodes):
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
            
            # Store the action taken (argmax) for logging purposes
            ep_actions.append(np.argmax(action_logits))
            
            # 2. Convert to Weights for Environment (Softmax)
            scenario_weights = softmax(action_logits)

            # Step Env
            next_obs, reward, done, _ = env.step(scenario_weights)
            
            # Store Transition
            # CRITICAL FIX: Pass 'value' as the 4th argument, NOT 'next_obs'.
            adversary.add_transition(obs, action_logits, reward, value, done)
            
            # Backward pass: Check buffer size OR if episode is done to flush buffer
            if adversary.trans_buffer.size >= batch_size or done:
                adversary.backward(R=0, summary_writer=summary_writer, global_step=global_step)
                
            obs = next_obs
            ep_reward += reward
            ep_steps += 1
            global_step += 1
            
        logger.info(f"Episode {ep}: Steps={ep_steps}, Reward={ep_reward:.2f}")
        
        # --- LOG EPISODE STATS ---
        log_episode_stats(summary_writer, ep, ep_reward, ep_steps, ep_actions, env.n_adversary_action)

        # Save less frequently (every 10 episodes) to save IO
        if ep > 0 and ep % 10 == 0:
            adversary.save(base_dir, global_step=ep)

    env.terminate()
    logger.info("Training Finished.")

if __name__ == '__main__':
    train_adversary_real()