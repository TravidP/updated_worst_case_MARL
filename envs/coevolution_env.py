from envs.adversarial_large_grid_env import AdversarialLargeGridEnv
from envs.large_grid_env import LargeGridEnv
import numpy as np
import logging

class CoevolutionLargeGridEnv(AdversarialLargeGridEnv):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        # We don't load a frozen controller here because we pass the trainable agent in step()
        super().__init__(config, port, output_path, is_record, record_stat, frozen_model_dir='dummy_path')
        self.adversary_step_duration = 600 # Default to 10 mins, can be overridden
    def _load_frozen_controller(self, agent_dir):
        # OVERRIDE: Do nothing. We control traffic externally via the trainable Traffic Agent.
        logging.info("Coevolution Env: 'dummy_path' passed. No frozen controller loaded (Trainable Agent Active).")
        pass

    def step(self, adversary_action, traffic_agent, summary_writer=None, global_traffic_step=0):
        """
        Executes one ADVERSARY Step (e.g., 10 minutes).
        Inside, it runs the Traffic Agent loop (per-step control & training).
        
        Args:
            adversary_action: Weights for traffic injection.
            traffic_agent: The trainable IA2C agent object.
            summary_writer: For logging traffic agent metrics.
            global_traffic_step: Current global step counter for the traffic agent.
            
        Returns:
            next_adv_obs: Observation for the Adversary (next segment).
            adv_reward: Reward for the Adversary (inverted traffic reward).
            done: Whether the full 1-hour episode is finished.
            info: Contains updated 'global_traffic_step'.
        """
        # --- 1. ADVERSARY ACTS (Inject Traffic) ---
        logging.info(f"\n[Adversary] Injecting Traffic... (Time: {self.cur_sec}s)")
        
        # Process Weights
        raw_weights = np.array(adversary_action, dtype=np.float32).flatten()
        weights = np.maximum(raw_weights, 0.0)
        total_weight = np.sum(weights)
        if total_weight > 1e-6:
            weights = weights / total_weight
        else:
            weights = np.ones_like(weights) / len(weights)
        # --- DEBUG PRINT START ---
        print("\n[DEBUG] Adversary Weights:")
        print(f"Raw Weights: {raw_weights}")
        print(f"Normalized Weights: {weights}")
        # --- DEBUG PRINT END ---
            
        # Inject Traffic
        self._inject_dynamic_traffic(weights, duration=self.adversary_step_duration)

        # --- 2. TRAFFIC AGENT LOOP (Inner Loop) ---
        # Run simulation for 'adversary_step_duration' (e.g., 600s)
        steps_to_run = int(self.adversary_step_duration / self.control_interval_sec)
        segment_traffic_reward = 0
        done = False
        
        traffic_obs = self._get_state() # Initial state for this segment

        for _ in range(steps_to_run):
            # A. Traffic Agent Decision
            # Note: We assume traffic_agent handles its own graph context internally 
            # or is called within the correct context in main.py.
            # However, for safety, IA2C usually needs explicit session calls.
            # Ideally, traffic_agent.forward uses its own sess.
            # --- A. Traffic Agent Decision ---
            # Determine if the agent is IQL-based (value-based) or A2C-based (actor-critic)
            is_iql = traffic_agent.name.startswith('iql')
            
            # Ensure we are in the agent's graph context
            # Fallback to sess.graph if 'graph' attribute is missing
            graph_context = traffic_agent.graph if hasattr(traffic_agent, 'graph') else traffic_agent.sess.graph
            
            with graph_context.as_default():
                if is_iql:
                    # IQL returns (actions, q_values). We only need actions here.
                    # Stochastic=True allows exploration (epsilon-greedy) during training.
                    actions, _ = traffic_agent.forward(traffic_obs, mode='act', stochastic=True)
                    values = None # IQL doesn't use state-value V(s) for transitions
                else:
                    # IA2C/MA2C returns (policies, values)
                    policies, values = traffic_agent.forward(traffic_obs, done=False, out_type='pv')
                    
                    actions = []
                    # Sample actions from the probability distributions
                    for pi in policies:
                        actions.append(np.random.choice(len(pi), p=pi))

            # B. Simulation Step (Yellow -> Green)
            # Yellow
            self._set_phase(actions, 'yellow', self.yellow_interval_sec)
            self._simulate(self.yellow_interval_sec)
            # Green
            rest = self.control_interval_sec - self.yellow_interval_sec
            self._set_phase(actions, 'green', rest)
            self._simulate(rest)

            # C. Measure Reward & Next State
            step_rewards = self._measure_reward_step()
            next_traffic_obs = self._get_state()
            
            # Check if episode ended (e.g. 3600s reached)
            if self.cur_sec >= self.episode_length_sec:
                done = True

            # D. Traffic Agent Learn (Per Step)
            # --- D. Traffic Agent Learn (Per Step) ---
            if is_iql:
                # IQL Transition: (obs, action, reward, next_obs, done)
                traffic_agent.add_transition(traffic_obs, actions, step_rewards, next_traffic_obs, done)
            else:
                # A2C Transition: (obs, action, reward, value, done)
                traffic_agent.add_transition(traffic_obs, actions, step_rewards, values, done)
            # --- FIX: Check the first agent's buffer in the list ---
            # IA2C uses 'trans_buffer_ls' instead of 'trans_buffer'
            # --- FIX: Check the buffer length correctly ---
            if hasattr(traffic_agent, 'trans_buffer_ls'):
                 # Multi-agent case (IA2C, MA2C) - check first agent's buffer list length
                 buffer_len = len(traffic_agent.trans_buffer_ls[0].obs)
            else:
                 # Single-agent case (A2C, CNNA2C) - check buffer list length
                 buffer_len = len(traffic_agent.trans_buffer.obs)

            if buffer_len >= traffic_agent.n_step:
                if is_iql:
                    # IQL Backward: Learns from Replay Buffer (no bootstrap R needed)
                    traffic_agent.backward(summary_writer=summary_writer, 
                                           global_step=global_traffic_step)
                else:
                    # IA2C/MA2C Backward: Needs bootstrap returns (R_ls).
                    # Since we are stepping continuously, we pass 0s or estimated values.
                    # For simplicity in this loop (similar to your snippet), we use 0s.
                    if hasattr(traffic_agent, 'n_agent'):
                        R_ls = [0] * traffic_agent.n_agent
                    else:
                        R_ls = 0
                    
                    traffic_agent.backward(R_ls, 
                                           summary_writer=summary_writer, 
                                           global_step=global_traffic_step)
        

            # Update loop variables
            segment_traffic_reward += np.sum(step_rewards)
            traffic_obs = next_traffic_obs
            global_traffic_step += 1
            
            if done:
                break

        # --- 3. RETURN ADVERSARY RESULTS ---
        # Adversary Reward = Negative of Traffic Reward (Zero-Sum)
        adv_reward = -segment_traffic_reward / 100.0
        
        # Get next Adversary State
        if done:
            next_adv_obs = np.zeros(self.n_s_adversary)
        else:
            next_adv_obs = self._get_adversary_state()

        info = {'global_traffic_step': global_traffic_step}
        
        return next_adv_obs, adv_reward, done, info