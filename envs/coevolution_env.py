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

    def step(self, adversary_action, traffic_agent, summary_writer=None, global_traffic_step=0,
             watchdog_touch=None):
        """
        Executes one ADVERSARY Step (e.g., 10 minutes).
        Inside, it runs the Traffic Agent loop (per-step control & training).
        
        Args:
            adversary_action: Unconstrained WCE logits for traffic mixing.
            traffic_agent: The trainable IA2C agent object.
            summary_writer: For logging traffic agent metrics.
            global_traffic_step: Current global step counter for the traffic agent.
            
        Returns:
            next_adv_obs: Observation for the Adversary (next segment).
            adv_reward: Reward for the Adversary (inverted traffic reward).
            done: Whether the full 1-hour episode is finished.
            info: Contains updated 'global_traffic_step' and segment reward stats.
        """
        # --- 1. ADVERSARY ACTS (Inject Traffic) ---
        if callable(watchdog_touch):
            watchdog_touch('coev:env_step:inject_traffic')
        logging.info(f"\n[Adversary] Injecting Traffic... (Time: {self.cur_sec}s)")
        
        # Keep the adversary action transform identical to standalone WCE training.
        weights = self._normalize_action_weights(adversary_action)
            
        # Inject Traffic
        self._inject_dynamic_traffic(weights, duration=self.adversary_step_duration)

        # --- 2. TRAFFIC AGENT LOOP (Inner Loop) ---
        # Run simulation for 'adversary_step_duration' (e.g., 600s)
        steps_to_run = int(self.adversary_step_duration / self.control_interval_sec)
        segment_traffic_reward = 0
        done = False
        
        traffic_obs = self._get_state() # Initial state for this segment

        for step_i in range(steps_to_run):
            if callable(watchdog_touch):
                watchdog_touch('coev:env_step:control_step_%d' % step_i)
            # A. Traffic Agent Decision
            # Note: We assume traffic_agent handles its own graph context internally 
            # or is called within the correct context in main.py.
            # However, for safety, IA2C usually needs explicit session calls.
            # Ideally, traffic_agent.forward uses its own sess.
            # --- A. Traffic Agent Decision ---
            # Determine if the agent is IQL-based (value-based) or A2C-based (actor-critic)
            is_iql = traffic_agent.name.startswith('iql')
            is_ppo = traffic_agent.name == 'ppo'
            
            # Ensure we are in the agent's graph context
            # Fallback to sess.graph if 'graph' attribute is missing
            graph_context = traffic_agent.graph if hasattr(traffic_agent, 'graph') else traffic_agent.sess.graph
            
            if callable(watchdog_touch):
                watchdog_touch('coev:env_step:traffic_forward')
            with graph_context.as_default():
                if is_iql:
                    # IQL returns (actions, q_values). We only need actions here.
                    # Use the same epsilon-greedy exploration mode as the standard trainer.
                    actions, _ = traffic_agent.forward(traffic_obs, mode='explore')
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
            if callable(watchdog_touch):
                watchdog_touch('coev:env_step:simulate_yellow')
            self._set_phase(actions, 'yellow', self.yellow_interval_sec)
            self._simulate(self.yellow_interval_sec)
            # Green
            rest = self.control_interval_sec - self.yellow_interval_sec
            if callable(watchdog_touch):
                watchdog_touch('coev:env_step:simulate_green')
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
                if is_ppo:
                    traffic_agent.add_transition(
                        traffic_obs, actions, step_rewards, values, done, policies
                    )
                else:
                    traffic_agent.add_transition(traffic_obs, actions, step_rewards, values, done)
            # --- FIX: Check the first agent's buffer in the list ---
            # IA2C uses 'trans_buffer_ls' instead of 'trans_buffer'
            # --- FIX: Check the buffer length correctly ---
            if hasattr(traffic_agent, 'trans_buffer_ls'):
                 # Multi-agent case (IA2C/MA2C/IQL): infer length from available buffer API.
                 first_buf = traffic_agent.trans_buffer_ls[0]
                 if hasattr(first_buf, 'obs'):
                     buffer_len = len(first_buf.obs)
                 elif hasattr(first_buf, 'size'):
                     buffer_len = first_buf.size
                 else:
                     buffer_len = 0
            else:
                 # Single-agent case (A2C/CNNA2C)
                 if hasattr(traffic_agent.trans_buffer, 'obs'):
                     buffer_len = len(traffic_agent.trans_buffer.obs)
                 elif hasattr(traffic_agent.trans_buffer, 'size'):
                     buffer_len = traffic_agent.trans_buffer.size
                 else:
                     buffer_len = 0

            if buffer_len >= traffic_agent.n_step:
                if is_iql:
                    # IQL Backward: Learns from Replay Buffer (no bootstrap R needed)
                    if callable(watchdog_touch):
                        watchdog_touch('coev:env_step:traffic_backward_iql')
                    traffic_agent.backward(summary_writer=summary_writer, 
                                           global_step=global_traffic_step)
                else:
                    # IA2C/MA2C Backward: bootstrap with next state value when not done.
                    if done:
                        if hasattr(traffic_agent, 'n_agent'):
                            R_ls = [0] * traffic_agent.n_agent
                        else:
                            R_ls = 0
                    else:
                        graph_context = traffic_agent.graph if hasattr(traffic_agent, 'graph') else traffic_agent.sess.graph
                        with graph_context.as_default():
                            _, next_values = traffic_agent.forward(next_traffic_obs, done=False, out_type='pv')
                        R_ls = next_values
                    
                    if callable(watchdog_touch):
                        watchdog_touch('coev:env_step:traffic_backward_a2c')
                    traffic_agent.backward(R_ls, 
                                           summary_writer=summary_writer, 
                                           global_step=global_traffic_step)
                    if callable(watchdog_touch):
                        watchdog_touch('coev:env_step:traffic_backward_a2c_done')
        

            # Update loop variables
            segment_traffic_reward += np.sum(step_rewards)
            traffic_obs = next_traffic_obs
            global_traffic_step += 1
            
            if done:
                break

        # --- 3. RETURN ADVERSARY RESULTS ---
        # Adversary Reward = Negative of Traffic Reward (Zero-Sum)
        adv_reward = -segment_traffic_reward / float(self.adversary_reward_scale)
        
        # Get next Adversary State
        if done:
            next_adv_obs = np.zeros(self.n_s_adversary)
        else:
            next_adv_obs = self._get_adversary_state()

        info = {
            'global_traffic_step': global_traffic_step,
            'segment_traffic_reward': float(segment_traffic_reward),
            'segment_wce_reward': float(adv_reward),
            'segment_control_steps': int(steps_to_run)
        }
        
        return next_adv_obs, adv_reward, done, info
