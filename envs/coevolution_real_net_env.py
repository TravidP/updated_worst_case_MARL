import numpy as np
import logging
from envs.adversarial_real_net_env import AdversarialRealNetEnv

class CoevolutionRealNetEnv(AdversarialRealNetEnv):
    """
    Co-Evolution Environment for Real Network.
    
    Inherits from AdversarialRealNetEnv to use:
    - GCN Graph Setup (Adjacency Matrix & Feature Padding)
    - Dynamic Traffic Injection
    
    Overrides:
    - step: To allow a TRAINABLE traffic agent loop (Co-evolution).
    """
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        # Initialize Parent
        # We pass frozen_model_dir=None because in Co-evolution, the agent is passed dynamically in step()
        super().__init__(config, port, output_path, is_record, record_stat, frozen_model_dir=None)
        
        # Adjust Adversary Step Duration
        self.adversary_step_duration = 600  # 10 minutes

        # Note: self.adj_matrix, self.num_nodes, and self.adversary_feat_dim 
        # are already set up by the parent class in _setup_graph_for_gcn().

    def _load_frozen_controller(self, agent_dir):
        """Override to ensure no frozen controller is loaded by accident."""
        pass

    def step(self, adversary_action, traffic_agent, summary_writer=None, global_traffic_step=0):
        """
        Co-evolution Step:
        1. Adversary sets traffic weights (GCN Action).
        2. Environment simulates traffic.
        3. Traffic Agent (IA2C/MA2C/IQL) observes and trains.
        """
        # 1. Adversary Action (Normalize)
        weights = self._normalize_action_weights(adversary_action)
        
        # 2. Inject Traffic
        self._inject_dynamic_traffic(weights, duration=self.adversary_step_duration)

        # 3. Inner Simulation Loop (Training Traffic Agent)
        steps_to_run = int(self.adversary_step_duration / self.control_interval_sec)
        segment_traffic_reward = 0
        done = False
        
        # Detect Agent Type
        is_iql = traffic_agent.name.startswith('iql')
        
        # Get initial traffic state
        traffic_obs = self._get_state()
        
        for _ in range(steps_to_run):
            # A. Get Traffic Agent Action
            if is_iql:
                # IQL: forward returns (actions, q_values)
                # mode='explore' handles epsilon-greedy exploration
                actions, _ = traffic_agent.forward(traffic_obs, mode='explore')
                values = None # Not used for IQL transitions
            else:
                # A2C/IA2C/MA2C: forward returns (policies, values)
                if hasattr(traffic_agent, 'n_agent'): # Multi-agent
                    policies, values = traffic_agent.forward(traffic_obs, done=False, out_type='pv')
                    actions = []
                    for pi in policies:
                        actions.append(np.random.choice(len(pi), p=pi))
                else: # Single agent
                    policy, value = traffic_agent.forward(traffic_obs, done=False, out_type='pv')
                    actions = np.random.choice(len(policy), p=policy)
                    values = value
            
            # B. Execute Simulation Step
            self._set_phase(actions, 'yellow', self.yellow_interval_sec)
            self._simulate(self.yellow_interval_sec)
            
            rest = self.control_interval_sec - self.yellow_interval_sec
            self._set_phase(actions, 'green', rest)
            self._simulate(rest)
            
            # C. Measure Reward & Next State
            next_traffic_obs = self._get_state()
            step_rewards = self._measure_reward_step()
            
            if self.cur_sec >= self.episode_length_sec:
                done = True

            # D. Traffic Agent Learn (Per Step)
            if is_iql:
                # IQL Transition: (obs, action, reward, next_obs, done)
                traffic_agent.add_transition(traffic_obs, actions, step_rewards, next_traffic_obs, done)
            else:
                # A2C Transition: (obs, action, reward, value, done)
                traffic_agent.add_transition(traffic_obs, actions, step_rewards, values, done)
            
            # Check Buffer Length
            if hasattr(traffic_agent, 'trans_buffer_ls'):
                 # Multi-agent case (IA2C, MA2C) - check first agent's buffer list length
                 buffer_len = len(traffic_agent.trans_buffer_ls[0].obs)
            else:
                 # Single-agent case (A2C, CNNA2C) - check buffer list length
                 buffer_len = len(traffic_agent.trans_buffer.obs)

            # Backward Pass if Batch is Full
            if buffer_len >= traffic_agent.n_step:
                if is_iql:
                    # IQL Backward: Learns from Replay Buffer (no bootstrap R needed)
                    traffic_agent.backward(summary_writer=summary_writer, 
                                           global_step=global_traffic_step)
                else:
                    # IA2C/MA2C Backward: Needs bootstrap returns (R_ls).
                    if not done:
                        # Bootstrap with value of next state
                        if hasattr(traffic_agent, 'n_agent'):
                             _, next_values = traffic_agent.forward(next_traffic_obs, False, 'pv')
                             R_ls = next_values
                        else:
                             _, next_value = traffic_agent.forward(next_traffic_obs, False, 'pv')
                             R_ls = next_value
                    else:
                        # Terminal state has value 0
                        if hasattr(traffic_agent, 'n_agent'):
                            R_ls = [0] * traffic_agent.n_agent
                        else:
                            R_ls = 0
                    
                    traffic_agent.backward(R_ls, 
                                           summary_writer=summary_writer, 
                                           global_step=global_traffic_step)

            # Update Loop Vars
            segment_traffic_reward += np.sum(step_rewards)
            traffic_obs = next_traffic_obs
            global_traffic_step += 1
            
            if done:
                break

        # 4. Calculate Adversary Result
        # Reward = Negative Traffic Reward (Zero-Sum Game)
        adv_reward = -segment_traffic_reward / 100.0
        
        if done:
            next_adv_obs = np.zeros(self.n_s_adversary)
        else:
            next_adv_obs = self._get_adversary_state()

        info = {'global_traffic_step': global_traffic_step}
        
        return next_adv_obs, adv_reward, done, info