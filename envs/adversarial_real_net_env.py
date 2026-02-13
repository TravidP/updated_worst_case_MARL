import numpy as np
import logging
import pandas as pd
import os
import tensorflow as tf
import configparser
import traci
from envs.real_net_env import RealNetEnv, NODES
from agents.models import A2C, IA2C, MA2C, IQL
from utils import find_file

class AdversarialRealNetEnv(RealNetEnv):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False, 
                 frozen_model_dir=None):
        # 1. Initialize Base RealNetEnv
        super().__init__(config, port, output_path, is_record, record_stat)
        
        # 2. Convert Base Scenarios to Adversary Flow Groups
        self.flow_groups = {}
        self.group_names = []
        
        if not hasattr(self, 'scenarios') or not self.scenarios:
            logging.warning("No scenarios loaded from parent class! Adversary will have no actions.")
        else:
            for i, scenario_data in enumerate(self.scenarios):
                # Use filename as group name if available, else index
                if hasattr(self, 'loaded_filenames') and i < len(self.loaded_filenames):
                    group_name = self.loaded_filenames[i]
                else:
                    group_name = f"scenario_{i}"
                
                # Convert list of dicts -> dict of {(origin, dest): rate}
                counts = {}
                for item in scenario_data:
                    key = (item['origin'], item['dest'])
                    counts[key] = item['rate']
                
                self.flow_groups[group_name] = counts
                self.group_names.append(group_name)
                
                total_flow = sum(counts.values())
                logging.info(f"Adversary Group '{group_name}' initialized. Total Flow: {total_flow:.1f} vph")

        # 3. GCN Structure Setup (Centralized)
        self._setup_graph_for_gcn()

        # 4. Adversary Configuration
        self.n_adversary_action = len(self.group_names)
        self.adversary_step_duration = 600 # 10 Minutes
        self.cur_sec = 0

        # 5. Load Frozen Signal Controller (Optional)
        # If None, we assume we are in Co-Evolution mode or training from scratch
        self.frozen_controller = None
        if frozen_model_dir:
            self._load_frozen_controller(frozen_model_dir)
        else:
            logging.info("AdversarialEnv: No frozen model provided. Running in trainable/co-evolution mode.")

    def _setup_graph_for_gcn(self):
        """
        Prepares the graph structure (Adjacency Matrix) and Feature Dimensions for GCNA2C.
        """
        # A. Sort Nodes (Numeric sort preferred)
        def sort_key(x):
            return int(x) if x.isdigit() else 999999
        self.sorted_node_names = sorted(self.node_names, key=sort_key)
        self.num_nodes = len(self.sorted_node_names)
        self.node_to_idx = {name: i for i, name in enumerate(self.sorted_node_names)}
        
        # B. Determine Feature Dimension per Node (for Padding)
        # We need a fixed feature size per node for the GCN input tensor [Batch, Nodes, Feats]
        # RealNetEnv stores state sizes in self.n_s_ls (list of state dims for each node)
        if hasattr(self, 'n_s_ls'):
            self.adversary_feat_dim = max(self.n_s_ls)
        else:
            self.adversary_feat_dim = 30 # Fallback default
            
        # Total flattened state dimension for the Adversary
        self.n_s_adversary = self.num_nodes * self.adversary_feat_dim

        # C. Build Normalized Adjacency Matrix
        self.adj_matrix = self._build_normalized_adjacency()

    def _build_normalized_adjacency(self):
        """
        Builds normalized adjacency matrix (D^-0.5 * (A+I) * D^-0.5) for GCN.
        """
        adj = np.eye(self.num_nodes) # Start with Self-loops (A + I)
        
        for name, data in NODES.items():
            if name in self.node_to_idx:
                src_idx = self.node_to_idx[name]
                # data[1] is the neighbor list
                neighbors = data[1]
                for neighbor in neighbors:
                    if neighbor in self.node_to_idx:
                        dst_idx = self.node_to_idx[neighbor]
                        adj[src_idx, dst_idx] = 1.0
                        adj[dst_idx, src_idx] = 1.0 # Symmetric
        
        # Normalize: D^{-0.5} * A_hat * D^{-0.5}
        row_sum = np.sum(adj, axis=1)
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(row_sum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        
        return np.dot(np.dot(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    def _get_adversary_state(self):
        """
        Returns:
            features: Flattened vector of node features [Nodes * adversary_feat_dim]. 
            Pads local observations to ensure consistent feature size per node for GCN.
        """
        features = []
        
        # Get raw observations from env (list of arrays corresponding to self.node_names)
        raw_obs = self._get_state()
        
        # Map raw obs to sorted node order
        name_to_obs = dict(zip(self.node_names, raw_obs))
        
        for name in self.sorted_node_names:
            ob = name_to_obs[name]
            current_len = len(ob)
            target_len = self.adversary_feat_dim
            
            # Pad or truncate to fixed size 'target_len'
            if current_len < target_len:
                padded = np.pad(ob, (0, target_len - current_len), 'constant')
                features.append(padded)
            else:
                features.append(ob[:target_len])
                
        return np.concatenate(features)

    def _normalize_action_weights(self, actions):
        """Helper to normalize adversary actions into a probability distribution."""
        weights = np.array(actions, dtype=np.float32).flatten()
        weights = np.maximum(weights, 0.0)
        total = np.sum(weights)
        if total > 1e-6:
            return weights / total
        return np.ones_like(weights) / len(weights)

    def step(self, adversary_action):
        """
        Adversary Step (Static Agent Mode):
        1. Receive action vector (weights).
        2. Inject mixed traffic.
        3. Run simulation with FROZEN Agent.
        """
        if self.frozen_controller is None:
            raise ValueError("step() called in AdversarialRealNetEnv but no frozen_controller loaded. "
                             "If you are training traffic agents, use CoevolutionRealNetEnv.")

        # 1. Normalize Action
        weights = self._normalize_action_weights(adversary_action)
        
        # 2. Inject Traffic
        self._inject_dynamic_traffic(weights, self.adversary_step_duration)

        # 3. Simulate with Frozen Agent
        steps = int(self.adversary_step_duration / self.control_interval_sec)
        total_r = 0
        done = False
        
        for _ in range(steps):
            # Get state for frozen controller
            obs = self._get_state() 
            
            # Forward pass frozen agent
            with self.frozen_graph.as_default():
                with self.frozen_sess.as_default():
                    policies = self.frozen_controller.forward(obs, False, 'p')
            
            # Sample actions
            if isinstance(policies, list):
                actions = [np.random.choice(len(p), p=p) for p in policies]
            else: 
                actions = np.random.choice(len(policies), p=policies)

            # Sumo Step
            self._set_phase(actions, 'yellow', self.yellow_interval_sec)
            self._simulate(self.yellow_interval_sec)
            self._set_phase(actions, 'green', self.control_interval_sec - self.yellow_interval_sec)
            self._simulate(self.control_interval_sec - self.yellow_interval_sec)
            
            # Adversary Reward: Negative of Traffic Agent Reward
            step_r = np.sum(self._measure_reward_step())
            total_r += -step_r 

            if self.cur_sec >= self.episode_length_sec:
                done = True
                break
                
        return self._get_adversary_state(), total_r, done, {}

    def _simulate(self, num_step):
        for _ in range(num_step):
            self.sim.simulationStep()
            self.cur_sec += 1
            if self.is_record:
                self._measure_traffic_step()

    def _inject_dynamic_traffic(self, weights, duration):
        current_time = self.sim.simulation.getTime()
        
        all_od_pairs = set()
        for g_data in self.flow_groups.values():
            all_od_pairs.update(g_data.keys())

        for od_pair in all_od_pairs:
            origin, dest = od_pair
            
            mixed_rate = 0.0
            for idx, group_name in enumerate(self.group_names):
                group_flow = self.flow_groups[group_name].get(od_pair, 0.0)
                mixed_rate += weights[idx] * group_flow
            
            if mixed_rate < 0.1: continue

            expected_n = (mixed_rate / 3600.0) * duration
            num_vehicles = np.random.poisson(expected_n)
            if num_vehicles == 0: continue

            route_id = f"route_{origin}_{dest}"
            valid_route = False

            if route_id in self.route_cache:
                valid_route = True
            else:
                try:
                    route_stage = self.sim.simulation.findRoute(origin, dest, vType="type1")
                    if route_stage and route_stage.edges:
                        self.sim.route.add(route_id, route_stage.edges)
                        self.route_cache.add(route_id)
                        valid_route = True
                except traci.TraCIException:
                    pass

            if not valid_route: continue

            raw_times = np.random.uniform(current_time, current_time + duration, num_vehicles)
            jitter = np.random.normal(0, 2.0, num_vehicles) 
            depart_times = np.sort(np.maximum(raw_times + jitter, current_time + 1.0))

            for t in depart_times:
                veh_id = f"adv_{int(t)}_{origin}_{dest}_{np.random.randint(999999)}"
                try:
                    self.sim.vehicle.add(vehID=veh_id, routeID=route_id, typeID="type1", depart=f"{t:.2f}")
                    self.sim.vehicle.setSpeedFactor(veh_id, np.random.normal(1.0, 0.1))
                except traci.TraCIException:
                    pass

    def _load_frozen_controller(self, agent_dir):
        logging.info(f"Loading frozen controller from: {agent_dir}")
        config_path = find_file(agent_dir + '/data/')
        saved_config = configparser.ConfigParser()
        saved_config.read(config_path)
        agent_type = saved_config.get('ENV_CONFIG', 'agent')
        
        self.frozen_graph = tf.Graph()
        self.frozen_sess = tf.Session(graph=self.frozen_graph)

        with self.frozen_graph.as_default():
            if agent_type == 'a2c':
                self.frozen_controller = A2C(self.n_s, self.n_a, 0, saved_config['MODEL_CONFIG'])
            elif agent_type == 'ia2c':
                self.frozen_controller = IA2C(self.n_s_ls, self.n_a_ls, self.n_w_ls, 0, saved_config['MODEL_CONFIG'])
            elif agent_type == 'ma2c':
                self.frozen_controller = MA2C(self.n_s_ls, self.n_a_ls, self.n_w_ls, self.n_f_ls, 0, saved_config['MODEL_CONFIG'])
            elif agent_type == 'iqld':
                self.frozen_controller = IQL(self.n_s_ls, self.n_a_ls, self.n_w_ls, 0, saved_config['MODEL_CONFIG'], seed=0, model_type='dqn')
            elif agent_type == 'iql':
                self.frozen_controller = IQL(self.n_s_ls, self.n_a_ls, self.n_w_ls, 0, saved_config['MODEL_CONFIG'], seed=0, model_type='lr')
            
            self.frozen_controller.load(agent_dir + '/model/')
        logging.info("Frozen controller loaded.")