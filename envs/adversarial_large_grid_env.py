import pandas as pd
import numpy as np
import logging
import traci
import os
import glob
import socket
import time
from envs.large_grid_env import LargeGridEnv
from agents.models import A2C, IA2C, MA2C, IQL, PPO
from tf_compat import tf
import configparser
from utils import find_file

class AdversarialLargeGridEnv(LargeGridEnv):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False, 
                 frozen_model_dir=None):
        
        super().__init__(config, port, output_path, is_record, record_stat)
        
# --- 1. Load CSV Scenarios Automatically as Flow Rates ---
        # --- 2. Process Loaded Scenarios for Adversary ---
        # Instead of re-reading CSVs, we transform the parent's data structures
        # into the format required for weight-based injection.
        self.flow_groups = {} 
        self.group_names = []
        
        # Safety check to ensure parent loaded data correctly
        if not hasattr(self, 'scenarios') or not self.scenarios:
            logging.warning("AdversarialEnv: No scenarios were loaded by the parent class.")
        else:
            # Iterate through the data already in memory
            for i, (scenario_data, filename) in enumerate(zip(self.scenarios, self.loaded_filenames)):
                
                # Construct a unique group name (e.g., "g0_traffic_peak")
                group_name = f"g{i}_{filename}"
                self.group_names.append(group_name)
                
                # Transform Data Structure:
                # Parent (Time-based): [{'origin': u, 'dest': v, 'rate': r}, ...]
                # Child (Weight-based): {(u, v): r, ...}
                counts = {}
                for item in scenario_data:
                    # Create tuple key for O(1) lookup during injection
                    key = (item['origin'], item['dest'])
                    counts[key] = item['rate']
                
                self.flow_groups[group_name] = counts
                
                # Log for verification
                total_flow = sum(counts.values())
                logging.info(f"Adversarial Group Init: Mapped '{group_name}' | Total Flow: {total_flow:.1f} vph")
        # --- 2. Load Frozen Signal Controller ---
        if frozen_model_dir:
            self._load_frozen_controller(frozen_model_dir)
        else:
            raise ValueError("frozen_model_dir is required!")
        
        # --- 3. Configuration ---
        self.n_adversary_action = len(self.group_names)
        self.adversary_step_duration = 600  # 10 Minutes
        self.adversary_reward_scale = 100.0
        self._init_adversary_state_space()
        self.known_routes = set()

    def _init_adversary_state_space(self):
        self.n_s_adversary = 0
        for node_name in self.node_names:
            node = self.nodes[node_name]
            self.n_s_adversary += node.num_state * 2 # Wave + Wait
        logging.info(f"Adversary Global Observation Dim: {self.n_s_adversary}")

    def _get_adversary_state(self):
        global_obs = []
        sorted_node_names = sorted(self.node_names, key=lambda x: int(x[2:]))
        for node_name in sorted_node_names:
            node = self.nodes[node_name]
            global_obs.append(node.wave_state)
            global_obs.append(node.wait_state)
        return np.concatenate(global_obs)

    def _normalize_action_weights(self, actions):
        """Convert unconstrained adversary logits into a valid mixture."""
        logits = np.array(actions, dtype=np.float32).flatten()
        if logits.size == 0:
            raise ValueError("Adversary action is empty.")
        shifted = logits - np.max(logits)
        exp_logits = np.exp(np.clip(shifted, -50.0, 50.0))
        total = np.sum(exp_logits)
        if not np.isfinite(total) or total <= 1e-6:
            return np.ones_like(logits) / float(len(logits))
        return exp_logits / total
    
    def reset(self, gui=False, test_ind=0):
        # Route IDs live inside the current SUMO process only.
        # Clear per-episode cache so routes are rebuilt after each reset.
        self.known_routes = set()
        _ = super().reset(gui=gui, test_ind=test_ind)
        self.cur_sec = 0 
        return self._get_adversary_state()

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
            elif agent_type == 'ppo':
                self.frozen_controller = PPO(self.n_s_ls, self.n_a_ls, self.n_w_ls, 0, saved_config['MODEL_CONFIG'])
            elif agent_type == 'iqld':
                self.frozen_controller = IQL(self.n_s_ls, self.n_a_ls, self.n_w_ls, 0, saved_config['MODEL_CONFIG'], seed=0, model_type='dqn')
            # --- ADDED BLOCK FOR IQLL ---
            elif agent_type == 'iqll':
                # model_type='lr' triggers LRQPolicy in agents/models.py
                self.frozen_controller = IQL(self.n_s_ls, self.n_a_ls, self.n_w_ls, 0, saved_config['MODEL_CONFIG'], seed=0, model_type='lr')
            # -----------------------------
            
            self.frozen_controller.load(agent_dir + '/model/')
        logging.info("Frozen controller loaded.")

    def step(self, adversary_action):
        """
        Adversary Step (Continuous Control):
        1. Receive unconstrained action logits for each traffic group.
        2. Normalize with softmax into a valid probability distribution.
        3. Inject traffic and simulate.
        """
        # --- DEBUG: Print Adversary (Worst Case Estimator) Action ---
        logging.info("\n" + "="*40)
        logging.info(f"--- ADVERSARY STEP (Worst Case Estimator) ---")
        
        # 1. Process the unconstrained action vector.
        raw_weights = np.array(adversary_action, dtype=np.float32).flatten()
        
        logging.info(f"Raw Adversary Output: {raw_weights}")

        # Softmax keeps the WCE action mapping identical across standalone and co-evolution.
        weights = self._normalize_action_weights(raw_weights)
        
        # --- DEBUG: Print Final Scenario Weights ---
        weight_info = {name: f"{w:.2f}" for name, w in zip(self.group_names, weights)}
        logging.info(f"Final Mixed Weights: {weight_info}")
        logging.info("="*40 + "\n")

        # 2. Inject Traffic (Using findRoute)
        self._inject_dynamic_traffic(weights, duration=self.adversary_step_duration)

        # 3. Inner Simulation Loop
        steps_to_run = int(self.adversary_step_duration / self.control_interval_sec)
        segment_reward = 0
        done = False
        
        sorted_node_names = sorted(self.node_names, key=lambda x: int(x[2:]))

        for step_i in range(steps_to_run):
            # --- DEBUG LOGGING START ---
            # if step_i % 10 == 0:  # Log every 10 control steps
            #     logging.info(f"Simulating adversarial step {step_i}/{steps_to_run}...")
            # --- DEBUG LOGGING END ---
            # A. Get State
            controller_local_states = self._get_state()
            
            # B. Get Action (Frozen)
            with self.frozen_graph.as_default():
                with self.frozen_sess.as_default():
                    # IQL (including IQLD and IQLL) uses a different forward signature
                    if self.frozen_controller.name == 'iql':
                        # Returns (actions, q_values). We only need the actions.
                        # Signature: forward(self, obs, mode='act', stochastic=False)
                        signal_actions, _ = self.frozen_controller.forward(controller_local_states, mode='act')
                    
                    # A2C, IA2C, MA2C use the 'done' argument and return probabilities
                    else:
                        # Signature: forward(self, ob, done, out_type='pv')
                        policies = self.frozen_controller.forward(controller_local_states, done=False, out_type='p')
                        signal_actions = [np.random.choice(len(pi), p=pi) for pi in policies]

            # C. Simulation Step
            self._set_phase(signal_actions, 'yellow', self.yellow_interval_sec)
            self._simulate(self.yellow_interval_sec)
            
            rest = self.control_interval_sec - self.yellow_interval_sec
            self._set_phase(signal_actions, 'green', rest)
            self._simulate(rest)
            
            # D. Reward
            step_rewards = self._measure_reward_step()
            segment_reward += -np.sum(step_rewards) # Adversary wants to maximize delay
            
            if self.cur_sec >= self.episode_length_sec:
                done = True
                break

        scaled_reward = segment_reward / float(self.adversary_reward_scale)
        return self._get_adversary_state(), scaled_reward, done, {}



    def _inject_dynamic_traffic(self, weights, duration):
        """
        Calculates mixed flow, finds routes, and injects vehicles.
        """
        # current_time = self.sim.get_current_time()
        current_time = self.sim.simulation.getTime()
        requested_vehicles = 0
        scheduled_vehicles = 0
        skipped_route = 0
        vehicle_add_fail = 0
        active_od_pairs = 0
        
        # 1. Identify all unique OD pairs
        all_od_pairs = set()
        for g_data in self.flow_groups.values():
            all_od_pairs.update(g_data.keys())

        # 2. Iterate OD pairs
        for od_pair in all_od_pairs:
            origin, dest = od_pair
            
            # A. Calculate Mixed Rate
            mixed_rate = 0.0
            for idx, group_name in enumerate(self.group_names):
                group_flow = self.flow_groups[group_name].get(od_pair, 0.0)
                mixed_rate += weights[idx] * group_flow
            
            if mixed_rate < 0.1: continue

            # B. Calculate Quantity (Poisson)
            expected_n = (mixed_rate / 3600.0) * duration
            num_vehicles = np.random.poisson(expected_n)
            if num_vehicles == 0: continue
            active_od_pairs += 1
            requested_vehicles += int(num_vehicles)

            # --- C. FIND ROUTE (The Fix) ---
            # --- OPTIMIZED ROUTE HANDLING ---
            route_id = f"route_{origin}_{dest}"
            valid_route = False

            # CHECK LOCAL CACHE INSTEAD OF CALLING TRACI
            if route_id in self.known_routes:
                valid_route = True
            else:
                try:
                    # Only call SUMO if we haven't seen this route before
                    route_stage = self.sim.simulation.findRoute(origin, dest, vType="type1")
                    if route_stage and route_stage.edges:
                        self.sim.route.add(route_id, route_stage.edges)
                        self.known_routes.add(route_id) # Update local cache
                        valid_route = True
                except traci.TraCIException:
                    pass

            if not valid_route:
                skipped_route += int(num_vehicles)
                continue

            # D. Schedule Vehicles
            raw_times = np.random.uniform(current_time, current_time + duration, num_vehicles)
            jitter = np.random.normal(0, 2.0, num_vehicles) 
            depart_times = np.sort(np.maximum(raw_times + jitter, current_time + 1.0))

            for t in depart_times:
                veh_id = f"adv_{int(t)}_{origin}_{dest}_{np.random.randint(99999)}"
                try:
                    self.sim.vehicle.add(
                        vehID=veh_id, 
                        routeID=route_id, 
                        typeID="type1", 
                        depart=f"{t:.2f}"
                    )
                    self.sim.vehicle.setSpeedFactor(veh_id, np.random.normal(1.0, 0.1))
                    scheduled_vehicles += 1
                except traci.TraCIException:
                    vehicle_add_fail += 1

        logging.info(
            "Adversary traffic inject @t=%.0f (+%ds): active_od=%d requested=%d scheduled=%d "
            "route_skipped=%d add_failed=%d known_routes=%d",
            current_time, duration, active_od_pairs, requested_vehicles, scheduled_vehicles,
            skipped_route, vehicle_add_fail, len(self.known_routes)
        )

    def _simulate(self, num_step):
        self._validate_traci_connection()
        for _ in range(num_step):
            try:
                self.sim.simulationStep()
            except (socket.timeout, traci.TraCIException, OSError) as e:
                logging.error("Adversarial simulationStep failed at t=%d: %s", self.cur_sec, e)
                raise RuntimeError('Adversarial simulationStep failed')
            self.cur_sec += 1
            if self.is_record:
                self._measure_traffic_step()
            if self.sim_progress_log_sec > 0:
                if (self.cur_sec - self._last_progress_sim_sec) >= self.sim_progress_log_sec:
                    active_vehicles = -1
                    try:
                        active_vehicles = self.sim.vehicle.getIDCount()
                    except Exception:
                        pass
                    now = time.time()
                    wall_elapsed = now - self._last_progress_wall_sec
                    logging.info(
                        "Adversarial sim progress: t=%d/%d, active_vehicles=%d, wall=%.1fs for +%ds sim",
                        self.cur_sec, self.episode_length_sec, active_vehicles,
                        wall_elapsed, self.cur_sec - self._last_progress_sim_sec
                    )
                    self._last_progress_sim_sec = self.cur_sec
                    self._last_progress_wall_sec = now
