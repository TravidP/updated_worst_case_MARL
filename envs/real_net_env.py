"""
Particular class of real traffic network
@author: Tianshu Chu
"""

import configparser
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
import numpy as np
import pandas as pd
import traci
from envs.env import PhaseMap, PhaseSet, TrafficSimulator
from real_net.data.build_file import gen_rou_file
from real_net.data.build_file import output_config, write_file
import glob

sns.set_color_codes()
SCENARIO_DURATION = 600  # 10 minutes per scenario

STATE_NAMES = ['wave']
# node: (phase key, neighbor list)
NODES = {'10026': ('6.0', ['9431', '9561', 'cluster_9563_9597', '9531']),
         '8794': ('4.0', ['cluster_8985_9609', '9837', '9058', 'cluster_9563_9597']),
         '8940': ('2.1', ['9007', '9429']),
         '8996': ('2.2', ['cluster_9389_9689', '9713']),
         '9007': ('2.3', ['9309', '8940']),
         '9058': ('4.0', ['cluster_8985_9609', '8794', 'joinedS_0']),
         '9153': ('2.0', ['9643']),
         '9309': ('4.0', ['9466', '9007', 'cluster_9043_9052']),
         '9413': ('2.3', ['9721', '9837']),
         '9429': ('5.0', ['cluster_9043_9052', 'joinedS_1', '8940']),
         '9431': ('2.4', ['9721', '9884', '9561', '10026']),
         '9433': ('2.5', ['joinedS_1']),
         '9466': ('4.0', ['9309', 'joinedS_0', 'cluster_9043_9052']),
         '9480': ('2.3', ['8996', '9713']),
         '9531': ('2.6', ['joinedS_1', '10026']),
         '9561': ('4.0', ['cluster_9389_9689', '10026', '9431', '9884']),
         '9643': ('2.3', ['9153']),
         '9713': ('3.0', ['9721', '9884', '8996']),
         '9721': ('6.0', ['9431', '9713', '9413']),
         '9837': ('3.1', ['9413', '8794', 'cluster_8985_9609']),
         '9884': ('2.7', ['9713', '9431', 'cluster_9389_9689', '9561']),
         'cluster_8751_9630': ('4.0', ['cluster_9389_9689']),
         'cluster_8985_9609': ('4.0', ['9837', '8794', '9058']),
         'cluster_9043_9052': ('4.1', ['cluster_9563_9597', '9466', '9309', '10026', 'joinedS_1']),
         'cluster_9389_9689': ('4.0', ['9884', '9561', 'cluster_8751_9630', '8996']),
         'cluster_9563_9597': ('4.2', ['10026', '8794', 'joinedS_0', 'cluster_9043_9052']),
         'joinedS_0': ('6.1', ['9058', 'cluster_9563_9597', '9466']),
         'joinedS_1': ('3.2', ['9531', '9429'])}

PHASES = {'4.0': ['GGgrrrGGgrrr', 'rrrGGgrrrGGg', 'rrGrrrrrGrrr', 'rrrrrGrrrrrG'],
          '4.1': ['GGgrrGGGrrr', 'rrGrrrrrrrr', 'rrrGgrrrGGg', 'rrrrGrrrrrG'],
          '4.2': ['GGGGrrrrrrrr', 'GGggrrGGggrr', 'rrrGGGGrrrrr', 'grrGGggrrGGg'],
          '2.0': ['GGrrr', 'ggGGG'],
          '2.1': ['GGGrrr', 'rrGGGg'],
          '2.2': ['Grr', 'gGG'],
          '2.3': ['GGGgrr', 'GrrrGG'],
          '2.4': ['GGGGrr', 'rrrrGG'],
          '2.5': ['Gg', 'rG'],
          '2.6': ['GGGg', 'rrrG'],
          '2.7': ['GGg', 'rrG'],
          '3.0': ['GGgrrrGGg', 'rrGrrrrrG', 'rrrGGGGrr'],
          '3.1': ['GgrrGG', 'rGrrrr', 'rrGGGr'],
          '3.2': ['GGGGrrrGG', 'rrrrGGGGr', 'GGGGrrGGr'],
          '5.0': ['GGGGgrrrrGGGggrrrr', 'grrrGrrrrgrrGGrrrr', 'GGGGGrrrrrrrrrrrrr',
                  'rrrrrrrrrGGGGGrrrr', 'rrrrrGGggrrrrrggGg'],
          '6.0': ['GGGgrrrGGGgrrr', 'rrrGrrrrrrGrrr', 'GGGGrrrrrrrrrr', 'rrrrrrrrrrGGGG',
                  'rrrrGGgrrrrGGg', 'rrrrrrGrrrrrrG'],
          '6.1': ['GGgrrGGGrrrGGGgrrrGGGg', 'rrGrrrrrrrrrrrGrrrrrrG', 'GGGrrrrrGGgrrrrGGgrrrr',
                  'GGGrrrrrrrGrrrrrrGrrrr', 'rrrGGGrrrrrrrrrrrrGGGG', 'rrrGGGrrrrrGGGgrrrGGGg']}


class RealNetPhase(PhaseMap):
    def __init__(self):
        self.phases = {}
        for key, val in PHASES.items():
            self.phases[key] = PhaseSet(val)


class RealNetController:
    def __init__(self, node_names, nodes):
        self.name = 'greedy'
        self.node_names = node_names
        self.nodes = nodes

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions

    def greedy(self, ob, node_name):
        # get the action space
        phases = PHASES[NODES[node_name][0]]
        flows = []
        node = self.nodes[node_name]
        # get the green waves
        for phase in phases:
            wave = 0
            visited_ilds = set()
            for i, signal in enumerate(phase):
                if signal == 'G':
                    # find controlled lane
                    lane = node.lanes_in[i]
                    # ild = 'ild:' + lane
                    ild = lane
                    # if it has not been counted, add the wave
                    if ild not in visited_ilds:
                        j = node.ilds_in.index(ild)
                        wave += ob[j]
                        visited_ilds.add(ild)
            flows.append(wave)
        return np.argmax(np.array(flows))


class RealNetEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.flow_rate = config.getint('flow_rate')
        super().__init__(config, output_path, is_record, record_stat, port=port)
        # --- Scenario Initialization ---
        self.route_cache = set()
        self.current_scenario_idx = 0
        self.next_switch_time = 0
        self.scenarios = self._load_scenarios()
        self.num_scenarios = len(self.scenarios)
        self.episode_length_sec = self.num_scenarios * SCENARIO_DURATION
        logging.info(f"Loaded {len(self.scenarios)} scenarios.")

    def _load_scenarios(self):
        """
        Automatically discovers and loads traffic CSVs from ./data_traffic/
        """
        traffic_data_dir = "./data_traffic_real/"
        
        # Ensure directory exists
        if not os.path.exists(traffic_data_dir):
            os.makedirs(traffic_data_dir)
            logging.warning(f"Created missing directory: {traffic_data_dir}")

        # Automatic discovery using glob
        search_path = os.path.join(traffic_data_dir, "*.csv")
        self.csv_paths = sorted(glob.glob(search_path))
        
        scenarios = []
        self.loaded_filenames = []  # To track filenames corresponding to scenarios
        
        if not self.csv_paths:
            logging.warning(f"No traffic scenarios found in {traffic_data_dir}. Please add .csv files.")
            return scenarios
        else:
            logging.info(f"Found {len(self.csv_paths)} scenario files: {[os.path.basename(p) for p in self.csv_paths]}")
        
        # Load each file
        for file_path in self.csv_paths:
            try:
                df = pd.read_csv(file_path)
                # Convert DataFrame to list of dictionaries for faster iteration
                scenario_data = []
                for _, row in df.iterrows():
                    scenario_data.append({
                        'origin': row['origin_edge'],
                        'dest': row['dest_edge'],
                        'rate': float(row['veh_per_hour'])
                    })
                scenarios.append(scenario_data)
                self.loaded_filenames.append(os.path.basename(file_path))
                logging.info(f"Loaded scenario from {os.path.basename(file_path)}: {len(scenario_data)} flows.")
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
                
        return scenarios

    def _get_node_phase_id(self, node_name):
        return self.phase_node_map[node_name]

    def _init_neighbor_map(self):
        return dict([(key, val[1]) for key, val in NODES.items()])

    def _init_map(self):
        self.neighbor_map = self._init_neighbor_map()
        self.phase_map = RealNetPhase()
        self.phase_node_map = dict([(key, val[0]) for key, val in NODES.items()])
        self.state_names = STATE_NAMES

    def _init_sim_config(self, seed):
        """
        Generates a shared dummy route file and a specific sumocfg file for this thread.
        Files are only created if they do not exist, preventing repetitive I/O.
        """
        # 1. Define the shared dummy route file path
        # This file can be shared across all threads/experiments
        route_file_name = 'dummy.rou.xml'
        route_file_path = os.path.join(self.data_path, 'in', route_file_name)
        
        # 2. Create the dummy route file ONLY if it doesn't exist
        if not os.path.exists(route_file_path):
            os.makedirs(os.path.dirname(route_file_path), exist_ok=True)
            with open(route_file_path, 'w') as f:
                f.write('<routes>\n')
                f.write('  <vType id="type1" length="5" accel="5" decel="10" speedDev="0.1"/>\n')
                f.write('</routes>\n')
            logging.info(f"Created shared dummy route file: {route_file_path}")

        # 3. Define the sumocfg file path (unique per thread to avoid lock conflicts)
        sumocfg_name = 'most_%d.sumocfg' % self.sim_thread
        sumocfg_file = os.path.join(self.data_path, sumocfg_name)

        # 4. Create the sumocfg file ONLY if it doesn't exist
        if not os.path.exists(sumocfg_file):
            # Manually construct config to point to the shared 'dummy.rou.xml'
            str_config = '<configuration>\n  <input>\n'
            str_config += '    <net-file value="in/most.net.xml"/>\n'
            str_config += f'    <route-files value="in/{route_file_name}"/>\n' # Use shared dummy
            str_config += '    <additional-files value="in/most.add.xml"/>\n'
            str_config += '  </input>\n  <time>\n'
            str_config += '    <begin value="0"/>\n    <end value="3600"/>\n'
            str_config += '  </time>\n</configuration>\n'
            
            write_file(sumocfg_file, str_config)
            logging.info(f"Created config file: {sumocfg_file}")
            
        return sumocfg_file

    def plot_stat(self, rewards):
        self.state_stat['reward'] = rewards
        for name, data in self.state_stat.items():
            fig = plt.figure(figsize=(8, 6))
            plot_cdf(data)
            plt.ylabel(name)
            fig.savefig(self.output_path + self.name + '_' + name + '.png')

    def _inject_scenario_traffic(self, scenario_idx, current_time):
        """
        Generates and adds vehicles for the entire 10-minute block 
        using TraCI's future departure capability.
        """
        if scenario_idx >= len(self.scenarios):
            return

        flow_data = self.scenarios[scenario_idx]
        duration = SCENARIO_DURATION
        
        count_spawned = 0

        for item in flow_data:
            origin = item['origin']
            dest = item['dest']
            rate = item['rate']
            
            # --- Poisson Generation ---
            expected_n = (rate / 3600.0) * duration
            num_vehicles = np.random.poisson(expected_n)
            
            if num_vehicles == 0: 
                continue

            # --- Route Handling (Cached) ---
            route_id = f"route_{origin}_{dest}"
            
            # Check cache to avoid costly TraCI calls
            valid_route = False
            if route_id in self.route_cache:
                valid_route = True
            else:
                try:
                    # Ask SUMO to find the path edges
                    # Use self.sim.simulation instead of traci.simulation
                    route_stage = self.sim.simulation.findRoute(origin, dest, vType="type1")
                    
                    if route_stage and route_stage.edges:
                        # Register the route in SUMO
                        self.sim.route.add(route_id, route_stage.edges)
                        
                        # Add to local cache so we never calculate this OD pair again
                        self.route_cache.add(route_id)
                        valid_route = True
                    else:
                        # Could cache failures as well to optimize
                        pass
                except traci.TraCIException as e:
                    # logging.warning(f"Route fail: {e}")
                    pass

            if not valid_route:
                continue
            
            # --- Schedule Vehicles ---
            # Distribute departure times across the 10-minute window
            raw_times = np.random.uniform(current_time, current_time + duration, num_vehicles)
            
            # Add noise/jitter
            jitter = np.random.normal(0, 2.0, num_vehicles)
            # Ensure depart time is at least current_time + 1 (future)
            depart_times = np.sort(np.maximum(raw_times + jitter, current_time + 1.0))

            for t in depart_times:
                # Unique ID: scenario_time_origin_dest_random
                veh_id = f"g{scenario_idx}_{int(t)}_{origin}_{dest}_{np.random.randint(99999)}"
                try:
                    self.sim.vehicle.add(
                        vehID=veh_id, 
                        routeID=route_id, 
                        typeID="type1", 
                        depart=f"{t:.2f}"
                    )
                    # Randomize speed factor slightly for realism
                    self.sim.vehicle.setSpeedFactor(veh_id, np.random.normal(1.0, 0.1))
                    count_spawned += 1
                except traci.TraCIException:
                    pass
        
        logging.info(f"Scenario {scenario_idx}: Scheduled {count_spawned} vehicles.")

    def _simulate(self, num_step):
        """
        Overrides the standard simulation loop to inject traffic dynamically.
        """
        # 1. Check if we need to inject the next group (Scenario)
        if self.cur_sec >= self.next_switch_time:
            if self.current_scenario_idx < len(self.scenarios):
                logging.info(f"Injecting traffic for Scenario {self.current_scenario_idx} at t={self.cur_sec}")
                self._inject_scenario_traffic(self.current_scenario_idx, self.cur_sec)
                
                # Advance counters
                self.current_scenario_idx += 1
                self.next_switch_time += SCENARIO_DURATION
            else:
                # No more scenarios, loop traffic or just continue
                # Uncomment next line to loop scenarios:
                # self.current_scenario_idx = 0 
                pass

        # 2. Standard SUMO Stepping
        for _ in range(num_step):
            self.sim.simulationStep()
            self.cur_sec += 1
            if self.is_record:
                self._measure_traffic_step()


def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, color=c, label=label)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO)
    config = configparser.ConfigParser()
    config.read('./config/config_test_real.ini')
    base_dir = './output_result/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    env = RealNetEnv(config['ENV_CONFIG'], 2, base_dir, is_record=True, record_stat=True)
    env.train_mode = False
    time.sleep(1)
    # ob = env.reset(gui=True)
    controller = RealNetController(env.node_names, env.nodes)
    env.init_test_seeds(list(range(10000, 100001, 10000)))
    rewards = []
    for i in range(10):
        ob = env.reset(test_ind=i)
        global_rewards = []
        cur_step = 0
        while True:
            next_ob, reward, done, global_reward = env.step(controller.forward(ob))
            # for node_name, node_ob in zip(env.node_names, next_ob):
                # logging.info('%d, %s:%r\n' % (cur_step, node_name, node_ob))
            global_rewards.append(global_reward)
            rewards += list(reward)
            cur_step += 1
            if done:
                break
            ob = next_ob
        env.terminate()
        logging.info('step: %d, avg reward: %.2f' % (cur_step, np.mean(global_rewards)))
        time.sleep(1)
    env.plot_stat(np.array(rewards))
    env.terminate()
    time.sleep(2)
    env.collect_tripinfo()
    env.output_data()
