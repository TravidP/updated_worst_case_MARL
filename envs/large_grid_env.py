"""
Particular class of large traffic grid
@author: Tianshu Chu
"""

import configparser
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
from envs.env import PhaseMap, PhaseSet, TrafficSimulator
from large_grid.data.build_file import gen_rou_file
import traci
import pandas as pd

sns.set_color_codes()


STATE_NAMES = ['wave', 'wait']
PHASE_NUM = 5
SCENARIO_DURATION = 600  # 10 minutes per CSV group
# map from ild order (alphabeta) to signal order (clockwise from north)
# STATE_PHASE_MAP = {'nt1': [2, 3, 1, 0], 'nt2': [2, 3, 1, 0],
#                    'nt3': [2, 3, 1, 0], 'nt4': [2, 3, 1, 0],
#                    'nt5': [2, 1, 0, 3], 'nt6': [3, 2, 0, 1],
#                    'nt7': [0, 2, 3, 1], 'nt8': [0, 2, 3, 1],
#                    'nt9': [1, 0, 2, 3], 'nt10': [1, 0, 2, 3],
#                    'nt11': [3, 1, 0, 2], 'nt12': [3, 1, 0, 2],
#                    'nt13': [3, 1, 0, 2], 'nt14': [3, 1, 0, 2],
#                    'nt15': [1, 2, 3, 0], 'nt16': [3, 2, 1, 0],
#                    'nt17': [2, 3, 1, 0], 'nt18': [2, 3, 1, 0],
#                    'nt19': [2, 3, 1, 0], 'nt20': [1, 2, 3, 0],
#                    'nt21': [0, 3, 2, 1], 'nt22': [0, 2, 3, 1],
#                    'nt23': [0, 2, 3, 1], 'nt24': [0, 2, 3, 1],
#                    'nt25': [1, 0, 2, 3]}
# MAX_CAR_NUM = 30


class LargeGridPhase(PhaseMap):
    def __init__(self):
        phases = ['GGgrrrGGgrrr', 'rrrGrGrrrGrG', 'rrrGGrrrrGGr',
                  'rrrGGGrrrrrr', 'rrrrrrrrrGGG']
        self.phases = {PHASE_NUM: PhaseSet(phases)}


class LargeGridController:
    def __init__(self, node_names):
        self.name = 'greedy'
        self.node_names = node_names

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions

    def greedy(self, ob, node_name):
        # hard code the mapping from state to number of cars
        flows = [ob[0] + ob[3], ob[2] + ob[5], ob[1] + ob[4],
                 ob[1] + ob[2], ob[4] + ob[5]]
        return np.argmax(np.array(flows))


class LargeGridEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.peak_flow1 = config.getint('peak_flow1')
        self.peak_flow2 = config.getint('peak_flow2')
        self.init_density = config.getfloat('init_density')
        super().__init__(config, output_path, is_record, record_stat, port=port)
        # --- 1. Automatic Discovery of Traffic Scenarios ---
        traffic_data_dir = "./data_traffic"
        
        if not os.path.exists(traffic_data_dir):
            os.makedirs(traffic_data_dir)
            logging.warning(f"Created missing directory: {traffic_data_dir}")

        search_path = os.path.join(traffic_data_dir, "*.csv")
        self.csv_paths = sorted(glob.glob(search_path))
        
        self.scenarios = [] # List of dictionaries containing flow data
        self.loaded_filenames = []  # <--- ADD THIS LINE
        if not self.csv_paths:
            logging.warning(f"No traffic scenarios found in {traffic_data_dir}. Please add .csv files.")
        else:
            logging.info(f"Found {len(self.csv_paths)} adversarial scenarios: {self.csv_paths}")

        # --- 2. Load Scenarios into Memory ---
        for i, path in enumerate(self.csv_paths):
            try:
                df = pd.read_csv(path)
                
                if 'veh_per_hour' not in df.columns:
                    logging.error(f"File {path} skipped: missing 'veh_per_hour' column")
                    continue

                # Store as a list of dicts for faster iteration during injection
                # Each item: {'origin': str, 'dest': str, 'rate': float}
                scenario_data = []
                for _, row in df.iterrows():
                    scenario_data.append({
                        'origin': row['origin_edge'],
                        'dest': row['dest_edge'],
                        'rate': row['veh_per_hour']
                    })
                
                self.scenarios.append(scenario_data)
                self.loaded_filenames.append(os.path.basename(path).replace('.csv', ''))
                logging.info(f"Loaded Scenario {i} from {os.path.basename(path)}")
                
            except Exception as e:
                logging.error(f"Error reading {path}: {e}")
            # --- 3. OVERRIDE EPISODE LENGTH ---
            # We overwrite the parent's variable here
            self.num_scenarios = len(self.scenarios)
            self.episode_length_sec = self.num_scenarios * SCENARIO_DURATION
            
            logging.info(f"Loaded {self.num_scenarios} scenarios.")
            logging.info(f"Updated Episode Duration: {self.episode_length_sec} seconds ({self.episode_length_sec/60:.1f} mins).")

        # Route cache to prevent calling findRoute excessively
        self.route_cache = set()

    def _get_node_phase_id(self, node_name):
        return PHASE_NUM

    def _init_large_neighbor_map(self):
        neighbor_map = {}
        # corner nodes
        neighbor_map['nt1'] = ['nt6', 'nt2']
        neighbor_map['nt5'] = ['nt10', 'nt4']
        neighbor_map['nt21'] = ['nt22', 'nt16']
        neighbor_map['nt25'] = ['nt20', 'nt24']
        # edge nodes
        neighbor_map['nt2'] = ['nt7', 'nt3', 'nt1']
        neighbor_map['nt3'] = ['nt8', 'nt4', 'nt2']
        neighbor_map['nt4'] = ['nt9', 'nt5', 'nt3']
        neighbor_map['nt22'] = ['nt23', 'nt17', 'nt21']
        neighbor_map['nt23'] = ['nt24', 'nt18', 'nt22']
        neighbor_map['nt24'] = ['nt25', 'nt19', 'nt23']
        neighbor_map['nt10'] = ['nt15', 'nt5', 'nt9']
        neighbor_map['nt15'] = ['nt20', 'nt10', 'nt14']
        neighbor_map['nt20'] = ['nt25', 'nt15', 'nt19']
        neighbor_map['nt6'] = ['nt11', 'nt7', 'nt1']
        neighbor_map['nt11'] = ['nt16', 'nt12', 'nt6']
        neighbor_map['nt16'] = ['nt21', 'nt17', 'nt11']
        # internal nodes
        for i in [7, 8, 9, 12, 13, 14, 17, 18, 19]:
            n_node = 'nt' + str(i + 5)
            s_node = 'nt' + str(i - 5)
            w_node = 'nt' + str(i - 1)
            e_node = 'nt' + str(i + 1)
            cur_node = 'nt' + str(i)
            neighbor_map[cur_node] = [n_node, e_node, s_node, w_node]
        return neighbor_map

    def _init_large_distance_map(self):
        distance_map = {}
        # corner nodes
        distance_map['nt1'] = {'nt3':2, 'nt7':2, 'nt11':2,
                               'nt4':3, 'nt8':3, 'nt12':3, 'nt16':3,
                               'nt5':4, 'nt9':4, 'nt13':4, 'nt17':4, 'nt21':4,
                               'nt10':5, 'nt14':5, 'nt18':5, 'nt22':5}
        distance_map['nt5'] = {'nt3':2, 'nt9':2, 'nt15':2,
                               'nt2':3, 'nt8':3, 'nt14':3, 'nt20':3,
                               'nt1':4, 'nt7':4, 'nt13':4, 'nt19':4, 'nt25':4,
                               'nt6':5, 'nt12':5, 'nt18':5, 'nt24':5}
        distance_map['nt21'] = {'nt11':2, 'nt17':2, 'nt23':2,
                                'nt6':3, 'nt12':3, 'nt18':3, 'nt24':3,
                                'nt1':4, 'nt7':4, 'nt13':4, 'nt19':4, 'nt25':4,
                                'nt2':5, 'nt8':5, 'nt14':5, 'nt20':5}
        distance_map['nt25'] = {'nt15':2, 'nt19':2, 'nt23':2,
                                'nt10':3, 'nt14':3, 'nt18':3, 'nt22':3,
                                'nt5':4, 'nt9':4, 'nt13':4, 'nt17':4, 'nt21':4,
                                'nt4':5, 'nt8':5, 'nt12':5, 'nt16':5}
        # edge nodes
        distance_map['nt2'] = {'nt4':2, 'nt6':2, 'nt8':2, 'nt12':2,
                               'nt5':3, 'nt9':3, 'nt11':3, 'nt13':3, 'nt17':3,
                               'nt10':4, 'nt14':4, 'nt16':4, 'nt18':4, 'nt22':4,
                               'nt15':5, 'nt19':5, 'nt21':5, 'nt23':5}
        distance_map['nt3'] = {'nt1':2, 'nt5':2, 'nt7':2, 'nt9':2, 'nt13':2,
                               'nt6':3, 'nt10':3, 'nt12':3, 'nt14':3, 'nt18':3,
                               'nt11':4, 'nt15':4, 'nt17':4, 'nt19':4, 'nt23':4,
                               'nt16':5, 'nt20':5, 'nt22':5, 'nt24':5}
        distance_map['nt4'] = {'nt2':2, 'nt8':2, 'nt10':2, 'nt14':2,
                               'nt1':3, 'nt7':3, 'nt13':3, 'nt15':3, 'nt19':3,
                               'nt6':4, 'nt12':4, 'nt18':4, 'nt20':4, 'nt24':4,
                               'nt11':5, 'nt17':5, 'nt23':5, 'nt25':5}
        distance_map['nt22'] = {'nt12':2, 'nt16':2, 'nt18':2, 'nt24':2,
                                'nt7':3, 'nt11':3, 'nt13':3, 'nt19':3, 'nt25':3,
                                'nt2':4, 'nt6':4, 'nt8':4, 'nt14':4, 'nt20':4,
                                'nt1':5, 'nt3':5, 'nt9':5, 'nt15':5}
        distance_map['nt23'] = {'nt13':2, 'nt17':2, 'nt19':2, 'nt21':2, 'nt25':2,
                                'nt8':3, 'nt12':3, 'nt14':3, 'nt16':3, 'nt20':3,
                                'nt3':4, 'nt7':4, 'nt9':4, 'nt11':4, 'nt15':4,
                                'nt2':5, 'nt4':5, 'nt6':5, 'nt10':5}
        distance_map['nt24'] = {'nt14':2, 'nt18':2, 'nt20':2, 'nt22':2,
                                'nt9':3, 'nt13':3, 'nt15':3, 'nt17':3, 'nt21':3,
                                'nt4':4, 'nt8':4, 'nt10':4, 'nt12':4, 'nt16':4,
                                'nt3':5, 'nt5':5, 'nt7':5, 'nt11':5}
        distance_map['nt10'] = {'nt4':2, 'nt8':2, 'nt14':2, 'nt20':2,
                                'nt3':3, 'nt7':3, 'nt13':3, 'nt19':3, 'nt25':3,
                                'nt2':4, 'nt6':4, 'nt12':4, 'nt18':4, 'nt24':4,
                                'nt1':5, 'nt11':5, 'nt17':5, 'nt23':5}
        distance_map['nt15'] = {'nt5':2, 'nt9':2, 'nt13':2, 'nt19':2, 'nt25':2,
                                'nt4':3, 'nt8':3, 'nt12':3, 'nt18':3, 'nt24':3,
                                'nt3':4, 'nt7':4, 'nt11':4, 'nt13':4, 'nt23':4,
                                'nt2':5, 'nt6':5, 'nt16':5, 'nt22':5}
        distance_map['nt20'] = {'nt10':2, 'nt14':2, 'nt18':2, 'nt24':2,
                                'nt5':3, 'nt9':3, 'nt13':3, 'nt17':3, 'nt23':3,
                                'nt4':4, 'nt8':4, 'nt12':4, 'nt16':4, 'nt22':4,
                                'nt3':5, 'nt7':5, 'nt11':5, 'nt21':5}
        distance_map['nt6'] = {'nt2':2, 'nt8':2, 'nt12':2, 'nt16':2,
                               'nt3':3, 'nt9':3, 'nt13':3, 'nt17':3, 'nt21':3,
                               'nt4':4, 'nt10':4, 'nt14':4, 'nt18':4, 'nt22':4,
                               'nt5':5, 'nt15':5, 'nt19':5, 'nt23':5}
        distance_map['nt11'] = {'nt1':2, 'nt7':2, 'nt13':2, 'nt17':2, 'nt21':2,
                                'nt2':3, 'nt8':3, 'nt14':3, 'nt18':3, 'nt22':3,
                                'nt3':4, 'nt9':4, 'nt15':4, 'nt19':4, 'nt23':4,
                                'nt4':5, 'nt10':5, 'nt20':5, 'nt24':5}
        distance_map['nt16'] = {'nt2':2, 'nt8':2, 'nt12':2, 'nt16':2,
                                'nt3':3, 'nt9':3, 'nt13':3, 'nt17':3, 'nt21':3,
                                'nt4':4, 'nt10':4, 'nt14':4, 'nt18':4, 'nt22':4,
                                'nt5':5, 'nt15':5, 'nt19':5, 'nt23':5}
        # internal nodes
        distance_map['nt7'] = {'nt1':2, 'nt3':2, 'nt9':2, 'nt11':2, 'nt13':2, 'nt17':2,
                               'nt4':3, 'nt10':3, 'nt14':3, 'nt16':3, 'nt18':3, 'nt22':3,
                               'nt5':4, 'nt15':4, 'nt19':4, 'nt21':4, 'nt23':4,
                               'nt20':5, 'nt24':5}
        distance_map['nt8'] = {'nt2':2, 'nt4':2, 'nt6':2, 'nt10':2, 'nt12':2, 'nt14':2, 'nt18':2,
                               'nt1':3, 'nt5':3, 'nt11':3, 'nt15':3, 'nt17':3, 'nt19':3, 'nt23':3,
                               'nt16':4, 'nt20':4, 'nt22':4, 'nt24':4,
                               'nt21':5, 'nt25':5}
        distance_map['nt9'] = {'nt3':2, 'nt5':2, 'nt7':2, 'nt13':2, 'nt15':2, 'nt19':2,
                               'nt2':3, 'nt6':3, 'nt12':3, 'nt18':3, 'nt20':3, 'nt24':3,
                               'nt1':4, 'nt11':4, 'nt17':4, 'nt23':4, 'nt25':4,
                               'nt16':5, 'nt22':5}
        distance_map['nt12'] = {'nt2':2, 'nt6':2, 'nt8':2, 'nt14':2, 'nt16':2, 'nt18':2, 'nt22':2,
                                'nt1':3, 'nt3':3, 'nt9':3, 'nt15':3, 'nt19':3, 'nt21':3, 'nt23':3,
                                'nt4':4, 'nt10':4, 'nt20':4, 'nt24':4,
                                'nt5':5, 'nt25':5}
        distance_map['nt13'] = {'nt3':2, 'nt7':2, 'nt9':2, 'nt11':2, 'nt15':2, 'nt17':2, 'nt19':2, 'nt23':2,
                                'nt2':3, 'nt4':3, 'nt6':3, 'nt10':3, 'nt16':3, 'nt20':3, 'nt22':3, 'nt24':3,
                                'nt1':4, 'nt5':4, 'nt21':4, 'nt25':4}
        distance_map['nt14'] = {'nt4':2, 'nt8':2, 'nt10':2, 'nt12':2, 'nt18':2, 'nt20':2, 'nt24':2,
                                'nt3':3, 'nt5':3, 'nt7':3, 'nt11':3, 'nt17':3, 'nt23':3, 'nt25':3,
                                'nt2':4, 'nt6':4, 'nt16':4, 'nt22':4,
                                'nt1':5, 'nt21':5}
        distance_map['nt17'] = {'nt7':2, 'nt11':2, 'nt13':2, 'nt19':2, 'nt21':2, 'nt23':2,
                                'nt2':3, 'nt6':3, 'nt8':3, 'nt14':3, 'nt20':3, 'nt24':3,
                                'nt1':4, 'nt3':4, 'nt9':4, 'nt15':4, 'nt25':4,
                                'nt4':5, 'nt10':5}
        distance_map['nt18'] = {'nt8':2, 'nt12':2, 'nt14':2, 'nt16':2, 'nt20':2, 'nt22':2, 'nt24':2,
                                'nt3':3, 'nt7':3, 'nt9':3, 'nt11':3, 'nt15':3, 'nt21':3, 'nt25':3,
                                'nt2':4, 'nt4':4, 'nt6':4, 'nt10':4,
                                'nt1':5, 'nt5':5}
        distance_map['nt19'] = {'nt9':2, 'nt13':2, 'nt15':2, 'nt17':2, 'nt23':2, 'nt25':2,
                                'nt4':3, 'nt8':3, 'nt10':3, 'nt12':3, 'nt16':3, 'nt22':3,
                                'nt3':4, 'nt5':4, 'nt7':4, 'nt11':4, 'nt21':4,
                                'nt2':5, 'nt6':5}
        return distance_map

    def _init_map(self):
        self.neighbor_map = self._init_large_neighbor_map()
        # for spatial discount
        self.distance_map = self._init_large_distance_map()
        self.max_distance = 6
        self.phase_map = LargeGridPhase()
        self.state_names = STATE_NAMES

    def _init_sim_config(self, seed):
        # return gen_rou_file(self.data_path,
        #                     self.peak_flow1,
        #                     self.peak_flow2,
        #                     self.init_density,
        #                     seed=seed,
        #                     thread=self.sim_thread)
        route_file_path = self.data_path + f'expdummy_{self.sim_thread}.rou.xml'
        with open(route_file_path, 'w') as f:
            f.write('<routes>\n')
            f.write('  <vType id="type1" length="5" accel="5" decel="10"/>\n')
            f.write('</routes>\n')
        # We still need the sumocfg file
        from large_grid.data.build_file import output_config, write_file
        sumocfg_file = self.data_path + f'expdummy_{self.sim_thread}.sumocfg'
        write_file(sumocfg_file, output_config(thread=self.sim_thread))
        return sumocfg_file
    
    def _inject_scenario_traffic(self, scenario_idx, current_time):
        """
        Generates and adds vehicles for the entire 10-minute block 
        using TraCI's future departure capability.
        """
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
            if route_id in self.route_cache:
                valid_route = True
            else:
                # 2. ONLY CALL SUMO IF UNKNOWN
                try:
                    # Ask SUMO to find the path edges
                    route_stage = self.sim.simulation.findRoute(origin, dest, vType="type1")
                    
                    if route_stage and route_stage.edges:
                        # Register the route in SUMO
                        self.sim.route.add(route_id, route_stage.edges)
                        
                        # Add to local cache so we never calculate this OD pair again
                        self.route_cache.add(route_id)
                        valid_route = True
                    else:
                        # Optional: Cache invalid routes too, to stop trying to find them?
                        # self.route_cache.add(route_id) # Treat as known but valid_route stays False
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

    # def _spawn_vehicles_from_od(self):
    #     """Inject vehicles based on current simulation second."""
    #     if self.cur_sec in self.od_grouped.groups:
    #         vehicles_now = self.od_grouped.get_group(self.cur_sec)
            
    #         for idx, row in vehicles_now.iterrows():
    #             veh_id = f"adv_{self.cur_sec}_{idx}"
    #             origin_edge = row['origin_edge'] 
    #             dest_edge = row['dest_edge']     
                
    #             # 1. FIX: Use self.sim.simulation instead of traci.simulation
    #             route_result = self.sim.simulation.findRoute(origin_edge, dest_edge, vType="type1")
                
    #             if route_result and route_result.edges:
    #                 route_id = f"route_{veh_id}"
                    
    #                 # 2. FIX: Use self.sim.route instead of traci.route
    #                 self.sim.route.add(route_id, route_result.edges)
                    
    #                 # 3. FIX: Use self.sim.vehicle instead of traci.vehicle
    #                 self.sim.vehicle.add(
    #                     vehID=veh_id,
    #                     routeID=route_id,
    #                     typeID="type1",
    #                     departSpeed=str(row['speed']), 
    #                     departLane="free"
    #                 )
    #             else:
    #                 logging.warning(f"Could not find route from {origin_edge} to {dest_edge}")

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
                # No more scenarios, but we continue simulation until done
                pass

        # 2. Standard SUMO Stepping
        for _ in range(num_step):
            self.sim.simulationStep()
            self.cur_sec += 1
            if self.is_record:
                self._measure_traffic_step()

    def reset(self, gui=False, test_ind=0):
        # Reset internal scenario tracking
        self.current_scenario_idx = 0
        self.next_switch_time = 0
        self.route_cache = set()

        return super().reset(gui=gui, test_ind=test_ind)
    
    def step(self, action):
        # We can now rely on the parent or standard checks for done
        # provided they check against self.episode_length_sec
        next_obs, reward, done, info = super().step(action)

        # Explicit safety check: ensure we don't finish until full duration
        if self.cur_sec < self.episode_length_sec:
            done = False
        else:
            done = True
            
        return next_obs, reward, done, info
    
    # def _init_sim_traffic(self):
    #     lanes = self.sim.lane.getIDList()
    #     internal_lanes = []
    #     external_lanes = []
    #     for lane in lanes:
    #         tokens = lane.split('_')[0].split(':')[1].split(',')
    #         if not tokens[0].startswith('nt'):
    #             continue
    #         if tokens[1].startswith('nt'):
    #             internal_lanes.append(lane)
    #         elif tokens[1].startswith('np'):
    #             external_lanes.append(lane)
    #     init_car_num = int(MAX_CAR_NUM * self.init_density)
    #     i = 1
    #     for lane in internal_lanes:
    #         for _ in range(init_car_num):
    #             dest_lane = np.random.choice(external_lanes)
    #             car_id = 'init_car_%d' % i
    #             self.sim.vehicle.add(car_id, route_id, typeID='type1', depart=0, departLane=lane,
    #                                  departPos='random_free', departSpeed=5, arrivalLane=dest_lane)


    def plot_stat(self, rewards):
        self.state_stat['reward'] = rewards
        for name, data in self.state_stat.items():
            fig = plt.figure(figsize=(8, 6))
            plot_cdf(data)
            plt.ylabel(name)
            fig.savefig(self.output_path + self.name + '_' + name + '.png')


def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, color=c, label=label)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO)
    config = configparser.ConfigParser()
    config.read('./config/config_test_large.ini')
    base_dir = './output_result/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    env = LargeGridEnv(config['ENV_CONFIG'], 2, base_dir, is_record=True, record_stat=True)
    env.train_mode = False
    time.sleep(2)
    ob = env.reset()
    controller = LargeGridController(env.node_names)
    rewards = []
    while True:
        next_ob, _, done, reward = env.step(controller.forward(ob))
        rewards.append(reward)
        if done:
            break
        ob = next_ob
    env.plot_stat(np.array(rewards))
    logging.info('avg reward: %.2f' % np.mean(rewards))
    env.terminate()
    time.sleep(2)
    env.collect_tripinfo()
    env.output_data()
