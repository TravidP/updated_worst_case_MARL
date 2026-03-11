"""
Traffic network simulator w/ defined sumo files
@author: Tianshu Chu
"""
import logging
import numpy as np
import os
import pandas as pd
import subprocess
from sumolib import checkBinary
import time
import traci
import xml.etree.cElementTree as ET
import socket
import threading

DEFAULT_PORT = 8000
SEC_IN_MS = 1000
TRACI_CONNECT_TIMEOUT_SEC = 20
TRACI_SOCKET_TIMEOUT_SEC = 180
TRACI_CLOSE_TIMEOUT_SEC = 15

# hard code real-net reward norm
REALNET_REWARD_NORM = 20

class PhaseSet:
    def __init__(self, phases):
        self.num_phase = len(phases)
        self.num_lane = len(phases[0])
        self.phases = phases
        # self._init_phase_set()

    @staticmethod
    def _get_phase_lanes(phase, signal='r'):
        phase_lanes = []
        for i, l in enumerate(phase):
            if l == signal:
                phase_lanes.append(i)
        return phase_lanes

    def _init_phase_set(self):
        self.red_lanes = []
        # self.green_lanes = []
        for phase in self.phases:
            self.red_lanes.append(self._get_phase_lanes(phase))
            # self.green_lanes.append(self._get_phase_lanes(phase, signal='G'))


class PhaseMap:
    def __init__(self):
        self.phases = {}

    def get_phase(self, phase_id, action):
        # phase_type is either green or yellow
        return self.phases[phase_id].phases[int(action)]

    def get_phase_num(self, phase_id):
        return self.phases[phase_id].num_phase

    def get_lane_num(self, phase_id):
        # the lane number is link number
        return self.phases[phase_id].num_lane

    def get_red_lanes(self, phase_id, action):
        # the lane number is link number
        return self.phases[phase_id].red_lanes[int(action)]


class Node:
    def __init__(self, name, neighbor=[], control=False):
        self.control = control # disabled
        # self.edges_in = []  # for reward
        self.lanes_in = []
        self.ilds_in = [] # for state
        self.fingerprint = [] # local policy
        self.name = name
        self.neighbor = neighbor
        self.num_state = 0 # wave and wait should have the same dim
        self.num_fingerprint = 0
        self.wave_state = [] # local state
        self.wait_state = [] # local state
        # self.waits = [] 
        self.phase_id = -1
        self.n_a = 0
        self.prev_action = -1


class TrafficSimulator:
    def __init__(self, config, output_path, is_record, record_stats, port=0):
        self.name = config.get('scenario')
        self.seed = config.getint('seed')
        self.control_interval_sec = config.getint('control_interval_sec')
        self.yellow_interval_sec = config.getint('yellow_interval_sec')
        self.episode_length_sec = config.getint('episode_length_sec')
        self.T = np.ceil(self.episode_length_sec / self.control_interval_sec)
        self.port = DEFAULT_PORT + port
        self.sim_thread = port
        self.obj = config.get('objective')
        self.data_path = config.get('data_path')
        if self.data_path and not self.data_path.endswith(os.sep):
            self.data_path += os.sep
        self.agent = config.get('agent')
        self.coop_gamma = config.getfloat('coop_gamma')
        self.cur_episode = 0
        self.norms = {'wave': config.getfloat('norm_wave'),
                      'wait': config.getfloat('norm_wait')}
        self.clips = {'wave': config.getfloat('clip_wave'),
                      'wait': config.getfloat('clip_wait')}
        self.coef_wait = config.getfloat('coef_wait')
        self.fast_wait_metric = config.getboolean('fast_wait_metric', fallback=False)
        self.sim_progress_log_sec = config.getint('sim_progress_log_sec', fallback=0)
        self.step_stage_warn_sec = config.getfloat('step_stage_warn_sec', fallback=15.0)
        self.stall_watchdog_sec = config.getint('stall_watchdog_sec', fallback=180)
        self.stall_watchdog_poll_sec = config.getint('stall_watchdog_poll_sec', fallback=10)
        self.trainer_stage_warn_sec = config.getfloat('trainer_stage_warn_sec', fallback=60.0)
        self.sim = None
        self.sumo_process = None
        self.sumo_log_fh = None
        self.sumo_log_file = None
        self._last_progress_sim_sec = 0
        self._last_progress_wall_sec = time.time()
        self._env_step_idx = 0
        if self.fast_wait_metric:
            logging.info('Using fast lane wait metric for state/reward computation.')
        if self.sim_progress_log_sec > 0:
            logging.info('Simulation heartbeat enabled every %d sim-seconds.', self.sim_progress_log_sec)
        logging.info(
            'Step probe: warn_if_stage_over=%.1fs, watchdog=%ss poll=%ss, trainer_warn=%.1fs',
            self.step_stage_warn_sec, self.stall_watchdog_sec, self.stall_watchdog_poll_sec,
            self.trainer_stage_warn_sec
        )
        self.train_mode = True
        test_seeds = config.get('test_seeds').split(',')
        test_seeds = [int(s) for s in test_seeds]
        self._init_map()
        self.init_data(is_record, record_stats, output_path)
        self.init_test_seeds(test_seeds)
        self._init_sim(self.seed)
        self._init_nodes()
        self.terminate()

    def _debug_traffic_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            phase = self.sim.trafficlight.getRedYellowGreenState(self.node_names[0])
            cur_traffic = {'episode': self.cur_episode,
                           'time_sec': self.cur_sec,
                           'node': node_name,
                           'action': node.prev_action,
                           'phase': phase}
            for i, ild in enumerate(node.ilds_in):
                cur_name = 'lane%d_' % i
                cur_traffic[cur_name + 'queue'] = self.sim.lane.getLastStepHaltingNumber(ild)
                cur_traffic[cur_name + 'flow'] = self.sim.lane.getLastStepVehicleNumber(ild)
                # cur_traffic[cur_name + 'wait'] = node.waits[i]
            self.traffic_data.append(cur_traffic)

    def _get_node_phase(self, action, node_name, phase_type):
        node = self.nodes[node_name]
        cur_phase = self.phase_map.get_phase(node.phase_id, action)
        if phase_type == 'green':
            return cur_phase
        prev_action = node.prev_action
        node.prev_action = action
        if (prev_action < 0) or (action == prev_action):
            return cur_phase
        prev_phase = self.phase_map.get_phase(node.phase_id, prev_action)
        switch_reds = []
        switch_greens = []
        for i, (p0, p1) in enumerate(zip(prev_phase, cur_phase)):
            if (p0 in 'Gg') and (p1 == 'r'):
                switch_reds.append(i)
            elif (p0 in 'r') and (p1 in 'Gg'):
                switch_greens.append(i)
        if not len(switch_reds):
            return cur_phase
        yellow_phase = list(cur_phase)
        for i in switch_reds:
            yellow_phase[i] = 'y'
        for i in switch_greens:
            yellow_phase[i] = 'r'
        return ''.join(yellow_phase)

    def _get_node_phase_id(self, node_name):
        # needs to be overwriteen
        raise NotImplementedError()

    def _get_node_state_num(self, node):
        assert len(node.lanes_in) == self.phase_map.get_lane_num(node.phase_id)
        # wait / wave states for each lane
        return len(node.ilds_in)

    def _get_state(self):
        # hard code the state ordering as wave, wait, fp
        state = []
        # measure the most recent state
        self._measure_state_step()

        # get the appropriate state vectors
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # wave is required in state
            if self.agent == 'greedy':
                state.append(node.wave_state)
            elif self.agent == 'a2c':
                if 'wait' in self.state_names:
                    state.append(np.concatenate([node.wave_state, node.wait_state]))
                else:
                    state.append(node.wave_state)
            else:
                cur_state = [node.wave_state]
                # include wave states of neighbors
                for nnode_name in node.neighbor:
                    if self.agent != 'ma2c':
                        cur_state.append(self.nodes[nnode_name].wave_state)
                    else:
                        # discount the neigboring states
                        cur_state.append(self.nodes[nnode_name].wave_state * self.coop_gamma)
                # include wait state
                if 'wait' in self.state_names:
                    cur_state.append(node.wait_state)
                # include fingerprints of neighbors
                if self.agent == 'ma2c':
                    for nnode_name in node.neighbor:
                        cur_state.append(self.nodes[nnode_name].fingerprint)
                state.append(np.concatenate(cur_state))

        if self.agent == 'a2c':
            state = np.concatenate(state)

        # # clean up the state and fingerprint measurements
        # for node in self.node_names:
        #     self.nodes[node].state = np.zeros(self.nodes[node].num_state)
        #     self.nodes[node].fingerprint = np.zeros(self.nodes[node].num_fingerprint)
        return state

    def _init_nodes(self):
        nodes = {}
        for node_name in self.sim.trafficlight.getIDList():
            if node_name in self.neighbor_map:
                neighbor = self.neighbor_map[node_name]
            else:
                logging.info('node %s can not be found!' % node_name)
                neighbor = []
            nodes[node_name] = Node(node_name,
                                    neighbor=neighbor,
                                    control=True)
            # controlled lanes: l:j,i_k
            lanes_in = self.sim.trafficlight.getControlledLanes(node_name)
            nodes[node_name].lanes_in = lanes_in
            # controlled edges: e:j,i
            # lane ilds: ild:j,i_k for road ji, lane k.
            # edges_in = []
            ilds_in = []
            for lane_name in lanes_in:
                ild_name = lane_name
                if ild_name not in ilds_in:
                    ilds_in.append(ild_name)
            # nodes[node_name].edges_in = edges_in
            nodes[node_name].ilds_in = ilds_in
        self.nodes = nodes
        self.node_names = sorted(list(nodes.keys()))
        s = 'Env: init %d node information:\n' % len(self.node_names)
        for node in self.nodes.values():
            s += node.name + ':\n'
            s += '\tneigbor: %r\n' % node.neighbor
            # s += '\tlanes_in: %r\n' % node.lanes_in
            s += '\tilds_in: %r\n' % node.ilds_in
            # s += '\tedges_in: %r\n' % node.edges_in
        logging.info(s)
        self._init_action_space()
        self._init_state_space()

    def _init_action_space(self):
        # for local and neighbor coop level
        self.n_a_ls = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            phase_id = self._get_node_phase_id(node_name)
            node.phase_id = phase_id
            node.n_a = self.phase_map.get_phase_num(phase_id)
            self.n_a_ls.append(node.n_a)
        # for global coop level
        self.n_a = np.prod(np.array(self.n_a_ls))

    def _init_map(self):
        # needs to be overwriteen
        self.neighbor_map = None
        self.phase_map = None
        self.state_names = None
        raise NotImplementedError()

    def _init_policy(self):
        policy = []
        for node_name in self.node_names:
            phase_num = self.nodes[node_name].n_a
            p = 1. / phase_num
            policy.append(np.array([p] * phase_num))
        return policy

    def _init_sim(self, seed, gui=False):
        sumocfg_file = self._init_sim_config(seed)
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), '-c', sumocfg_file]
        command += ['--seed', str(seed)]
        command += ['--remote-port', str(self.port)]
        command += ['--no-step-log', 'True']
        if self.name != 'real_net':
            command += ['--time-to-teleport', '600'] # long teleport for safety
        else:
            command += ['--time-to-teleport', '300']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        # collect trip info if necessary
        if self.is_record:
            command += ['--tripinfo-output',
                        self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))]
        sumo_log_dir = self.output_path if self.output_path else self.data_path
        self.sumo_log_file = os.path.join(sumo_log_dir, 'sumo_port_%d.log' % self.port)
        self.sumo_log_fh = open(self.sumo_log_file, 'a')
        logging.info('Starting SUMO on port %d (log: %s)', self.port, self.sumo_log_file)
        self.sumo_process = subprocess.Popen(
            command,
            stdout=self.sumo_log_fh,
            stderr=subprocess.STDOUT
        )
        time.sleep(2)

        old_default_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(TRACI_CONNECT_TIMEOUT_SEC)
            try:
                self.sim = traci.connect(port=self.port, wait=True, numRetries=5)
            except TypeError:
                try:
                    self.sim = traci.connect(port=self.port, waitUntilConnected=True, numRetries=5)
                except TypeError:
                    try:
                        self.sim = traci.connect(port=self.port, numRetries=5)
                    except TypeError:
                        self.sim = traci.connect(port=self.port)
            if hasattr(self.sim, '_socket') and self.sim._socket is not None:
                self.sim._socket.settimeout(TRACI_SOCKET_TIMEOUT_SEC)
            logging.info('TraCI connection established on port %d', self.port)
        except Exception as e:
            logging.error('Failed to connect to TraCI on port %d: %s', self.port, e)
            if self.sumo_process is not None:
                try:
                    self.sumo_process.terminate()
                    self.sumo_process.wait(timeout=5)
                except Exception:
                    try:
                        self.sumo_process.kill()
                    except Exception:
                        pass
            self.sumo_process = None
            self.sim = None
            raise RuntimeError('Could not initialize SUMO/TraCI connection')
        finally:
            socket.setdefaulttimeout(old_default_timeout)

    def _init_sim_config(self):
        # needs to be overwriteen
        raise NotImplementedError()

    def _init_sim_traffic(self):
        return

    def _init_state_space(self):
        self._reset_state()
        self.n_s_ls = []
        self.n_w_ls = []
        self.n_f_ls = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            num_wave = node.num_state
            num_fingerprint = 0
            for nnode_name in node.neighbor:
                if self.agent not in ['a2c', 'greedy']:
                    # all marl agents have neighborhood communication
                    num_wave += self.nodes[nnode_name].num_state
                if self.agent == 'ma2c':
                    # only ma2c uses neighbor's policy
                    num_fingerprint += self.nodes[nnode_name].num_fingerprint
            num_wait = 0 if 'wait' not in self.state_names else node.num_state
            self.n_s_ls.append(num_wave + num_wait + num_fingerprint)
            self.n_f_ls.append(num_fingerprint)
            self.n_w_ls.append(num_wait)
        self.n_s = np.sum(np.array(self.n_s_ls))

    def _measure_reward_step(self):
        rewards = []
        for node_name in self.node_names:
            queues = []
            waits = []
            for ild in self.nodes[node_name].ilds_in:
                if self.obj in ['queue', 'hybrid']:
                    if self.name == 'real_net':
                        cur_queue = min(10, self.sim.lane.getLastStepHaltingNumber(ild))
                    else:
                        cur_queue = self.sim.lanearea.getLastStepHaltingNumber(ild)
                    queues.append(cur_queue)
                if self.obj in ['wait', 'hybrid']:
                    waits.append(self._get_lane_wait_metric(ild))
                # if self.name == 'real_net':
                #     lane_name = ild.split(':')[1]
                # else:
                #     lane_name = 'e:' + ild.split(':')[1]
                # queues.append(self.sim.lane.getLastStepHaltingNumber(lane_name))

            queue = np.sum(np.array(queues)) if len(queues) else 0
            wait = np.sum(np.array(waits)) if len(waits) else 0
            # if self.obj in ['wait', 'hybrid']:
            #     wait = np.sum(self.nodes[node_name].waits * (queues > 0))
            if self.obj == 'queue':
                reward = - queue
            elif self.obj == 'wait':
                reward = - wait
            else:
                reward = - queue - self.coef_wait * wait
            rewards.append(reward)
        return np.array(rewards)

    def _measure_state_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            for state_name in self.state_names:
                if state_name == 'wave':
                    cur_state = []
                    for ild in node.ilds_in:
                        if self.name == 'real_net':
                            cur_wave = self.sim.lane.getLastStepVehicleNumber(ild)
                        else:
                            cur_wave = self.sim.lanearea.getLastStepVehicleNumber(ild)
                        cur_state.append(cur_wave)
                    cur_state = np.array(cur_state)
                else:
                    cur_state = []
                    for ild in node.ilds_in:
                        cur_state.append(self._get_lane_wait_metric(ild))
                    cur_state = np.array(cur_state)
                if self.record_stats:
                    self.state_stat[state_name] += list(cur_state)
                # normalization
                norm_cur_state = self._norm_clip_state(cur_state,
                                                       self.norms[state_name],
                                                       self.clips[state_name])
                if state_name == 'wave':
                    node.wave_state = norm_cur_state
                else:
                    node.wait_state = norm_cur_state

    def _measure_traffic_step(self):
        cars = self.sim.vehicle.getIDList()
        num_tot_car = len(cars)
        num_in_car = self.sim.simulation.getDepartedNumber()
        num_out_car = self.sim.simulation.getArrivedNumber()
        if num_tot_car > 0:
            avg_waiting_time = np.mean([self.sim.vehicle.getWaitingTime(car) for car in cars])
            avg_speed = np.mean([self.sim.vehicle.getSpeed(car) for car in cars])
        else:
            avg_speed = 0
            avg_waiting_time = 0
        # all trip-related measurements are not supported by traci,
        # need to read from outputfile afterwards
        queues = []
        for node_name in self.node_names:
            for ild in self.nodes[node_name].ilds_in:
                queues.append(self.sim.lane.getLastStepHaltingNumber(ild))
        queue_arr = np.array(queues, dtype=np.float32)
        total_queue = float(np.sum(queue_arr)) if len(queue_arr) else 0.0
        avg_queue = float(np.mean(queue_arr)) if len(queue_arr) else 0.0
        std_queue = float(np.std(queue_arr)) if len(queue_arr) else 0.0
        cur_traffic = {'episode': self.cur_episode,
                       'time_sec': self.cur_sec,
                       'number_total_car': num_tot_car,
                       'number_departed_car': num_in_car,
                       'number_arrived_car': num_out_car,
                       'avg_wait_sec': avg_waiting_time,
                       'avg_speed_mps': avg_speed,
                       'total_queue': total_queue,
                       'std_queue': std_queue,
                       'avg_queue': avg_queue}
        self.traffic_data.append(cur_traffic)

    def _get_lane_wait_metric(self, ild):
        """Per-lane wait metric used for state and reward.

        fast_wait_metric=True uses an aggregate lane waiting-time query to avoid
        per-vehicle loops that become expensive under congestion.
        """
        if self.fast_wait_metric:
            try:
                nveh = self.sim.lane.getLastStepVehicleNumber(ild)
                if nveh <= 0:
                    return 0.0
                return self.sim.lane.getWaitingTime(ild) / float(nveh)
            except Exception:
                return 0.0

        max_pos = 0.0
        car_wait = 0.0
        if self.name == 'real_net':
            cur_cars = self.sim.lane.getLastStepVehicleIDs(ild)
        else:
            cur_cars = self.sim.lanearea.getLastStepVehicleIDs(ild)
        for vid in cur_cars:
            car_pos = self.sim.vehicle.getLanePosition(vid)
            if car_pos > max_pos:
                max_pos = car_pos
                car_wait = self.sim.vehicle.getWaitingTime(vid)
        return car_wait

    @staticmethod
    def _norm_clip_state(x, norm, clip=-1):
        x = x / norm
        return x if clip < 0 else np.clip(x, 0, clip)

    def _reset_state(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # prev action for yellow phase before each switch
            node.prev_action = 0
            # fingerprint is previous policy[:-1]
            node.num_fingerprint = node.n_a - 1
            node.num_state = self._get_node_state_num(node)
            # node.waves = np.zeros(node.num_state)
            # node.waits = np.zeros(node.num_state)

    def _set_phase(self, action, phase_type, phase_duration):
        for node_name, a in zip(self.node_names, list(action)):
            phase = self._get_node_phase(a, node_name, phase_type)
            self.sim.trafficlight.setRedYellowGreenState(node_name, phase)
            self.sim.trafficlight.setPhaseDuration(node_name, phase_duration)

    def _simulate(self, num_step):
        # reward = np.zeros(len(self.control_node_names))
        self._validate_traci_connection()
        for _ in range(num_step):
            try:
                self.sim.simulationStep()
            except (socket.timeout, traci.TraCIException, OSError) as e:
                logging.error('TraCI simulationStep failed at t=%d: %s', self.cur_sec, e)
                raise RuntimeError('TraCI simulationStep failed')
            # self._measure_state_step()
            # reward += self._measure_reward_step()
            self.cur_sec += 1
            if self.is_record:
                # self._debug_traffic_step()
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
                        'Sim progress: t=%d/%d, active_vehicles=%d, wall=%.1fs for +%ds sim',
                        self.cur_sec, self.episode_length_sec, active_vehicles,
                        wall_elapsed, self.cur_sec - self._last_progress_sim_sec
                    )
                    self._last_progress_sim_sec = self.cur_sec
                    self._last_progress_wall_sec = now
        # return reward

    def _validate_traci_connection(self):
        if self.sim is None:
            raise RuntimeError('TraCI connection not initialized')
        try:
            _ = self.sim.simulation.getTime()
        except Exception as e:
            logging.error('TraCI connection validation failed: %s', e)
            raise RuntimeError('TraCI connection invalid')

    def _log_slow_step_stage(self, stage_name, elapsed_sec):
        if elapsed_sec >= self.step_stage_warn_sec:
            logging.warning(
                "Slow env.step stage '%s': %.2fs (episode=%d, sim_t=%d, env_step=%d)",
                stage_name, elapsed_sec, self.cur_episode, self.cur_sec, self._env_step_idx
            )

    def _transfer_action(self, action):
        '''Transfer global action to a list of local actions'''
        phase_nums = []
        for node in self.control_node_names:
            phase_nums.append(self.nodes[node].phase_num)
        action_ls = []
        for i in range(len(phase_nums) - 1):
            action, cur_action = divmod(action, phase_nums[i])
            action_ls.append(cur_action)
        action_ls.append(action)
        return action_ls

    def _update_waits(self, action):
        for node_name, a in zip(self.node_names, action):
            red_lanes = set()
            node = self.nodes[node_name]
            for i in self.phase_map.get_red_lanes(node.phase_id, a):
                red_lanes.add(node.lanes_in[i])
            for i in range(len(node.waits)):
                lane = node.ilds_in[i]
                if lane in red_lanes:
                    node.waits[i] += self.control_interval_sec
                else:
                    node.waits[i] = 0

    def collect_tripinfo(self):
        # read trip xml, has to be called externally to get complete file
        trip_file = self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))
        tree = ET.ElementTree(file=trip_file)
        for child in tree.getroot():
            cur_trip = child.attrib
            cur_dict = {}
            cur_dict['episode'] = self.cur_episode
            cur_dict['id'] = cur_trip['id']
            cur_dict['depart_sec'] = cur_trip['depart']
            cur_dict['arrival_sec'] = cur_trip['arrival']
            cur_dict['duration_sec'] = cur_trip['duration']
            cur_dict['wait_step'] = cur_trip['waitingCount']
            cur_dict['wait_sec'] = cur_trip['waitingTime']
            self.trip_data.append(cur_dict)
        # delete the current xml
        cmd = 'rm ' + trip_file
        subprocess.check_call(cmd, shell=True)

    def init_data(self, is_record, record_stats, output_path):
        self.is_record = is_record
        self.record_stats = record_stats
        self.output_path = output_path
        if self.is_record:
            self.traffic_data = []
            self.control_data = []
            self.trip_data = []
        if self.record_stats:
            self.state_stat = {}
            for state_name in self.state_names:
                self.state_stat[state_name] = []

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

    def output_data(self):
        if not self.is_record:
            logging.error('Env: no record to output!')
        control_data = pd.DataFrame(self.control_data)
        control_data.to_csv(self.output_path + ('%s_%s_control.csv' % (self.name, self.agent)))
        traffic_data = pd.DataFrame(self.traffic_data)
        traffic_data.to_csv(self.output_path + ('%s_%s_traffic.csv' % (self.name, self.agent)))
        trip_data = pd.DataFrame(self.trip_data)
        trip_data.to_csv(self.output_path + ('%s_%s_trip.csv' % (self.name, self.agent)))

    def reset(self, gui=False, test_ind=0):
        if (self.sim is not None) or (self.sumo_process is not None):
            self.terminate()
        self._reset_state()
        if self.train_mode:
            seed = self.seed
        else:
            seed = self.test_seeds[test_ind]
        # self._init_sim(seed, gui=True)
        self._init_sim(seed, gui=gui)
        self.cur_sec = 0
        self._last_progress_sim_sec = 0
        self._last_progress_wall_sec = time.time()
        self._env_step_idx = 0
        self.cur_episode += 1
        # initialize fingerprint
        if self.agent == 'ma2c':
            self.update_fingerprint(self._init_policy())
        self._init_sim_traffic()
        # next environment random condition should be different
        self.seed += 1
        return self._get_state()

    def terminate(self):
        sim = self.sim
        sumo_process = self.sumo_process
        sumo_log_fh = self.sumo_log_fh
        close_timed_out = False

        # Make terminate idempotent immediately.
        self.sim = None
        self.sumo_process = None
        self.sumo_log_fh = None

        if sim is not None:
            close_error = {'exc': None}

            def _close_sim():
                try:
                    if hasattr(sim, '_socket') and sim._socket is not None:
                        sim._socket.settimeout(TRACI_SOCKET_TIMEOUT_SEC)
                    try:
                        sim.close(wait=True)
                    except TypeError:
                        try:
                            sim.close(True)
                        except TypeError:
                            sim.close()
                except Exception as e:
                    close_error['exc'] = e

            close_thread = threading.Thread(target=_close_sim, daemon=True)
            close_thread.start()
            close_thread.join(timeout=TRACI_CLOSE_TIMEOUT_SEC)
            if close_thread.is_alive():
                close_timed_out = True
                logging.warning('TraCI close timed out; proceeding with process cleanup.')
            elif close_error['exc'] is not None:
                logging.warning('TraCI close warning: %s', close_error['exc'])

        if sumo_process is not None:
            try:
                if sumo_process.poll() is None:
                    if not close_timed_out:
                        try:
                            sumo_process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            logging.warning('SUMO did not exit after TraCI close; terminating process.')
                            sumo_process.terminate()
                            sumo_process.wait(timeout=5)
                    else:
                        sumo_process.terminate()
                        sumo_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logging.warning('SUMO did not terminate gracefully; killing process.')
                try:
                    sumo_process.kill()
                except Exception as e:
                    logging.warning('Error killing SUMO process: %s', e)
            except Exception as e:
                logging.warning('Error terminating SUMO process: %s', e)

        if sumo_log_fh is not None:
            try:
                sumo_log_fh.close()
            except Exception:
                pass

    def step(self, action):
        self._env_step_idx += 1
        step_start = time.time()
        stage_name = 'transfer_action'

        try:
            if self.agent == 'a2c':
                action = self._transfer_action(action)

            stage_name = 'set_phase_yellow'
            t0 = time.time()
            self._set_phase(action, 'yellow', self.yellow_interval_sec)
            self._log_slow_step_stage(stage_name, time.time() - t0)

            stage_name = 'simulate_yellow'
            t0 = time.time()
            self._simulate(self.yellow_interval_sec)
            self._log_slow_step_stage(stage_name, time.time() - t0)

            rest_interval_sec = self.control_interval_sec - self.yellow_interval_sec

            stage_name = 'set_phase_green'
            t0 = time.time()
            self._set_phase(action, 'green', rest_interval_sec)
            self._log_slow_step_stage(stage_name, time.time() - t0)

            stage_name = 'simulate_green'
            t0 = time.time()
            self._simulate(rest_interval_sec)
            self._log_slow_step_stage(stage_name, time.time() - t0)

            stage_name = 'get_state'
            t0 = time.time()
            state = self._get_state()
            self._log_slow_step_stage(stage_name, time.time() - t0)

            stage_name = 'measure_reward'
            t0 = time.time()
            reward = self._measure_reward_step()
            self._log_slow_step_stage(stage_name, time.time() - t0)

            done = (self.cur_sec >= self.episode_length_sec)
            global_reward = np.sum(reward) # for fair comparison

            if self.is_record:
                action_r = ','.join(['%d' % a for a in action])
                cur_control = {'episode': self.cur_episode,
                               'time_sec': self.cur_sec,
                               'step': self.cur_sec / self.control_interval_sec,
                               'action': action_r,
                               'reward': global_reward}
                self.control_data.append(cur_control)

            # use local rewards in test
            if not self.train_mode:
                self._log_slow_step_stage('total', time.time() - step_start)
                return state, reward, done, global_reward

            if self.agent in ['a2c', 'greedy']:
                reward = global_reward
            elif self.agent != 'ma2c':
                # global reward is shared in independent rl
                new_reward = [global_reward] * len(reward)
                reward = np.array(new_reward)
                if self.name == 'real_net':
                    # reward normalization in env for realnet
                    reward = reward / (len(self.node_names) * REALNET_REWARD_NORM)
            else:
                # discounted global reward for ma2c
                new_reward = []
                for node_name, r in zip(self.node_names, reward):
                    cur_reward = r
                    for nnode_name in self.nodes[node_name].neighbor:
                        i = self.node_names.index(nnode_name)
                        cur_reward += self.coop_gamma * reward[i]
                    if self.name != 'real_net':
                        new_reward.append(cur_reward)
                    else:
                        n_node = 1 + len(self.nodes[node_name].neighbor)
                        new_reward.append(cur_reward / (n_node * REALNET_REWARD_NORM))
                reward = np.array(new_reward)

            self._log_slow_step_stage('total', time.time() - step_start)
            return state, reward, done, global_reward
        except Exception as e:
            logging.error(
                "env.step failed at stage '%s' (episode=%d, sim_t=%d, env_step=%d): %s",
                stage_name, self.cur_episode, self.cur_sec, self._env_step_idx, e
            )
            raise

    def update_fingerprint(self, policy):
        for node_name, pi in zip(self.node_names, policy):
            self.nodes[node_name].fingerprint = np.array(pi)[:-1]
