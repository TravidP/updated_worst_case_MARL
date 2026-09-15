"""Read-only reuse of legacy maps/state encoders; isolated SUMO files and queues."""
import configparser
import json
import os
import socket
import shutil
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import traci
import traci.constants as tc
from envs.env import TrafficSimulator
from envs.large_grid_env import LargeGridEnv
from envs.real_net_env import RealNetEnv
from experiments.core import ROOT, QueueMetric, file_hash, write_json
from experiments.demand import profiles


def validate_visualization(enabled):
    if type(enabled) is not bool:
        raise ValueError('visualization must be boolean')
    if enabled:
        if shutil.which('sumo-gui') is None:
            raise ValueError('Visualization requires sumo-gui on PATH; use --no-visualization for headless runs')
        if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
            raise ValueError('Visualization requires an X display (DISPLAY); start from a desktop session or use --no-visualization')


class RevisedMixin:
    def __init__(self, network, family, output, seed=9001, visualization=False):
        validate_visualization(visualization)
        self.visualization = visualization
        self.network = network
        self.work = Path(output).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.groups = profiles(network)
        self.startups = []
        self.routes = {}
        self.rows = []
        self.lane_rows = []
        self.connection_label = 'revision_' + uuid.uuid4().hex
        self.log_stream = None
        suffix = 'large' if network == 'grid' else 'real'
        config = configparser.ConfigParser()
        from experiments.protocol import config_path
        config.read(str(config_path(network, family)))
        self.config = config
        cfg = config['ENV_CONFIG']
        cfg['seed'] = str(seed)
        cfg['objective'] = 'queue'
        cfg['coef_wait'] = '0'
        cfg['episode_length_sec'] = '6600'
        cfg['fast_wait_metric'] = 'true'
        self.net_file = ROOT / ('large_grid/data/exp.net.xml' if network == 'grid'
                                else 'real_net_subnet/data/in/most.net.xml')
        add_file = ROOT / ('large_grid/data/exp.add.xml' if network == 'grid'
                           else 'real_net_subnet/data/in/most.add.xml')
        self.asset_hashes = {'network': file_hash(self.net_file), 'additional': file_hash(add_file)}
        tree = ET.parse(str(add_file))
        for element in tree.iter():
            if 'file' in element.attrib:
                element.set('file', os.devnull)
        self.additional = self.work / 'detectors.add.xml'
        tree.write(str(self.additional))
        self.empty_routes = self.work / 'empty.rou.xml'
        self.empty_routes.write_text('<routes><vType id="type1" vClass="passenger" length="5" accel="5" decel="10" speedDev="0"/></routes>')
        super().__init__(cfg, output_path=str(self.work) + os.sep, is_record=False)
        self.metric = QueueMetric({n: self.nodes[n].ilds_in for n in self.node_names},
                                  {n: self.nodes[n].neighbor for n in self.node_names})
        expected = 150 if network == 'grid' else 116
        if len(self.metric.lanes) != expected:
            raise ValueError('Unexpected monitored lane set')
        self.wce_nodes = (sorted(self.node_names, key=lambda n: int(n[2:]))
                          if network == 'grid' else sorted(self.node_names))
        self.feature_width = max(len(self.nodes[n].ilds_in) for n in self.wce_nodes)

    def _load_dynamic_scenarios(self):
        self.scenarios = [g['rows'] for g in self.groups]
        self.loaded_filenames = [g['name'] for g in self.groups]

    def _load_scenarios(self):
        self.loaded_filenames = [g['name'] for g in self.groups]
        return [g['rows'] for g in self.groups]

    def _init_sim(self, seed, gui=False):
        attempt = len(self.startups)
        self.trip_file = self.work / ('trips_{}.xml'.format(attempt))
        command = ['sumo-gui' if gui else 'sumo', '--net-file', str(self.net_file),
                   '--additional-files', str(self.additional), '--route-files', str(self.empty_routes),
                   '--seed', str(int(seed)), '--step-length', '1', '--no-step-log', 'true',
                   '--no-warnings', 'true', '--duration-log.disable', 'true',
                   '--time-to-teleport', '600' if self.network == 'grid' else '300',
                   '--tripinfo-output', str(self.trip_file)]
        if gui:
            command += ['--start', '--quit-on-end']
        self.log_stream = (self.work / ('sumo_{}.log'.format(attempt))).open('w')
        # Fail visibly when local sockets are unavailable instead of retrying port=None.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        traci.start(command, port=port, numRetries=5, label=self.connection_label, stdout=self.log_stream)
        self.sim = traci.getConnection(self.connection_label)
        self.effective_seed = int(command[command.index('--seed') + 1])
        startup = {'requested_seed': int(seed), 'effective_seed': self.effective_seed,
                   'visualization': bool(gui),
                   'command': command, 'sumo_version': list(self.sim.getVersion())}
        self.startups.append(startup)
        write_json(self.work / ('startup_{}.json'.format(attempt)), startup)
        self.rows, self.lane_rows, self.controller_rows = [], [], []
        self.scheduled = 0
        if hasattr(self, 'metric'):
            for lane in self.metric.lanes:
                self.sim.lane.subscribe(lane, [tc.LAST_STEP_VEHICLE_HALTING_NUMBER])

    def terminate(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
        if self.log_stream is not None:
            self.log_stream.close()
            self.log_stream = None

    def reset_episode(self, seed, evaluation=False, horizon=6600):
        self.episode_length_sec = horizon
        self.T = horizon // 5
        self.train_mode = not evaluation
        self.seed = int(seed)
        self.init_test_seeds([int(seed)])
        return TrafficSimulator.reset(self, gui=self.visualization, test_ind=0)

    def resolve_route(self, origin, destination):
        key = (origin, destination)
        if key not in self.routes:
            result = self.sim.simulation.findRoute(origin, destination, vType='type1')
            if not result.edges:
                raise ValueError('Unreachable OD: ' + str(key))
            self.routes[key] = list(result.edges)
        return self.routes[key]

    def prepare_routes(self):
        if self.cur_sec != 0 or self.scheduled:
            raise ValueError('Route preparation requires an empty network')
        for group in self.groups:
            for origin, destination, rate in group['rows']:
                if rate > 0:
                    self.resolve_route(origin, destination)

    def inject(self, vehicles):
        for vehicle in vehicles:
            route_id = 'route_' + vehicle['id']
            self.sim.route.add(route_id, vehicle['edges'])
            self.sim.vehicle.add(vehicle['id'], route_id, typeID='type1', depart=str(vehicle['depart']))
            self.sim.vehicle.setSpeedFactor(vehicle['id'], vehicle['speed_factor'])
            self.scheduled += 1

    def _simulate(self, seconds):
        for _ in range(seconds):
            self.sim.simulationStep()
            self.cur_sec += 1
            queue = [self.sim.lane.getSubscriptionResults(l)[tc.LAST_STEP_VEHICLE_HALTING_NUMBER]
                     for l in self.metric.lanes]
            vehicles = self.sim.vehicle.getIDList()
            speed_results = self.sim.vehicle.getAllSubscriptionResults()
            for vehicle in vehicles:
                if vehicle not in speed_results:
                    self.sim.vehicle.subscribe(vehicle, [tc.VAR_SPEED])
            speed_results = self.sim.vehicle.getAllSubscriptionResults()
            if getattr(self, 'verify_subscriptions', False):
                for vehicle in vehicles:
                    if speed_results[vehicle][tc.VAR_SPEED] != self.sim.vehicle.getSpeed(vehicle):
                        raise AssertionError('Subscription speed disagrees with direct TraCI read')
            self.lane_rows.append(queue)
            self.rows.append({'time': self.cur_sec, 'queue': float(sum(queue)),
                'active': len(vehicles), 'speed_sum': sum(speed_results[v][tc.VAR_SPEED] for v in vehicles),
                'inserted': self.sim.simulation.getDepartedNumber(),
                'completed': self.sim.simulation.getArrivedNumber(),
                'pending': len(self.sim.simulation.getPendingVehicles()),
                'teleports': self.sim.simulation.getStartingTeleportNumber(),
                'collisions': self.sim.simulation.getCollidingVehiclesNumber()})

    def step(self, actions):
        if self.cur_sec + 5 > self.episode_length_sec:
            raise ValueError('Controller step would exceed horizon')
        self._set_phase(actions, 'yellow', 2)
        self._simulate(2)
        self._set_phase(actions, 'green', 3)
        self._simulate(3)
        rewards = self.metric.rewards(self.lane_rows[-5:], self.agent)
        self.controller_rows.append({'time': self.cur_sec, 'learner_rewards': rewards.tolist()})
        return self._get_state(), rewards, self.cur_sec == self.episode_length_sec

    def wce_observation(self):
        self._measure_state_step()
        features = []
        for name in self.wce_nodes:
            node = self.nodes[name]
            wave = np.pad(node.wave_state, (0, self.feature_width - len(node.wave_state)), 'constant')
            features.extend(wave)
            if self.network == 'grid':
                features.extend(np.pad(node.wait_state, (0, self.feature_width - len(node.wait_state)), 'constant'))
        return np.asarray(features, dtype=np.float32)

    def adjacency(self):
        a = np.eye(len(self.wce_nodes), dtype=np.float32)
        for i, n in enumerate(self.wce_nodes):
            for other in self.nodes[n].neighbor:
                a[i, self.wce_nodes.index(other)] = 1.
        a = np.maximum(a, a.T)
        d = np.diag(1. / np.sqrt(a.sum(axis=1)))
        return d.dot(a).dot(d)

    def export_episode(self, path):
        path = Path(path)
        np.savez_compressed(str(path.with_suffix('.npz')), lanes=np.array(self.metric.lanes),
                            queue=np.asarray(self.lane_rows), time=np.arange(1, self.cur_sec + 1))
        with path.with_suffix('.jsonl').open('x') as stream:
            for row in self.rows:
                stream.write(json.dumps(row) + '\n')
        with path.with_suffix('.controls.jsonl').open('x') as stream:
            for row in self.controller_rows:
                stream.write(json.dumps(row) + '\n')


class GridEnvironment(RevisedMixin, LargeGridEnv):
    pass


class MonacoEnvironment(RevisedMixin, RealNetEnv):
    pass


def make_environment(network, family, output, seed=9001, visualization=False):
    return (GridEnvironment if network == 'grid' else MonacoEnvironment)(
        network, family, output, seed, visualization=visualization)
