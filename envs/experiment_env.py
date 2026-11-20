"""Read-only reuse of legacy maps/state encoders; isolated SUMO files and queues."""
import json
import os
import fcntl
import hashlib
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
from experiments.protocol import root
from experiments.metrics import EpisodeMetrics


TRACI_START_LOCK = Path('/tmp') / ('cbwce_traci_start_' +
                                   hashlib.sha256(str(ROOT).encode()).hexdigest()[:16] + '.lock')


def start_traci_locked(command, label, stdout):
    """Serialize only SUMO startup so parallel workers cannot reuse a free port."""
    lock_fd = os.open(str(TRACI_START_LOCK), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # This installed TraCI release can return None from getFreeSocketPort()
        # when port=None. Select on loopback ourselves, but keep the process
        # lock until the connection is live so sibling workers cannot race for
        # the released port. Retry with a new port for unrelated local users.
        error = None
        for _ in range(10):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            try:
                return traci.start(command, port=port, numRetries=10,
                                   label=label, stdout=stdout)
            except traci.exceptions.FatalTraCIError as exc:
                error = exc
        raise error
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def validate_visualization(enabled):
    if type(enabled) is not bool:
        raise ValueError('visualization must be boolean')
    if enabled:
        if shutil.which('sumo-gui') is None:
            raise ValueError('Visualization requires sumo-gui on PATH; use --no-visualization for headless runs')
        if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
            raise ValueError('Visualization requires an X display (DISPLAY); start from a desktop session or use --no-visualization')


class RevisedMixin:
    def __init__(self, network, family, output, seed=9001, visualization=False,
                 config_path=None, audit=False):
        validate_visualization(visualization)
        self.audit = audit
        self.collect_trips = False
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
        from experiments.configuration import load_controller_config, load_wce_config
        config, self.config_path = load_controller_config(
            network, family, path=config_path, seed=seed)
        wce, self.wce_config_path = load_wce_config(network)
        self.config = config
        self.wce_config = wce
        cfg = config['ENV_CONFIG']
        self.net_file = root('assets') / network / 'network.net.xml'
        add_file = root('assets') / network / 'detectors.add.xml'
        cfg['data_path'] = str(root('assets') / network)
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
                                  {n: self.nodes[n].neighbor for n in self.node_names}, self.coop_gamma)
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
        # Runtime files are overwritten locally; identities live in run manifests.
        self.trip_file = self.work / ('trips_{}.xml'.format(attempt))
        command = ['sumo-gui' if gui else 'sumo', '--net-file', str(self.net_file),
                   '--additional-files', str(self.additional), '--route-files', str(self.empty_routes),
                   '--seed', str(int(seed)), '--step-length', '1', '--no-step-log', 'true',
                   '--no-warnings', 'true', '--duration-log.disable', 'true',
                   '--time-to-teleport', '600' if self.network == 'grid' else '300',
]
        if gui:
            command += ['--start', '--quit-on-end']
        if self.collect_trips:
            command += ['--tripinfo-output', str(self.trip_file)]
        self.log_stream = (self.work / 'sumo.log').open('w')
        try:
            start_traci_locked(command, self.connection_label, self.log_stream)
        except BaseException:
            self.log_stream.close()
            self.log_stream = None
            raise
        self.sim = traci.getConnection(self.connection_label)
        self.effective_seed = int(command[command.index('--seed') + 1])
        startup = {'requested_seed': int(seed), 'effective_seed': self.effective_seed,
                   'visualization': bool(gui),
                   'command': command, 'sumo_version': list(self.sim.getVersion())}
        self.startups[:] = [startup]
        self.rows, self.lane_rows, self.controller_rows = [], [], []
        self.wait_rows = []
        self.cumulative_stopped_time = 0.
        self.scheduled = 0
        if hasattr(self, 'metric'):
            self.metrics = EpisodeMetrics(self.metric.lanes, self.audit)
            self.sim_routes = set()
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

    def finish_tripinfo(self):
        """Close SUMO so XML is complete; caller removes it after saving the row."""
        from experiments.tripinfo import summarize_trips
        self.terminate()
        return summarize_trips(self.trip_file)

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
            route_id = 'r_' + hashlib.sha256(' '.join(vehicle['edges']).encode()).hexdigest()[:24]
            if route_id not in self.sim_routes:
                self.sim.route.add(route_id, vehicle['edges'])
                self.sim_routes.add(route_id)
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
            speeds = self.sim.vehicle.getAllSubscriptionResults()
            for vehicle in vehicles:
                if vehicle not in speeds:
                    self.sim.vehicle.subscribe(vehicle, [tc.VAR_SPEED])
            speeds = self.sim.vehicle.getAllSubscriptionResults()
            self.metrics.add(self.cur_sec, queue, active=len(vehicles),
                speed_sum=sum(speeds[v][tc.VAR_SPEED] for v in vehicles),
                inserted=self.sim.simulation.getDepartedNumber(),
                completed=self.sim.simulation.getArrivedNumber(),
                pending=len(self.sim.simulation.getPendingVehicles()),
                teleports=self.sim.simulation.getStartingTeleportNumber(),
                collisions=self.sim.simulation.getCollidingVehiclesNumber())

    def step(self, actions):
        if self.cur_sec + self.control_interval_sec > self.episode_length_sec:
            raise ValueError('Controller step would exceed horizon')
        yellow = self.yellow_interval_sec
        green = self.control_interval_sec - yellow
        self._set_phase(actions, 'yellow', yellow)
        self._simulate(yellow)
        self._set_phase(actions, 'green', green)
        self._simulate(green)
        rewards = self.metric.rewards([self.metrics.last_lanes], self.agent)
        if not np.all(np.isfinite(rewards)):
            raise FloatingPointError('Nonfinite controller reward')
        if self.audit:
            self.controller_rows.append(dict(time=self.cur_sec,actions=list(map(int,actions)),rewards=list(map(float,rewards))))
        return self._get_state(), rewards, self.cur_sec == self.episode_length_sec

    def wce_observation(self):
        """Node-major wave features only, padded to a common lane width."""
        self._measure_state_step()
        features = []
        for name in self.wce_nodes:
            node = self.nodes[name]
            wave = np.pad(node.wave_state, (0, self.feature_width - len(node.wave_state)), 'constant')
            features.extend(wave)
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
        if self.audit:
            np.savez_compressed(str(path)+'.npz', lanes=np.asarray(self.metric.lanes),
                                queue=np.asarray(self.metrics.trace), time=np.arange(1,self.cur_sec+1),
                                control_time=np.asarray([r['time'] for r in self.controller_rows]),
                                actions=np.asarray([r['actions'] for r in self.controller_rows]),
                                rewards=np.asarray([r['rewards'] for r in self.controller_rows]))



class GridEnvironment(RevisedMixin, LargeGridEnv):
    pass


class MonacoEnvironment(RevisedMixin, RealNetEnv):
    pass


def make_environment(network, family, output, seed=9001, visualization=False, config_path=None, audit=False):
    cls = GridEnvironment if network == 'grid' else MonacoEnvironment
    return cls(network, family, output, seed, visualization=visualization,
               config_path=config_path, audit=audit)
