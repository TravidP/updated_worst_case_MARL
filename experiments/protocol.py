"""Shared public protocol and output conventions."""
import json
import uuid
from datetime import datetime
from experiments.core import ROOT

PROTOCOL_PATH = ROOT / 'config/revised/protocol.json'


def settings():
    return json.loads(PROTOCOL_PATH.read_text())


def dataset_root(network):
    return ROOT / ('data_traffic/revised' if network == 'grid' else 'real_net_subnet/demand_groups/revised')


def scenario_root(network, split):
    """Versioned scenario/artifact root; training profiles remain shared inputs."""
    if split not in ('seen', 'validation', 'test'):
        raise ValueError('Unknown split: ' + str(split))
    split_directory = 'test' if split == 'seen' else split
    return dataset_root(network) / ('protocol_v{}'.format(settings()['version'])) / split_directory


def config_path(network, controller):
    name = 'config_{}_{}.ini'.format(controller, 'large' if network == 'grid' else 'real')
    prepared = ROOT / 'config/revised' / name
    return prepared if prepared.exists() else ROOT / 'config' / name


def output_root(stage, network='grid', method='baseline'):
    if stage == 'parent' or (stage == 'continue' and method == 'baseline'):
        folder = 'runs'
    elif stage == 'wce':
        folder = 'output_adversary' if network == 'grid' else 'output_adversary_monaco'
    elif stage == 'continue':
        folder = 'output_coevolution' if network == 'grid' else 'output_coevolution_real'
    elif stage == 'report':
        folder = 'output_result'
    else:
        folder = 'runs_eval'
    return ROOT / folder / 'revised'


def run_id(pilot=False):
    return ('pilot_' if pilot else 'publication_') + datetime.utcnow().strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]


def output_path(stage, network, controller, seed, method='baseline', pilot=False):
    return output_root(stage, network, method) / network / controller / ('seed_' + str(seed)) / (method if stage == 'continue' else stage) / run_id(pilot)


def allowed_output(path):
    path = path.resolve()
    roots = [ROOT / x / 'revised' for x in ['runs','output_adversary','output_adversary_monaco',
             'output_coevolution','output_coevolution_real','runs_eval','output_result','figs']]
    return any(path != r and r in path.parents for r in roots)
