"""Versioned, relocatable experiment contract. No publication validation defaults."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def root(kind):
    defaults = {'assets': ROOT/'assets', 'data': ROOT.parent/'cbwce_data',
                'run': ROOT.parent/'cbwce_runs', 'cache': ROOT.parent/'cbwce_cache'}
    return Path(os.environ.get('CBWCE_'+kind.upper()+'_ROOT', defaults[kind])).resolve()

def settings():
    return json.loads((ROOT/'config/protocol.json').read_text())

def dataset_root(network):
    return root('data')/network

def scenario_root(network, split):
    return dataset_root(network)/'v8.1'/split

def config_path(network, controller):
    return ROOT/'config/revised'/('config_%s_%s.ini' % (controller, 'large' if network=='grid' else 'real'))

def scientific_config(config):
    result = {s: dict(config[s]) for s in config.sections()}
    for key in ('data_path', 'seed', 'test_seeds'):
        result.get('ENV_CONFIG', {}).pop(key, None)
    return result
