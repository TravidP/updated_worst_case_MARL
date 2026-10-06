"""Prepare explicit normalized datasets without replacing existing inputs."""
import csv
import json
import io
from experiments.core import ROOT, file_hash, write_json
from experiments.protocol import dataset_root, scenario_root, settings


def immutable_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError('Prepared input differs; preserve it and select a new protocol version: ' + str(path))
    else:
        with path.open('x') as f:
            f.write(content)


def prepare():
    from experiments.demand import original_profiles
    from experiments.scenarios import definitions
    for network in settings()['networks']:
        root = dataset_root(network)
        manifest = []
        for group in original_profiles(network):
            path = root / 'train' / (group['name'] + '.csv')
            out = io.StringIO()
            writer = csv.writer(out, lineterminator='\n')
            writer.writerow(['origin_edge', 'dest_edge', 'veh_per_hour'])
            writer.writerows(group['rows'])
            immutable_text(path, out.getvalue())
            manifest.append(dict(group, rows=None, prepared=str(path.relative_to(ROOT)), prepared_hash=file_hash(path)))
        immutable_text(root / 'train/manifest.json', json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        for split in ('seen', 'validation', 'test'):
            directory = scenario_root(network, split)
            immutable_text(directory / (split + '_scenarios.json'), json.dumps(definitions(network, split), indent=2, sort_keys=True) + '\n')
        # Revised INIs are authoritative, independently tuned inputs. Preparation
        # validates them but must never regenerate them from legacy config files.
        from experiments.configuration import load_controller_config, load_wce_config
        for family in settings()['controllers']:
            load_controller_config(network, family)
        load_wce_config(network)
    return {'status': 'prepared', 'networks': settings()['networks'], 'training_profiles_per_network': 11,
            'validation_scenarios': 6, 'test_scenarios': 12, 'seen_scenarios': 11}
