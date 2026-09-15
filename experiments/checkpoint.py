"""Strict revision checkpoints; load only trusted, locally produced bundles."""
import json
import pickle
from pathlib import Path
from experiments.core import digest, file_hash, write_json


def save_checkpoint(path, models, runner_state, parents=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    signatures = {}
    for name, model in models.items():
        signatures[name] = model.signature
        directory = path / name
        directory.mkdir()
        with model.graph.as_default():
            model.saver.save(model.sess, str(directory / 'variables'), write_meta_graph=False)
        with (directory / 'state.pkl').open('xb') as stream:
            pickle.dump(model.state(), stream, protocol=4)
    with (path / 'runner.pkl').open('xb') as stream:
        pickle.dump(runner_state, stream, protocol=4)
    files = {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob('*')) if p.is_file()}
    manifest = {'version': 1, 'signatures': signatures, 'files': files, 'parents': parents or {}}
    manifest['hash'] = digest(manifest)
    # The last exclusive write marks a complete checkpoint. Partial bundles fail loading.
    write_json(path / 'manifest.json', manifest)
    return manifest['hash']


def inspect_checkpoint(path):
    path = Path(path)
    with (path / 'manifest.json').open() as stream:
        manifest = json.load(stream)
    content = dict(manifest)
    expected = content.pop('hash')
    if content.get('version') != 1 or digest(content) != expected:
        raise ValueError('Incompatible checkpoint manifest')
    for relative, expected_hash in manifest['files'].items():
        target = (path / relative).resolve()
        if path.resolve() not in target.parents or file_hash(target) != expected_hash:
            raise ValueError('Checkpoint file integrity failure')
    return manifest


def load_checkpoint(path, models):
    path = Path(path)
    manifest = inspect_checkpoint(path)
    for name, model in models.items():
        if manifest['signatures'].get(name) != model.signature:
            raise ValueError('Incompatible checkpoint model: ' + name)
    for name, model in models.items():
        with model.graph.as_default():
            model.saver.restore(model.sess, str(path / name / 'variables'))
        with (path / name / 'state.pkl').open('rb') as stream:
            model.restore_state(pickle.load(stream))
    with (path / 'runner.pkl').open('rb') as stream:
        state = pickle.load(stream)
    return state


def checkpoint_identity(path, network, family, seed):
    """Validate the originating run identity without changing checkpoint format."""
    origin = Path(path).parent / 'manifest.json'
    if not origin.is_file():
        raise ValueError('Checkpoint requires its originating run manifest')
    manifest = json.loads(origin.read_text())
    body = dict(manifest)
    claimed = body.pop('manifest_hash', None)
    if claimed is None or digest(body) != claimed:
        raise ValueError('Checkpoint origin manifest integrity failure')
    if (manifest.get('network'), manifest.get('controller'), manifest.get('seed')) != (network, family, seed):
        raise ValueError('Checkpoint network/controller/training-seed mismatch')
    return manifest
