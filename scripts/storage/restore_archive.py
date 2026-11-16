"""Restore a downloaded archive and its manifest into a NEW directory, offline."""
import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def relative(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe archive path: ' + name)
    return p


def restore(manifest, parts_dir, destination):
    m = json.loads(Path(manifest).read_text())
    parts_dir, destination = Path(parts_dir), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Restore destination must be new')
    expected = {x['path']: x for x in m['members']}
    if len(expected) != len(m['members']):
        raise ValueError('Duplicate member paths')
    for name in expected:
        relative(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.cbwce-restore-', dir=destination.parent) as tmp:
        work = Path(tmp)
        payload, tree = work / 'payload', work / 'tree'
        tree.mkdir()
        if m.get('transport_parts'):
            with payload.open('wb') as out:
                for part in m['transport_parts']:
                    path = parts_dir / Path(part['path']).name
                    if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
                        raise ValueError('Transport fragment mismatch: ' + path.name)
                    with path.open('rb') as f:
                        shutil.copyfileobj(f, out, 1024 * 1024)
        else:
            path = parts_dir / Path(m['archive']).name
            with path.open('rb') as f, payload.open('wb') as out:
                shutil.copyfileobj(f, out, 1024 * 1024)
        if digest(payload) != m['archive_sha256']:
            raise ValueError('Archive SHA256 mismatch')
        verified = {}
        if m.get('archive_kind') == 'raw_file':
            first = None
            for item in m['members']:
                if item['sha256'] != m['archive_sha256'] or item['size'] != payload.stat().st_size:
                    raise ValueError('Raw member identity mismatch')
                target = tree / relative(item['path'])
                target.parent.mkdir(parents=True, exist_ok=True)
                if first is None:
                    shutil.copyfile(payload, target)
                    first = target
                else:
                    os.link(first, target)
                verified[item['path']] = (item['sha256'], item['size'])
        else:
            with tarfile.open(payload, 'r:gz') as archive:
                for member in archive:
                    if member.name not in expected or member.name in verified:
                        raise ValueError('Unexpected or repeated tar member')
                    item = expected[member.name]
                    target = tree / relative(member.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if member.islnk():
                        relative(member.linkname)
                        if verified.get(member.linkname) != (item['sha256'], item['size']):
                            raise ValueError('Invalid hardlink target')
                        os.link(tree / member.linkname, target)
                    elif member.isfile():
                        if member.size != item['size']:
                            raise ValueError('Member size mismatch')
                        h = hashlib.sha256()
                        with archive.extractfile(member) as source, target.open('xb') as out:
                            for block in iter(lambda: source.read(1024 * 1024), b''):
                                h.update(block)
                                out.write(block)
                        if h.hexdigest() != item['sha256']:
                            raise ValueError('Member SHA256 mismatch')
                        os.chmod(target, member.mode & 0o777)
                    else:
                        raise ValueError('Only regular files and internal hardlinks are allowed')
                    verified[member.name] = (item['sha256'], item['size'])
        if set(verified) != set(expected):
            raise ValueError('Incomplete archive')
        for name, item in expected.items():
            if 'mtime_ns' in item:
                os.utime(tree / name, ns=(item['mtime_ns'], item['mtime_ns']))
        os.rename(tree, destination)
    return dict(status='restored', files=len(expected), destination=str(destination))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--parts-dir', required=True)
    p.add_argument('--destination', required=True)
    a = p.parse_args()
    print(json.dumps(restore(a.manifest, a.parts_dir, a.destination)))
