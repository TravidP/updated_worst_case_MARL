"""Explicit, resumable archive batches. Network transfers are handled separately."""
import argparse
import gzip
import hashlib
import json
import os
import tarfile
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def source(root, item):
    path = root / item['path']
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Unsafe path: ' + item['path'])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (item['size'], item['mtime_ns']):
        raise ValueError('Source changed: ' + item['path'])
    return path


def pack(root, catalog, staging, number, limit_mib):
    name = 'batch_%06d' % number
    manifest = catalog / (name + '.json')
    if manifest.exists():
        m = json.loads(manifest.read_text())
        print(json.dumps({k:v for k,v in m.items() if k != 'members'}))
        return
    done = set()
    for p in catalog.glob('batch_*.json'):
        m = json.loads(p.read_text())
        if m['state'] in ('download_verified', 'local_removed_untracked_only', 'local_removed'):
            done.update(x['path'] for x in m['members'])
    output = staging / (name + '.tar.gz')
    if output.exists():
        raise ValueError('Unfinished archive exists; inspect before retrying')
    selected = []
    limit = limit_mib * 1024**2
    with output.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w|', dereference=False) as archive:
                with (catalog / 'inventory.jsonl').open() as inventory:
                    for line in inventory:
                        item = json.loads(line)
                        if item['path'] in done or item['tracked']:
                            continue
                        p = root / item['path']
                        if not p.exists():
                            continue
                        if item['size'] + 1024**2 >= limit:
                            continue  # Oversized source handled by a separate multipart pass.
                        if selected and raw.tell() + item['size'] + 1024**2 > limit:
                            break
                        p = source(root, item)
                        item['sha256'] = sha(p)
                        archive.add(p, arcname=item['path'], recursive=False)
                        selected.append(item)
                        gz.flush()
    if not selected:
        output.unlink()
        print(json.dumps({'state': 'no_eligible_files'}))
        return
    assert output.stat().st_size <= 512 * 1024**2
    m = {'state': 'packed', 'archive': str(output), 'archive_sha256': sha(output),
         'archive_bytes': output.stat().st_size, 'source_bytes': sum(x['size'] for x in selected),
         'files': len(selected), 'members': selected}
    save(manifest, m)
    print(json.dumps({k:v for k,v in m.items() if k != 'members'}))


def verify(root, catalog, staging, number, drive_id):
    manifest = catalog / ('batch_%06d.json' % number)
    m = json.loads(manifest.read_text())
    restored = staging / ('restored_%06d.tar.gz' % number)
    assert sha(restored) == m['archive_sha256'], 'Downloaded archive hash mismatch'
    with tarfile.open(restored, 'r:gz') as t:
        members = t.getmembers()
        assert len(members) == len(m['members'])
        assert [x.name for x in members] == [x['path'] for x in m['members']]
        for item, member in zip(m['members'], members):
            assert member.isfile() and member.size == item['size']
            h = hashlib.sha256()
            with t.extractfile(member) as f:
                for b in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(b)
            assert h.hexdigest() == item['sha256'], item['path']
    m.update(state='download_verified', drive_id=drive_id)
    save(manifest, m)
    print(json.dumps({'state': m['state'], 'files': len(m['members'])}))


def cleanup(root, catalog, staging, number, manifest_drive_id):
    manifest = catalog / ('batch_%06d.json' % number)
    m = json.loads(manifest.read_text())
    assert m['state'] == 'download_verified'
    assert m.get('drive_id') and manifest_drive_id
    # Complete all content checks before the first deletion.
    for item in m['members']:
        assert not item['tracked']
        assert sha(source(root, item)) == item['sha256']
    allocated = 0
    with (catalog / 'deletion_journal.jsonl').open('a') as log:
        for item in m['members']:
            p = source(root, item)
            blocks = p.stat().st_blocks * 512
            r = dict(path=item['path'], sha256=item['sha256'], bytes=item['size'],
                     allocated_bytes=blocks, archive_drive_id=m['drive_id'])
            for state in ['verified_before_unlink', 'local_removed']:
                log.write(json.dumps(dict(r, state=state)) + '\n')
                log.flush()
                os.fsync(log.fileno())
                if state == 'verified_before_unlink':
                    p.unlink()
            allocated += blocks
    m.update(state='local_removed', removed_files=len(m['members']),
             removed_bytes=m['source_bytes'], removed_allocated_bytes=allocated,
             manifest_drive_id=manifest_drive_id)
    save(manifest, m)
    Path(m['archive']).unlink()
    for part in m.get('transport_parts', []):
        p = Path(part['path'])
        if p.resolve().parent != staging.resolve():
            raise ValueError('Unsafe transport staging path')
        p.unlink(missing_ok=True)
    (staging / ('restored_%06d.tar.gz' % number)).unlink()
    print(json.dumps({k:v for k,v in m.items() if k != 'members'}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['pack', 'verify', 'cleanup'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--number', type=int, required=True)
    p.add_argument('--limit-mib', type=int, default=448)
    p.add_argument('--drive-id')
    p.add_argument('--manifest-drive-id')
    a = p.parse_args()
    root = a.root.resolve()
    catalog, staging = root / 'cbwce_archive/catalogs', root / 'cbwce_archive/staging'
    if a.action == 'pack':
        pack(root, catalog, staging, a.number, a.limit_mib)
    elif a.action == 'verify':
        verify(root, catalog, staging, a.number, a.drive_id)
    else:
        cleanup(root, catalog, staging, a.number, a.manifest_drive_id)


if __name__ == '__main__':
    main()
