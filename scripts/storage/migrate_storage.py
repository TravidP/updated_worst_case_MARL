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


def linked_items(root, item, groups, excluded):
    primary = source(root, item)
    inode = (primary.stat().st_dev, primary.stat().st_ino)
    result = [dict(item)]
    for alias in groups.get(item['path'], []):
        if alias['path'] == item['path'] or alias['path'] in excluded:
            continue
        p = root / alias['path']
        if not p.exists():
            continue
        p = source(root, alias)
        if (p.stat().st_dev, p.stat().st_ino) != inode:
            raise ValueError('Hardlink relationship changed: ' + alias['path'])
        result.append(dict(alias, hardlink_to=item['path']))
    return result


def hardlink_index(catalog):
    p = catalog / 'hardlink_groups.json'
    if not p.exists():
        return {}
    groups = json.loads(p.read_text())
    return {item['path']: group for group in groups for item in group}


def pack(root, catalog, staging, number, limit_mib):
    name = 'batch_%06d' % number
    manifest = catalog / (name + '.json')
    if manifest.exists():
        m = json.loads(manifest.read_text())
        print(json.dumps({k:v for k,v in m.items() if k != 'members'}))
        return
    done = set()
    for p in catalog.glob('batch_*.json'):
        if not p.stem.removeprefix('batch_').isdigit():
            continue
        m = json.loads(p.read_text())
        if m['state'] in ('download_verified', 'local_removed_untracked_only', 'local_removed'):
            done.update(x['path'] for x in m['members'])
    output = staging / (name + '.tar.gz')
    if output.exists():
        raise ValueError('Unfinished archive exists; inspect before retrying')
    selected = []
    selected_names = set()
    groups = hardlink_index(catalog)
    limit = limit_mib * 1024**2
    with output.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=6, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w|', dereference=False) as archive:
                with (catalog / 'inventory.jsonl').open() as inventory:
                    for line in inventory:
                        item = json.loads(line)
                        if item['path'] in done or item['path'] in selected_names or item['tracked']:
                            continue
                        if selected and item['path'].split('/')[0] != selected[0]['path'].split('/')[0]:
                            break
                        p = root / item['path']
                        if not p.exists():
                            continue
                        if item['size'] + 1024**2 >= limit:
                            continue  # Oversized source handled by a separate multipart pass.
                        if selected and raw.tell() + item['size'] + 1024**2 > limit:
                            break
                        p = source(root, item)
                        items = linked_items(root, item, groups, done | selected_names)
                        content_hash = sha(p)
                        for entry in items:
                            entry['sha256'] = content_hash
                            archive.add(root / entry['path'], arcname=entry['path'], recursive=False)
                            selected.append(entry)
                            selected_names.add(entry['path'])
                        gz.flush()
    if not selected:
        output.unlink()
        print(json.dumps({'state': 'no_eligible_files'}))
        return
    assert output.stat().st_size <= 512 * 1024**2
    m = {'state': 'packed', 'archive': str(output), 'archive_sha256': sha(output),
         'archive_bytes': output.stat().st_size, 'source_bytes': sum(x['size'] for x in selected),
         'files': len(selected), 'members': selected, 'category': selected[0]['path'].split('/')[0]}
    save(manifest, m)
    print(json.dumps({k:v for k,v in m.items() if k != 'members'}))


def pack_large(root, catalog, staging, number):
    manifest = catalog / ('batch_%06d.json' % number)
    if manifest.exists():
        m = json.loads(manifest.read_text())
        print(json.dumps({k:v for k,v in m.items() if k != 'members'}))
        return
    done = set()
    for p in catalog.glob('batch_*.json'):
        if not p.stem.removeprefix('batch_').isdigit():
            continue
        m = json.loads(p.read_text())
        if m['state'] in ('download_verified', 'local_removed', 'local_removed_untracked_only'):
            done.update(x['path'] for x in m['members'])
    with (catalog / 'inventory.jsonl').open() as inventory:
        for line in inventory:
            item = json.loads(line)
            if item['tracked'] or item['path'] in done or item['size'] < 447 * 1024**2:
                continue
            if not (root / item['path']).exists():
                continue
            p = source(root, item)
            items = linked_items(root, item, hardlink_index(catalog), done)
            content_hash = sha(p)
            for entry in items:
                entry['sha256'] = content_hash
            item = items[0]
            m = dict(state='packed', archive=str(p), archive_kind='raw_file',
                     archive_is_source=True, archive_sha256=item['sha256'],
                     archive_bytes=item['size'], source_bytes=sum(x['size'] for x in items),
                     files=len(items), members=items, category=item['path'].split('/')[0])
            save(manifest, m)
            print(json.dumps({k:v for k,v in m.items() if k != 'members'}))
            return
    print(json.dumps({'state': 'no_eligible_files'}))


def verify(root, catalog, staging, number, drive_id):
    manifest = catalog / ('batch_%06d.json' % number)
    m = json.loads(manifest.read_text())
    restored = staging / ('restored_%06d.tar.gz' % number)
    assert sha(restored) == m['archive_sha256'], 'Downloaded archive hash mismatch'
    if m.get('archive_kind') == 'raw_file':
        for item in m['members']:
            assert restored.stat().st_size == item['size']
            assert m['archive_sha256'] == item['sha256']
    else:
        with tarfile.open(restored, 'r:gz') as t:
            members = t.getmembers()
            assert len(members) == len(m['members'])
            assert [x.name for x in members] == [x['path'] for x in m['members']]
            names = {x['path'] for x in m['members']}
            for item, member in zip(m['members'], members):
                assert (member.isfile() and member.size == item['size']) or (member.islnk() and member.linkname in names)
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
    # Resume only deletions evidenced by this archive's durable journal.
    prior = {}
    journal = catalog / 'deletion_journal.jsonl'
    if journal.exists():
        with journal.open() as history:
            for line in history:
                r = json.loads(line)
                if r.get('archive_drive_id') == m['drive_id']:
                    prior[r['path']] = r
    pending, allocated = [], 0
    for item in m['members']:
        if item['tracked']:
            continue
        p = root / item['path']
        if not p.exists():
            r = prior.get(item['path'])
            assert r and r['sha256'] == item['sha256'], 'Missing file without a durable deletion record'
            allocated += r['allocated_bytes']
            continue
        assert sha(source(root, item)) == item['sha256']
        pending.append(item)
    with journal.open('a') as log:
        # Publish complete intent before deleting a group, then fsync completion.
        # A crash is recoverable from intent + source existence without per-file fsync.
        for offset in range(0, len(pending), 100):
            records = []
            for item in pending[offset:offset + 100]:
                p = source(root, item)
                r = dict(path=item['path'], sha256=item['sha256'], bytes=item['size'],
                         allocated_bytes=0, archive_drive_id=m['drive_id'])
                records.append(r)
                log.write(json.dumps(dict(r, state='verified_before_unlink')) + '\n')
            log.flush()
            os.fsync(log.fileno())
            for r in records:
                p = root / r['path']
                stat = p.stat()
                r['allocated_bytes'] = stat.st_blocks * 512 if stat.st_nlink == 1 else 0
                p.unlink()
                allocated += r['allocated_bytes']
                log.write(json.dumps(dict(r, state='local_removed')) + '\n')
            log.flush()
            os.fsync(log.fileno())
    removable = [x for x in m['members'] if not x['tracked']]
    m.update(state='local_removed_untracked_only' if len(removable) != len(m['members']) else 'local_removed', removed_files=len(removable),
             removed_bytes=sum(x['size'] for x in removable), removed_allocated_bytes=allocated,
             manifest_drive_id=manifest_drive_id)
    save(manifest, m)
    if not m.get('archive_is_source'):
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
    p.add_argument('action', choices=['pack', 'pack-large', 'verify', 'cleanup'])
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
    elif a.action == 'pack-large':
        pack_large(root, catalog, staging, a.number)
    elif a.action == 'verify':
        verify(root, catalog, staging, a.number, a.drive_id)
    else:
        cleanup(root, catalog, staging, a.number, a.manifest_drive_id)


if __name__ == '__main__':
    main()
