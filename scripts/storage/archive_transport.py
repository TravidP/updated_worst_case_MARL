"""Split/restore archive transport fragments; never deletes original results."""
import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from migrate_storage import save

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('action', choices=['split', 'record', 'download'])
p.add_argument('--root', type=Path, required=True)
p.add_argument('--number', type=int, required=True)
p.add_argument('--json', default='[]')
a = p.parse_args()
root = a.root.resolve()
staging = root / 'cbwce_archive/staging'
manifest = root / ('cbwce_archive/catalogs/batch_%06d.json' % a.number)
m = json.loads(manifest.read_text())
if a.action == 'split':
    parts = []
    with open(m['archive'], 'rb') as f:
        index = 0
        while True:
            chunk = f.read(95 * 1024**2)
            if not chunk:
                break
            path = staging / ('batch_%06d.tar.gz.part%03d' % (a.number, index))
            path.write_bytes(chunk)
            parts.append(dict(path=str(path), bytes=len(chunk), sha256=hashlib.sha256(chunk).hexdigest()))
            index += 1
    m['transport_parts'] = parts
    m['transport_protocol'] = ('Concatenate transport_parts in listed order and check archive_sha256; restore identical bytes to every members[].path (hardlink_to may be recreated), without tar extraction.' if m.get('archive_kind') == 'raw_file' else 'Concatenate transport_parts in listed order, check archive_sha256, then extract and verify member hashes.')
    save(manifest, m)
    print(json.dumps(parts))
elif a.action == 'record':
    ids = json.loads(a.json)
    assert len(ids) == len(m['transport_parts'])
    for part, file_id in zip(m['transport_parts'], ids):
        part['drive_id'] = file_id
    m['state'] = 'uploaded'
    save(manifest, m)
    print('Upload identities recorded')
else:
    urls = json.loads(a.json)
    assert len(urls) == len(m['transport_parts'])
    restored = staging / ('restored_%06d.tar.gz' % a.number)
    offsets = []
    total = 0
    for part in m['transport_parts']:
        offsets.append(total)
        total += part['bytes']
    with restored.open('wb') as out:
        out.truncate(total)
    def download_one(index):
        url, part = urls[index], m['transport_parts'][index]
        assert url.startswith('https://')
        h, size = hashlib.sha256(), 0
        with urllib.request.urlopen(url, timeout=60) as response, restored.open('r+b') as out:
            out.seek(offsets[index])
            for block in iter(lambda: response.read(1024 * 1024), b''):
                size += len(block)
                assert size <= part['bytes'], 'Fragment exceeds its assigned range'
                h.update(block)
                out.write(block)
        assert h.hexdigest() == part['sha256'] and size == part['bytes']
    with ThreadPoolExecutor(max_workers=min(4, len(urls))) as pool:
        list(pool.map(download_one, range(len(urls))))
    print('All transport fragments verified')
