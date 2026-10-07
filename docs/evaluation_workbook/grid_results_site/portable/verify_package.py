#!/usr/bin/env python3
"""Optional checksum verification, requiring only Python 3.8+ standard library."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'FILES_SHA256.json').read_text(encoding='utf-8'))
failed = []
for relative, expected in manifest['files'].items():
    path = root / relative
    if not path.is_file():
        failed.append(relative + ': missing')
        continue
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        failed.append(relative + ': checksum mismatch')
if failed:
    print('\n'.join(failed))
    sys.exit(1)
print('Verified all {} packaged files.'.format(len(manifest['files'])))
