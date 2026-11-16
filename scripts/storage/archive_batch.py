"""Pack a bounded batch from an explicit inventory; never deletes source data."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit-mib', type=int, default=256)
    args = parser.parse_args()
    root = args.root.resolve()
    selected, total = [], 0
    for line in args.inventory.read_text().splitlines():
        item = json.loads(line)
        path = root / item['path']
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Unsafe inventory path')
        if not path.exists():
            continue
        if item['size'] > args.limit_mib * 1024**2:
            continue  # Large individual files require a separate split protocol.
        if total + item['size'] > args.limit_mib * 1024**2:
            break
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (item['size'], item['mtime_ns']):
            raise ValueError('Source changed: ' + str(path))
        item['sha256'] = digest(path)
        selected.append(item)
        total += item['size']
    if not selected:
        raise ValueError('No eligible files')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, 'w:gz', compresslevel=1) as archive:
        for item in selected:
            archive.add(root / item['path'], arcname=item['path'], recursive=False)
    # Re-read every packed member; never trust a source hash alone.
    with tarfile.open(args.output, 'r:gz') as archive:
        for item in selected:
            h = hashlib.sha256()
            stream = archive.extractfile(item['path'])
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
            if h.hexdigest() != item['sha256']:
                raise ValueError('Packed member changed: ' + item['path'])
    manifest = {'state': 'packed', 'archive': args.output.name,
                'archive_sha256': digest(args.output),
                'archive_bytes': args.output.stat().st_size,
                'source_bytes': total, 'members': selected}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='members'}))


if __name__ == '__main__':
    main()
