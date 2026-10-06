#!/usr/bin/env python3
"""Restore checksum-verified display CSVs without overwriting conflicts."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def restore(archive, site):
    release = json.loads((site / 'release.json').read_text())
    if archive.stat().st_size != release['bytes'] or digest(archive.read_bytes()) != release['sha256']:
        raise ValueError('Archive size/SHA-256 mismatch / 归档校验失败')
    pending = []
    with zipfile.ZipFile(archive) as z:
        prefix = release['archiveRoot'] + '/'
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        manifest = json.loads(z.read(prefix + 'FILES_SHA256.json'))['files']
        for relative, expected in manifest.items():
            path = PurePosixPath(relative)
            if path.is_absolute() or '..' in path.parts or chr(92) in relative:
                raise ValueError('Unsafe archive path')
            data = z.read(prefix + relative)
            if digest(data) != expected:
                raise ValueError('Member checksum mismatch: ' + relative)
            if not relative.startswith('dist/data/'):
                continue
            target = site / relative
            if target.is_symlink() or not target.resolve().is_relative_to(site.resolve()):
                raise ValueError('Unsafe destination: ' + relative)
            if target.exists():
                if digest(target.read_bytes()) != expected:
                    raise ValueError('Existing file differs; refusing overwrite: ' + relative)
                continue
            if ('/series/' in relative and path.suffix == '.csv') or path.name == 'rollout_metrics.csv':
                pending.append((target, data))
            else:
                raise ValueError('Missing catalog/summary; restore repository first: ' + relative)
        series = [p for p in manifest if '/series/' in p and p.endswith('.csv')]
        if len(series) != release['seriesCsvFiles']:
            raise ValueError('Unexpected series count')
        # All checks complete before writing any destination.
        for target, data in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output:
                output.write(data)
    print('Verified / 校验通过: {} series; restored / 已恢复: {} files'.format(len(series), len(pending)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', required=True, type=Path)
    args = parser.parse_args()
    try:
        restore(args.zip, Path(__file__).resolve().parent)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, 'Restore failed / 恢复失败: {}\n'.format(error))


if __name__ == '__main__':
    main()
