#!/usr/bin/env python3
"""Build a complete portable ZIP from the original results-site checkout."""
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
NAME = 'CBWCE_Results_Portable_20261007'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def csv_count(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = csv.reader(stream)
        header = next(rows)
        if not header:
            raise ValueError('Missing CSV header: ' + str(path))
        return sum(1 for row in rows if row)


def validate_data(root):
    data = root / 'dist' / 'data'
    registry = json.loads((data / 'evaluation_sets.json').read_text(encoding='utf-8'))
    catalogs = []
    for path in sorted(data.rglob('catalog.json')):
        catalog = json.loads(path.read_text(encoding='utf-8'))
        counts = catalog['counts']
        series = list((path.parent / 'series').rglob('*.csv'))
        assert len(series) == counts['groups'], str(path)
        assert csv_count(path.parent / 'metrics_summary.csv') == counts['groups'], str(path)
        assert csv_count(path.parent / 'rollout_metrics.csv') == counts['rollouts'], str(path)
        for scenario in catalog['scenarios']:
            for controller in catalog['controllers']:
                for method in catalog['methods']:
                    expected = path.parent / 'series' / controller['id'] / method['id'] / scenario['split'] / (scenario['id'] + '.csv')
                    assert expected.is_file(), str(expected)
        for series_path in series:
            assert csv_count(series_path) == counts['stepsPerRollout'], str(series_path)
        catalogs.append({'catalog': path.relative_to(root).as_posix(), 'counts': counts})
    for evaluation in registry['evaluationSets']:
        for network in evaluation['networks']:
            if not network['available']:
                continue
            path = root / 'dist' / network['base'] / 'catalog.json'
            catalog = json.loads(path.read_text(encoding='utf-8'))
            assert catalog['counts']['rollouts'] == network['expectedRollouts']
            assert catalog['counts']['groups'] == network.get('expectedGroups', 460)
    csv_files = list(data.rglob('*.csv'))
    series_files = [p for p in csv_files if 'series' in p.parts]
    assert len(csv_files) == 993, len(csv_files)
    assert len(series_files) == 980, len(series_files)
    return {'snapshotDate': '2026-10-07', 'interfaceLanguage': 'en', 'csvFiles': len(csv_files), 'timeSeriesCsvFiles': len(series_files), 'catalogs': catalogs}


def main():
    output = SITE / 'packages'
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cbwce-portable-build-', dir='/tmp') as temporary:
        root = Path(temporary) / NAME
        root.mkdir()
        shutil.copytree(SITE / 'dist', root / 'dist')
        for path in (root / 'dist').rglob('*'):
            if path.is_symlink():
                raise ValueError('Do not package symlinks: ' + str(path))
        print('Validating every catalog and CSV...', flush=True)
        info = validate_data(root)
        for name in ['README.md', 'verify_package.py', 'Start-Windows.cmd']:
            shutil.copyfile(HERE / name, root / name)
        # Use CRLF for the Windows command file.
        command = (root / 'Start-Windows.cmd').read_text().replace('\r\n', '\n')
        (root / 'Start-Windows.cmd').write_bytes(command.replace('\n', '\r\n').encode('utf-8'))
        for name in ['Start-Linux.sh', 'Start-macOS.command']:
            shutil.copyfile(HERE / 'start.sh', root / name)
            (root / name).chmod(0o755)
        shutil.copyfile(SITE / 'server.py', root / 'server.py')
        source = root / 'source'
        source.mkdir()
        shutil.copyfile(HERE / 'launcher.go', source / 'launcher.go')
        shutil.copyfile(HERE / 'build_package.py', source / 'build_package.py')
        (source / 'README.md').write_text('''# Launcher source and rebuild\n\nThe launcher uses only Go standard-library packages. Build it with Go:\n\n```sh\nGOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -o Start-CBWCE.exe launcher.go\nGOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -o cbwce-viewer launcher.go\nGOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go build -trimpath -o cbwce-viewer launcher.go\n```\n\nUse GOARCH=arm64 for Windows/Linux ARM64, and GOARCH=amd64 for Intel macOS. Place the executable beside the extracted dist folder or under bin/<platform>-<architecture>/.\n\nIn the original project checkout, run `python3 docs/evaluation_workbook/grid_results_site/portable/build_package.py` to recreate the ZIP with all launchers and checksum files. That builder needs the original portable templates and dist data. Go is required for rebuilding, but not for running the included executables.\n''', encoding='utf-8')
        targets = [('windows', 'amd64'), ('windows', 'arm64'), ('linux', 'amd64'), ('linux', 'arm64'), ('darwin', 'amd64'), ('darwin', 'arm64')]
        def compile_target(target):
            system, architecture = target
            platform = 'macos' if system == 'darwin' else system
            executable = root / 'Start-CBWCE.exe' if target == ('windows', 'amd64') else root / 'bin' / (platform + '-' + architecture) / ('cbwce-viewer.exe' if system == 'windows' else 'cbwce-viewer')
            executable.parent.mkdir(parents=True, exist_ok=True)
            environment = dict(os.environ, GOOS=system, GOARCH=architecture, CGO_ENABLED='0', GO111MODULE='off', GOCACHE=str(Path(temporary) / 'go-cache'))
            subprocess.run([os.environ.get('CBWCE_GO_BINARY', 'go'), 'build', '-trimpath', '-ldflags=-s -w', '-o', str(executable), str(HERE / 'launcher.go')], env=environment, check=True)
            executable.chmod(0o755)
            return platform + '-' + architecture
        print('Building six standalone native launchers...', flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(compile_target, targets):
                print('Built ' + result, flush=True)
        info['nativeLaunchers'] = ['windows-amd64', 'windows-arm64', 'linux-amd64', 'linux-arm64', 'macos-amd64', 'macos-arm64']
        (root / 'PACKAGE_INFO.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
        files = sorted(path for path in root.rglob('*') if path.is_file())
        manifest = {'algorithm': 'sha256', 'files': {path.relative_to(root).as_posix(): sha256(path) for path in files}}
        (root / 'FILES_SHA256.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        subprocess.run(['python3', str(root / 'verify_package.py')], check=True)
        archive = output / (NAME + '.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as package:
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    member = zipfile.ZipInfo(NAME + '/' + path.relative_to(root).as_posix(), date_time=(2026, 10, 7, 12, 0, 0))
                    member.create_system = 3
                    member.external_attr = (0o100755 if os.access(path, os.X_OK) else 0o100644) << 16
                    package.writestr(member, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        with zipfile.ZipFile(archive) as package:
            assert package.testzip() is None
            assert len(package.namelist()) == len(manifest['files']) + 1
            # Verify each compressed member against the original dist or launcher.
            for relative, digest in manifest['files'].items():
                assert hashlib.sha256(package.read(NAME + '/' + relative)).hexdigest() == digest, relative
        digest = sha256(archive)
        (output / (NAME + '.zip.sha256')).write_text(digest + '  ' + archive.name + '\n', encoding='ascii')
        shutil.copyfile(HERE / 'README.md', output / (NAME + '_README.md'))
        print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest, 'files': len(manifest['files']) + 1, 'csvFiles': info['csvFiles'], 'seriesFiles': info['timeSeriesCsvFiles']}), flush=True)

if __name__ == '__main__':
    main()
