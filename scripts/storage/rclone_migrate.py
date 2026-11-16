"""Single-writer, verified rclone migration. Never deletes remote objects."""
import argparse
import configparser
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from migrate_storage import pack, pack_large, verify, cleanup, save, sha

DONE = {'local_removed', 'local_removed_untracked_only'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--rclone', required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--remote', default='cbwce_archive')
    p.add_argument('--root-folder-id', required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--max-batches', type=int, default=1000)
    a = p.parse_args()
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(a.config)
    if cfg[a.remote].get('root_folder_id') != a.root_folder_id:
        raise ValueError('Remote archive root mismatch')
    if a.config.stat().st_mode & 0o077:
        raise ValueError('Credential file must be private')
    root = a.root.resolve()
    catalog = root / 'cbwce_archive/catalogs'
    staging = root / 'cbwce_archive/staging'
    lock = (catalog / 'migration.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = []
    signal.signal(signal.SIGUSR1, lambda *_: stop.append(True))
    env = dict(os.environ, XDG_CACHE_HOME='/tmp/cbwce-rclone-cache')
    base = [a.rclone, '--config', str(a.config), '--cache-dir', '/tmp/cbwce-rclone-cache',
            '--retries', '5', '--low-level-retries', '10']
    def rc(*args):
        return subprocess.check_output(base + list(map(str, args)), env=env, text=True)
    def state(phase, n):
        save(catalog / 'rclone_worker.json', dict(pid=os.getpid(), phase=phase,
             batch=n, updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        print(json.dumps(dict(phase=phase, batch=n)), flush=True)
    def metadata(remote, size):
        rows = json.loads(rc('lsjson', remote))
        if len(rows) != 1 or rows[0]['Size'] != size or not rows[0].get('ID'):
            raise ValueError('Remote metadata mismatch')
        return rows[0]['ID']
    n = a.start
    try:
        for n in range(a.start, a.start + a.max_batches):
            if stop:
                break
            manifest = catalog / ('batch_%06d.json' % n)
            if manifest.exists() and json.loads(manifest.read_text())['state'] in DONE:
                continue
            state('packing', n)
            pack(root, catalog, staging, n, 448)
            if not manifest.exists():
                pack_large(root, catalog, staging, n)
            if not manifest.exists():
                state('inventory_untracked_complete', n)
                return
            m = json.loads(manifest.read_text())
            if m.get('transport_parts'):
                raise ValueError('Finish connector batch before rclone takeover')
            folder = '03_figures_reports' if m['category'] == 'evaluation results figs' else '02_legacy_results'
            filename = 'batch_%06d%s' % (n, '.raw' if m.get('archive_kind') == 'raw_file' else '.tar.gz')
            remote = a.remote + ':' + folder + '/' + filename
            cloud_manifest = catalog / ('batch_%06d_rclone_manifest.json' % n)
            cloud_name = a.remote + ':' + folder + '/batch_%06d.manifest.json' % n
            if m['state'] != 'download_verified':
                state('uploading', n)
                if sha(Path(m['archive'])) != m['archive_sha256']:
                    raise ValueError('Local archive changed')
                rc('copyto', m['archive'], remote, '--checksum', '--immutable')
                drive_id = metadata(remote, m['archive_bytes'])
                m.update(archive_file=filename, remote_path=remote, drive_id=drive_id)
                save(manifest, m)
                if not cloud_manifest.exists():
                    save(cloud_manifest, dict(m, state='uploaded'))
                state('publishing_manifest', n)
                rc('copyto', cloud_manifest, cloud_name, '--checksum', '--immutable')
                manifest_id = metadata(cloud_name, cloud_manifest.stat().st_size)
                checked_manifest = staging / ('manifest_%06d.checked.json' % n)
                rc('copyto', cloud_name, checked_manifest, '--ignore-times')
                if sha(checked_manifest) != sha(cloud_manifest):
                    raise ValueError('Downloaded manifest mismatch')
                checked_manifest.unlink()
                m['manifest_drive_id'] = manifest_id
                save(manifest, m)
                state('downloading_verification', n)
                rc('copyto', remote, staging / ('restored_%06d.tar.gz' % n), '--ignore-times')
                state('verifying', n)
                verify(root, catalog, staging, n, drive_id)
            m = json.loads(manifest.read_text())
            state('cleaning_verified_sources', n)
            cleanup(root, catalog, staging, n, m['manifest_drive_id'])
            subprocess.run([sys.executable, str(Path(__file__).with_name('status.py')), '--root', str(root)], check=True)
            rc('copyto', catalog / 'migration.json', a.remote + ':00_handover/migration_progress.json', '--checksum')
            state('batch_complete', n)
        state('stopped_at_batch_boundary', n)
    except Exception as exc:
        state('failed_' + type(exc).__name__, n)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
