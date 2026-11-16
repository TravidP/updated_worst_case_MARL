import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
import runpy
import sys
import hashlib
import os
from unittest.mock import patch
from pathlib import Path

spec = importlib.util.spec_from_file_location('migration', Path(__file__).parents[1] / 'scripts/storage/migrate_storage.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


restore_spec = importlib.util.spec_from_file_location('restore_archive', Path(__file__).parents[1] / 'scripts/storage/restore_archive.py')
restorer = importlib.util.module_from_spec(restore_spec)
restore_spec.loader.exec_module(restorer)


class ArchiveSafetyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = self.root / 'cbwce_archive/catalogs'
        self.staging = self.root / 'cbwce_archive/staging'
        self.catalog.mkdir(parents=True)
        self.staging.mkdir()
        self.source = self.root / 'results.txt'
        self.source.write_text('restorable result\n')
        stat = self.source.stat()
        item = dict(path='results.txt', size=stat.st_size, mtime_ns=stat.st_mtime_ns, tracked=False)
        (self.catalog / 'inventory.jsonl').write_text(json.dumps(item)+'\n')
        with contextlib.redirect_stdout(io.StringIO()):
            migration.pack(self.root, self.catalog, self.staging, 2, 2)
        self.manifest = self.catalog / 'batch_000002.json'
        self.restored = self.staging / 'restored_000002.tar.gz'
        shutil.copyfile(self.staging / 'batch_000002.tar.gz', self.restored)

    def tearDown(self):
        self.temp.cleanup()

    def test_rclone_worker_roundtrip_and_resume(self):
        import configparser
        remote = self.root / 'cloud'
        remote.mkdir()
        config = self.root / 'private.conf'
        config.write_text('[cbwce_archive]\nroot_folder_id = test-root\n')
        config.chmod(0o600)
        def fake_rc(command, **kwargs):
            action = next(x for x in command if x in ('copyto', 'lsjson'))
            args = command[command.index(action) + 1:]
            def path(value):
                return remote / value.split(':', 1)[1] if value.startswith('cbwce_archive:') else Path(value)
            if action == 'copyto':
                src, dst = map(path, args[:2])
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                return ''
            obj = path(args[0])
            return json.dumps([dict(Size=obj.stat().st_size, ID='test-' + obj.name)])
        script = Path(__file__).parents[1] / 'scripts/storage/rclone_migrate.py'
        argv = [str(script), '--root', str(self.root), '--rclone', 'fake-rclone',
                '--config', str(config), '--root-folder-id', 'test-root', '--start', '2', '--max-batches', '1']
        (self.catalog / 'migration.json').write_text('{}')
        with patch.dict(sys.modules, {'migrate_storage': migration}), patch.object(sys, 'argv', argv), \
             patch('subprocess.check_output', side_effect=fake_rc), patch('subprocess.run'), \
             patch('signal.signal'), contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(script), run_name='__main__')
            runpy.run_path(str(script), run_name='__main__')
        self.assertFalse(self.source.exists())
        self.assertEqual(json.loads(self.manifest.read_text())['state'], 'local_removed')
        restored = self.root / 'offline_restore'
        restorer.restore(remote / '02_legacy_results/batch_000002.manifest.json',
                         remote / '02_legacy_results', restored)
        self.assertEqual((restored / 'results.txt').read_text(), 'restorable result\n')

    def test_corrupt_download_never_allows_cleanup(self):
        self.restored.write_bytes(b'corrupted')
        with self.assertRaises(AssertionError):
            migration.verify(self.root, self.catalog, self.staging, 2, 'test-only-id')
        with self.assertRaises(AssertionError):
            migration.cleanup(self.root, self.catalog, self.staging, 2, 'test-only-manifest')
        self.assertTrue(self.source.exists())

    def test_changed_source_is_preserved(self):
        with contextlib.redirect_stdout(io.StringIO()):
            migration.verify(self.root, self.catalog, self.staging, 2, 'test-only-id')
        self.source.write_text('new result must survive')
        with self.assertRaises(ValueError):
            migration.cleanup(self.root, self.catalog, self.staging, 2, 'test-only-manifest')
        self.assertTrue(self.source.exists())

    def test_resume_after_durable_intent_and_unlink(self):
        with contextlib.redirect_stdout(io.StringIO()):
            migration.verify(self.root, self.catalog, self.staging, 2, 'test-only-id')
        m = json.loads(self.manifest.read_text())
        item = m['members'][0]
        record = dict(path=item['path'], sha256=item['sha256'], bytes=item['size'],
                      allocated_bytes=self.source.stat().st_blocks * 512,
                      archive_drive_id='test-only-id', state='verified_before_unlink')
        (self.catalog / 'deletion_journal.jsonl').write_text(json.dumps(record)+'\n')
        self.source.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            migration.cleanup(self.root, self.catalog, self.staging, 2, 'test-only-manifest')
        self.assertEqual(json.loads(self.manifest.read_text())['state'], 'local_removed')

    def test_parallel_fragment_restore(self):
        data = (self.staging / 'batch_000002.tar.gz').read_bytes()
        chunks = [data[:30], data[30:70], data[70:]]
        urls = ['https://test.invalid/' + str(i) for i in range(3)]
        m = json.loads(self.manifest.read_text())
        m['transport_parts'] = [dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest()) for b in chunks]
        self.manifest.write_text(json.dumps(m))
        argv = ['archive_transport.py', 'download', '--root', str(self.root), '--number', '2', '--json', json.dumps(urls)]
        script = Path(__file__).parents[1] / 'scripts/storage/archive_transport.py'
        with patch.dict(sys.modules, {'migrate_storage': migration}), patch.object(sys, 'argv', argv), \
             patch('urllib.request.urlopen', side_effect=lambda url, timeout: io.BytesIO(chunks[urls.index(url)])), \
             contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(script), run_name='__main__')
        self.assertEqual(self.restored.read_bytes(), data)

    def test_raw_file_restore_and_single_delete(self):
        data = self.source.read_bytes()
        m = json.loads(self.manifest.read_text())
        m.update(archive=str(self.source), archive_kind='raw_file', archive_is_source=True,
                 archive_sha256=hashlib.sha256(data).hexdigest(), archive_bytes=len(data))
        self.manifest.write_text(json.dumps(m))
        self.restored.write_bytes(data)
        restore_manifest = self.catalog / 'raw_restore.json'
        restore_m = dict(m, members=m['members'] + [dict(m['members'][0], path='second.txt')])
        restore_manifest.write_text(json.dumps(restore_m))
        raw_tree = self.root / 'raw-restore'
        restorer.restore(restore_manifest, self.root, raw_tree)
        self.assertEqual((raw_tree / 'results.txt').stat().st_ino, (raw_tree / 'second.txt').stat().st_ino)
        with contextlib.redirect_stdout(io.StringIO()):
            migration.verify(self.root, self.catalog, self.staging, 2, 'test-only-raw')
            migration.cleanup(self.root, self.catalog, self.staging, 2, 'test-only-manifest')
        self.assertFalse(self.source.exists())

    def check_hardlinks(self, tracked_alias):
        alias = self.root / 'alias.txt'
        os.link(self.source, alias)
        original = json.loads((self.catalog / 'inventory.jsonl').read_text())
        linked = dict(original, path='alias.txt', tracked=tracked_alias)
        (self.catalog / 'inventory.jsonl').write_text(json.dumps(original)+'\n'+json.dumps(linked)+'\n')
        (self.catalog / 'hardlink_groups.json').write_text(json.dumps([[original, linked]]))
        blocks = self.source.stat().st_blocks * 512
        with contextlib.redirect_stdout(io.StringIO()):
            migration.pack(self.root, self.catalog, self.staging, 3, 2)
            shutil.copyfile(self.staging / 'batch_000003.tar.gz', self.staging / 'restored_000003.tar.gz')
            migration.verify(self.root, self.catalog, self.staging, 3, 'test-hardlinks')
            restored_tree = self.root / 'hardlink-restore'
            restorer.restore(self.catalog / 'batch_000003.json', self.staging, restored_tree)
            self.assertEqual((restored_tree / 'results.txt').stat().st_ino, (restored_tree / 'alias.txt').stat().st_ino)
            migration.cleanup(self.root, self.catalog, self.staging, 3, 'test-hardlink-manifest')
        m = json.loads((self.catalog / 'batch_000003.json').read_text())
        self.assertEqual(len(m['members']), 2)
        self.assertFalse(self.source.exists())
        self.assertEqual(alias.exists(), tracked_alias)
        self.assertEqual(m['removed_allocated_bytes'], 0 if tracked_alias else blocks)

    def test_hardlinks_count_blocks_only_once(self):
        self.check_hardlinks(False)

    def test_tracked_hardlink_is_archived_but_preserved(self):
        self.check_hardlinks(True)

    def test_offline_restore_and_refuse_overwrite(self):
        output = self.root / 'restored-tree'
        result = restorer.restore(self.manifest, self.staging, output)
        self.assertEqual(result['files'], 1)
        self.assertEqual((output / 'results.txt').read_bytes(), self.source.read_bytes())
        with self.assertRaises(FileExistsError):
            restorer.restore(self.manifest, self.staging, output)

    def test_restore_rejects_traversal_before_writing(self):
        m = json.loads(self.manifest.read_text())
        m['members'][0]['path'] = '../outside.txt'
        self.manifest.write_text(json.dumps(m))
        output = self.root / 'restored-tree'
        with self.assertRaises(ValueError):
            restorer.restore(self.manifest, self.staging, output)
        self.assertFalse(output.exists())

    def test_verified_cleanup_records_exact_file(self):
        with contextlib.redirect_stdout(io.StringIO()):
            migration.verify(self.root, self.catalog, self.staging, 2, 'test-only-id')
            migration.cleanup(self.root, self.catalog, self.staging, 2, 'test-only-manifest')
        self.assertFalse(self.source.exists())
        m = json.loads(self.manifest.read_text())
        self.assertEqual(m['removed_files'], 1)
        self.assertEqual(m['state'], 'local_removed')
        lines = (self.catalog / 'deletion_journal.jsonl').read_text().splitlines()
        self.assertEqual([json.loads(x)['state'] for x in lines], ['verified_before_unlink', 'local_removed'])


if __name__ == '__main__':
    unittest.main()
