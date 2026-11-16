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
from unittest.mock import patch
from pathlib import Path

spec = importlib.util.spec_from_file_location('migration', Path(__file__).parents[1] / 'scripts/storage/migrate_storage.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


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
        with contextlib.redirect_stdout(io.StringIO()):
            migration.verify(self.root, self.catalog, self.staging, 2, 'test-only-raw')
            migration.cleanup(self.root, self.catalog, self.staging, 2, 'test-only-manifest')
        self.assertFalse(self.source.exists())

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
