import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('migration', Path(__file__).parents[1] / 'scripts/migrate_storage.py')
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
