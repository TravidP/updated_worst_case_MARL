"""Small synthetic archives only; no SUMO, evaluation or real data mutation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('restore_results', ROOT / 'docs/evaluation_workbook/grid_results_site/restore_results.py')
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.site = Path(self.temp.name) / 'site'
        (self.site / 'dist/data').mkdir(parents=True)
        self.csv = b'time,queue_min,queue_mean,queue_max,n\n0,1,2,3,10\n'
        files = {'dist/data/catalog.json': b'{}', 'dist/data/series/sample.csv': self.csv}
        (self.site / 'dist/data/catalog.json').write_bytes(b'{}')
        self.archive = Path(self.temp.name) / 'sample.zip'
        with zipfile.ZipFile(self.archive, 'w') as z:
            for name, data in files.items():
                z.writestr('sample/' + name, data)
            z.writestr('sample/FILES_SHA256.json', json.dumps({'files': {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}))
        (self.site / 'release.json').write_text(json.dumps({'archiveRoot': 'sample', 'bytes': self.archive.stat().st_size, 'sha256': hashlib.sha256(self.archive.read_bytes()).hexdigest(), 'seriesCsvFiles': 1}))

    def test_restore_and_idempotence(self):
        restore.restore(self.archive, self.site)
        restore.restore(self.archive, self.site)
        self.assertEqual((self.site / 'dist/data/series/sample.csv').read_bytes(), self.csv)

    def test_conflict_prevents_writes(self):
        (self.site / 'dist/data/catalog.json').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            restore.restore(self.archive, self.site)
        self.assertFalse((self.site / 'dist/data/series/sample.csv').exists())

    def test_corrupt_archive_rejected(self):
        with self.archive.open('ab') as f:
            f.write(b'corrupt')
        with self.assertRaises(ValueError):
            restore.restore(self.archive, self.site)


if __name__ == '__main__':
    unittest.main()
