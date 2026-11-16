"""Summarize durable archive manifests without treating uploads as completed cleanup."""
import argparse
import json
import os
import subprocess
from pathlib import Path
from migrate_storage import save

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
root = a.root.resolve()
catalog = root / 'cbwce_archive/catalogs'
status_file = catalog / 'migration.json'
status = json.loads(status_file.read_text())
batches = []
for path in sorted(catalog.glob('batch_*.json')):
    if not path.stem.removeprefix('batch_').isdigit():
        continue
    m = json.loads(path.read_text())
    batches.append(dict(batch=path.stem, state=m['state'], files=m.get('files', len(m['members'])),
                        source_bytes=m['source_bytes'], removed_bytes=m.get('removed_bytes', 0),
                        removed_files=m.get('removed_files', 0),
                        removed_allocated_bytes=m.get('removed_allocated_bytes', 0),
                        manifest_drive_id=m.get('manifest_drive_id')))
# The pilot used an older manifest schema; allocation evidence is in its journal.
if batches and batches[0]['batch']=='batch_000001' and not batches[0]['removed_allocated_bytes']:
    with (catalog / 'deletion_journal.jsonl').open() as f:
        batches[0]['removed_allocated_bytes'] = sum(r['allocated_bytes'] for r in map(json.loads, f)
            if r.get('archive_drive_id')=='1z3UNwIeJvLUvgVJIaCvcuEWkJzICyD0J' and r['state']=='local_removed')
stat = os.statvfs(root)
git = subprocess.run(['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'], capture_output=True, text=True)
status['commit'] = git.stdout.strip() if git.returncode == 0 else None
status.update(status='migration_in_progress', remote_branch_published=True,
              drive_capacity='User confirmed unlimited institutional storage', batches=batches,
              local_removed_bytes=sum(x['removed_bytes'] for x in batches),
              local_removed_files=sum(x['removed_files'] for x in batches),
              removed_allocated_bytes=sum(x['removed_allocated_bytes'] for x in batches),
              available_bytes_after=stat.f_bavail*stat.f_frsize)
save(status_file, status)
print(json.dumps({k:status[k] for k in ['status','local_removed_bytes','local_removed_files','removed_allocated_bytes','available_bytes_after']}))
