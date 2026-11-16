#!/usr/bin/env python3
"""Whitelist a source release, excluding images, caches and runtime outputs."""
import argparse
import fnmatch
import json
import shutil
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.release import package

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
rule=json.loads((ROOT/'packaging/release_allowlist.json').read_text())
with tempfile.TemporaryDirectory(prefix='cbwce-code-release-') as tmp:
    stage=Path(tmp);candidates=[ROOT/f for f in rule['files']]
    for directory in rule['directories']:candidates.extend((ROOT/directory).rglob('*'))
    for source in candidates:
        relative=source.relative_to(ROOT)
        if any(part in rule['excluded_directories'] for part in relative.parts):continue
        if any(source.name.endswith(s) for s in rule['excluded_suffixes']):continue
        if any(fnmatch.fnmatch(source.name, pattern) for pattern in rule.get('excluded_patterns', [])):continue
        if source.is_symlink():raise ValueError('External symlinks are not release inputs')
        if not source.is_file():continue
        target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    print(package('code',stage,a.output))
