"""Checksummed, relocatable releases. Archives never delete their source."""
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from experiments.core import file_hash
from experiments.storage import atomic_json
from experiments.protocol import ROOT,root

def inventory(source):
    result={}
    for p in sorted(Path(source).rglob('*')):
        if any(part in ('.git','__pycache__','.venv') for part in p.parts):continue
        if p.is_symlink():raise ValueError('Release must not depend on symlinks: '+str(p))
        if p.is_file():result[str(p.relative_to(source))]=file_hash(p)
    return result

def package(kind,source,output):
    source=Path(source).resolve() if source else (ROOT if kind=='code' else root('data'))
    output=Path(output).resolve()
    if not source.is_dir():raise ValueError('Missing release input')
    if output.exists() or source in output.parents:raise ValueError('Output must be new and outside source')
    if kind=='run' and not (source/'result.json').exists():raise ValueError('Run archive requires a result record')
    output.parent.mkdir(parents=True,exist_ok=True);files=inventory(source)
    fd,tmp=tempfile.mkstemp(dir=str(output.parent),suffix='.partial');os.close(fd)
    try:
        with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:
            for relative in files:tar.add(str(source/relative),arcname=relative,recursive=False)
        with tarfile.open(tmp,'r:gz') as tar:
            import hashlib
            for relative,expected in files.items():
                if hashlib.sha256(tar.extractfile(relative).read()).hexdigest()!=expected:raise ValueError('Archive verification failed')
        os.rename(tmp,str(output))
        atomic_json(str(output)+'.manifest.json',dict(version='8.1',kind=kind,files=files,archive_sha256=file_hash(output)))
        return str(output)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def archive(source,destination):
    """Login-side transfer: copy to destination filesystem, verify, then rename."""
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination.exists() or source==destination or source in destination.parents:raise ValueError('Archive destination must be new and separate')
    files=inventory(source);destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='.archive-',dir=str(destination.parent)))
    try:
        for relative in files:
            target=tmp/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,target)
        if inventory(tmp)!=files:raise ValueError('Archive transfer verification failed')
        atomic_json(tmp/'archive_inventory.json',files);os.rename(str(tmp),str(destination))
        return str(destination)
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise
