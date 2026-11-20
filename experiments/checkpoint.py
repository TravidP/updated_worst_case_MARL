"""Atomic immutable bundles, inference-only restore, and reference-aware retention."""
import fcntl
import json
import os
import pickle
import shutil
import uuid
from pathlib import Path
from experiments.core import digest, file_hash
from experiments.storage import atomic_json

def inspect_checkpoint(path):
    path=Path(path).resolve()
    manifest=json.loads((path/'manifest.json').read_text())
    body=dict(manifest);claimed=body.pop('hash')
    if manifest['version']!='8.1' or digest(body)!=claimed:raise ValueError('Invalid checkpoint manifest')
    for relative,expected in manifest['files'].items():
        target=(path/relative).resolve()
        if path not in target.parents or not target.is_file() or file_hash(target)!=expected:
            raise ValueError('Checkpoint integrity failure: '+relative)
    return manifest

def save_checkpoint(path, models, state, kind='full', parents=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(str(path))
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    tmp.mkdir()
    try:
        signatures={}
        for name,model in models.items():
            directory=tmp/name;directory.mkdir()
            saver=model.saver if kind=='full' else model.inference_saver
            with model.graph.as_default():
                saver.save(model.sess,str(directory/'variables'),write_meta_graph=False,write_state=False)
            signatures[name]=model.signature
            if kind=='full':
                if not model.training:raise ValueError('Cannot serialize frozen model as full training state')
                with (directory/'state.pkl').open('wb') as f:pickle.dump(model.state(),f,protocol=4)
                replay=getattr(model,'replay',None)
                if hasattr(replay,'save'):replay.save(directory/'replay')
        with (tmp/'runner.pkl').open('wb') as f:pickle.dump(state,f,protocol=4)
        manifest=dict(version='8.1',kind=kind,signatures=signatures,parents=parents or {},
                      files={str(p.relative_to(tmp)):file_hash(p) for p in sorted(tmp.rglob('*')) if p.is_file()})
        manifest['hash']=digest(manifest);atomic_json(tmp/'manifest.json',manifest)
        inspect_checkpoint(tmp)
        os.rename(str(tmp),str(path))
        return manifest['hash']
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise

def load_checkpoint(path, models, inference=False):
    path=Path(path);manifest=inspect_checkpoint(path)
    if not inference and manifest['kind']!='full':raise ValueError('Inference snapshot cannot resume training')
    for name,model in models.items():
        if manifest['signatures'].get(name)!=model.signature:raise ValueError('Model signature mismatch: '+name)
        if not inference and not model.training:raise ValueError('Full restore requires training model')
        with model.graph.as_default():
            (model.inference_saver if inference else model.saver).restore(model.sess,str(path/name/'variables'))
        if not inference:
            with (path/name/'state.pkl').open('rb') as f:model.restore_state(pickle.load(f))
            replay=getattr(model,'replay',None)
            if hasattr(replay,'load'):replay.load(path/name/'replay')
    with (path/'runner.pkl').open('rb') as f:return pickle.load(f)

class Checkpoints:
    def __init__(self,run):
        self.run=Path(run).resolve();self.path=self.run/'checkpoints';self.path.mkdir(parents=True,exist_ok=True)
        self.index=self.run/'references.json'
        if not self.index.exists():atomic_json(self.index,dict(roles={},latest=[],window=[],pins=[]))

    def read(self):return json.loads(self.index.read_text())
    def role(self,name):
        value=self.read()['roles'].get(name)
        return self.path/value if value else None
    def publish(self,name,models,state,kind='full',parents=None):
        save_checkpoint(self.path/name,models,state,kind,parents);return name

    def update(self,role=None,name=None,latest=False,window=False):
        with (self.run/'.references.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            data=self.read()
            if name:inspect_checkpoint(self.path/name)
            if role:data['roles'][role]=name
            if latest:data['latest']=(data['latest']+[name])[-2:];data['roles']['latest']=name
            if window:data['window']=(data['window']+[name])[-5:]
            atomic_json(self.index,data)
            return data

    def pin(self,name):
        with (self.run/'.references.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);data=self.read();inspect_checkpoint(self.path/name)
            data['pins']=sorted(set(data['pins']+[name]));atomic_json(self.index,data)

    def collect(self):
        with (self.run/'.references.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);data=self.read()
            keep=set(data['roles'].values())|set(data['latest'])|set(data['window'])|set(data['pins'])
            # Recovery-state selections must remain resolvable after rollback.
            for name in list(keep):
                if not name:continue
                inspect_checkpoint(self.path/name)
                with (self.path/name/'runner.pkl').open('rb') as f:state=pickle.load(f)
                # Older recovery bundles may still reference a retired checkpoint role.
                keep.update(x for x in (state.get('best_full'),state.get('common')) if x)
                selection=state.get('selection',{})
                for row in selection.get('history',[])[-5:]:keep.add(row['checkpoint'])
                if selection.get('best'):keep.add(selection['best']['checkpoint'])
            removed=[]
            for p in self.path.iterdir():
                if p.is_dir() and not p.name.startswith('.') and p.name not in keep:
                    inspect_checkpoint(p);shutil.rmtree(p);removed.append(p.name)
            return removed

def pin_external(path):
    path=Path(path).resolve();Checkpoints(path.parent.parent).pin(path.name)
    return dict(hash=inspect_checkpoint(path)['hash'],name=path.name)
