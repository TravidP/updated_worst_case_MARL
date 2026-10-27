#!/usr/bin/env python3
"""Evaluate an explicitly selected endpoint; final tests never update selection."""
import argparse
import configparser
import json
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.checkpoint import Checkpoints,inspect_checkpoint
from experiments.core import digest,file_hash
from experiments.storage import atomic_json,append_record
from experiments.validation import load_manifest,evaluate
from experiments.protocol import root
from experiments.runtime import identity as runtime_identity
from experiments.runner import fingerprint

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--selection',choices=['best','best_at_common_budget'],default='best')
    p.add_argument('--pilot',action='store_true');p.add_argument('--rollouts',type=int,default=10)
    a=p.parse_args();run=Path(a.run).resolve();out=Path(a.output).resolve()
    origin=json.loads((run/'manifest.json').read_text());result=json.loads((run/'result.json').read_text())
    if bool(origin['pilot'])!=a.pilot:raise ValueError('Pilot/publication mismatch')
    if result['status'] in ('failed','interrupted'):raise ValueError('Run has not reached a terminal selection')
    if a.rollouts<1 or (not a.pilot and a.rollouts!=10):raise ValueError('Publication requires ten realizations')
    manifest,mhash=load_manifest(a.manifest,origin['network'],not a.pilot,split='test')
    if not a.pilot:
        from experiments.runtime import verify_runtime
        verify_runtime()
    store=Checkpoints(run);cp=store.role('best_inference' if a.selection=='best' else a.selection)
    if not cp:raise ValueError('Requested selection is unavailable')
    signature=inspect_checkpoint(cp)['signatures']['controller'];store.pin(cp.name)
    out.mkdir(parents=True,exist_ok=False)
    identity=dict(network=origin['network'],controller=origin['controller'],seed=origin['seed'],method=origin['method'],
                  selection=a.selection,checkpoint_hash=inspect_checkpoint(cp)['hash'],validation_hash=origin['validation_hash'],
                  test_hash=mhash,pilot=a.pilot,rollouts=a.rollouts,training_status=result['status'],
                  source=fingerprint(),runtime=runtime_identity())
    atomic_json(out/'manifest.json',identity)
    try:
        with tempfile.TemporaryDirectory(prefix='cbwce-final-') as temp:
            cfg=configparser.ConfigParser();cfg.read_dict(signature['config'])
            cfg['ENV_CONFIG']['data_path']=str(root('assets')/origin['network'])
            path=Path(temp)/'controller.ini'
            with path.open('w') as f:cfg.write(f)
            request=dict(stage='evaluate',network=origin['network'],controller=origin['controller'],seed=origin['seed'],
                         checkpoint=str(cp),config=str(path),manifest=manifest,manifest_hash=mhash,repeats=a.rollouts,detail=True)
            # Persistent result cache is keyed by the exact immutable endpoint and test inputs.
            key=digest(dict(checkpoint=identity['checkpoint_hash'],test=mhash,rollouts=a.rollouts,
                            source=identity['source'],runtime=identity['runtime']))
            cache=root('cache')/'final_evaluations';cache.mkdir(parents=True,exist_ok=True);cached=cache/(key+'.json')
            if cached.exists():
                saved=json.loads(cached.read_text())
                if digest(saved['result'])!=saved['hash']:raise ValueError('Corrupt evaluation cache')
                evaluation=saved['result']
            else:
                evaluation=evaluate(request);atomic_json(cached,dict(hash=digest(evaluation),result=evaluation))
            for row in evaluation['rollouts']:append_record(out/'rollouts.jsonl.gz',row)
            audit=out/'audit';audit.mkdir()
            for scene in manifest['scenarios']:
                if scene['id'] not in ('seen_Uniform','Uniform','peak_1.5'):continue
                sub=dict(request,manifest=dict(manifest,scenarios=[scene]),repeats=1,audit=True,audit_output=str(audit))
                checked=evaluate(sub)
                expected=next(r for r in evaluation['rollouts'] if r['scenario']==scene['id'] and r['realization']==0)
                if checked['rollouts'][0]['mean_queue']!=expected['mean_queue']:raise ValueError('Audit mode changed evaluation')
            atomic_json(out/'result.json',dict(status='complete',score=evaluation['score'],rollouts=len(evaluation['rollouts']),
                        rollouts_sha256=file_hash(out/'rollouts.jsonl.gz'),cache_key=key))
    except BaseException as exc:
        atomic_json(out/'result.json',dict(status='failed',error=repr(exc)));raise

if __name__=='__main__':main()
