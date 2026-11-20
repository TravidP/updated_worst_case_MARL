"""Restartable final evaluation with one immutable compressed shard per rollout."""
import configparser
import fcntl
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from experiments.checkpoint import Checkpoints, inspect_checkpoint
from experiments.core import digest, file_hash
from experiments.metrics import METRICS_SCHEMA
from experiments.protocol import root
from experiments.runner import fingerprint
from experiments.runtime import identity as runtime_identity, verify_runtime
from experiments.storage import atomic_json, atomic_gzip_json, read_gzip_json
from experiments.validation import load_manifest, evaluate


def checked_row(path, identity, scenario, index):
    saved = read_gzip_json(path)
    row = saved['row']
    if (saved.get('identity') != identity or saved.get('hash') != digest(row) or
            row.get('scenario') != scenario or row.get('realization') != index or
            row.get('metrics_schema') != METRICS_SCHEMA or row.get('sample_count') != 3600 or
            row.get('horizon') != 3600):
        raise ValueError('Corrupt/incompatible evaluation shard: '+str(path))
    if not np.isfinite(row['mean_queue']) or row['mean_queue'] < 0:
        raise ValueError('Invalid evaluation score')
    return row


def evaluate_run(run, manifest_path, output, selection='best', pilot=False, rollouts=10,
                 scenarios=None, resume=False):
    if selection not in ('best','final'):raise ValueError('Unknown evaluation selection')
    run=Path(run).resolve();out=Path(output).resolve()
    origin=json.loads((run/'manifest.json').read_text())
    result=json.loads((run/'result.json').read_text())
    if bool(origin['pilot']) != pilot:raise ValueError('Pilot/publication mismatch')
    if result['status'] not in ('early_stopped','no_learning','max_budget_reached'):
        raise ValueError('Run has not reached a terminal selection')
    if rollouts<1 or (not pilot and rollouts!=10):raise ValueError('Publication requires ten realizations')
    manifest,mhash=load_manifest(manifest_path,origin['network'],not pilot,split='test')
    actual=runtime_identity()
    if not pilot:
        verify_runtime()
        if origin['source']!=fingerprint() or origin.get('runtime')!=actual:
            raise ValueError('Evaluation source/runtime differs from the selected training run')
    chosen=manifest['scenarios']
    if scenarios:
        if len(set(scenarios))!=len(scenarios) or not set(scenarios)<=set(s['id'] for s in chosen):
            raise ValueError('Unknown/duplicate requested scenario')
        chosen=[s for s in chosen if s['id'] in scenarios]
    store=Checkpoints(run)
    cp=store.role(selection+'_inference')
    if not cp:raise ValueError('Requested selection is unavailable')
    checkpoint=inspect_checkpoint(cp);signature=checkpoint['signatures']['controller']
    store.pin(cp.name)
    identity=dict(network=origin['network'],controller=origin['controller'],seed=origin['seed'],
        method=origin['method'],selection=selection,checkpoint_hash=checkpoint['hash'],
        validation_hash=origin['validation_hash'],test_hash=mhash,pilot=pilot,rollouts=rollouts,
        best_episode=result.get('best_episode'),selected_episode=result.get('best_episode') if selection=='best' else result['completed_episodes'],completed_episodes=result.get('completed_episodes'),
        training_status=result['status'],source=fingerprint(),runtime=actual,
        metrics_schema=METRICS_SCHEMA,scenarios=[s['id'] for s in chosen],
        scenario_families={s['id']:s['family'] for s in manifest['scenarios']})
    if out.exists() and not resume:raise FileExistsError(out)
    if resume and not (out/'manifest.json').exists():raise ValueError('No evaluation manifest to resume')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text())!=identity:raise ValueError('Evaluation resume identity mismatch')
        else:atomic_json(out/'manifest.json',identity)
        try:
            rows=[]
            with tempfile.TemporaryDirectory(prefix='cbwce-final-',dir=os.environ.get('TMPDIR')) as temp:
                cfg=configparser.ConfigParser();cfg.read_dict(signature['config'])
                cfg['ENV_CONFIG']['data_path']=str(root('assets')/origin['network'])
                path=Path(temp)/'controller.ini'
                with path.open('w') as stream:cfg.write(stream)
                request=dict(stage='evaluate',network=origin['network'],controller=origin['controller'],seed=origin['seed'],
                    checkpoint=str(cp),config=str(path),manifest=manifest,manifest_hash=mhash,repeats=rollouts,
                    detail=True,tripinfo=True)
                for scene in chosen:
                    row_identity=digest(dict(checkpoint=checkpoint['hash'],test=mhash,scene=scene['id'],
                        source=identity['source'],runtime=actual,metrics=METRICS_SCHEMA))
                    cache=root('cache')/'final_evaluations'/row_identity
                    cache.mkdir(parents=True,exist_ok=True)
                    local=out/'rows';local.mkdir(exist_ok=True)
                    with (cache/'.lock').open('a') as cache_lock:
                        fcntl.flock(cache_lock,fcntl.LOCK_EX)
                        missing=[]
                        for i in range(rollouts):
                            name=scene['id']+'_%02d.json.gz'%i
                            if (local/name).exists():checked_row(local/name,row_identity,scene['id'],i)
                            elif (cache/name).exists():
                                row=checked_row(cache/name,row_identity,scene['id'],i)
                                atomic_gzip_json(local/name,dict(identity=row_identity,row=row,hash=digest(row)))
                            else:missing.append(i)
                        if missing:
                            sub=dict(request,manifest=dict(manifest,scenarios=[scene]),indices=missing,
                                     rows_directory=str(local),row_identity=row_identity)
                            outcome=evaluate(sub)
                            if outcome and outcome.get("interrupted"):
                                result=dict(status="interrupted",rollouts=len(list(local.glob("*.json.gz"))))
                                atomic_json(out/"result.json",result)
                                return result
                        scene_rows=[]
                        for i in range(rollouts):
                            name=scene['id']+'_%02d.json.gz'%i
                            row=checked_row(local/name,row_identity,scene['id'],i)
                            if not (cache/name).exists():
                                atomic_gzip_json(cache/name,dict(identity=row_identity,row=row,hash=digest(row)))
                            scene_rows.append(row)
                        rows.extend(scene_rows)
                    if scene['id'] in ('seen_Uniform','peak_1.5'):
                        audit=out/'audit'/scene['id'];audit.mkdir(parents=True,exist_ok=True)
                        certificate=audit/'result.json'
                        expected=digest({k:v for k,v in scene_rows[0].items() if k!='wall_seconds'})
                        if certificate.exists():
                            saved=json.loads(certificate.read_text())
                            if saved['identity']!=row_identity or saved['metrics_hash']!=expected:
                                raise ValueError('Audit certificate mismatch')
                            for filename,h in saved['files'].items():
                                if file_hash(audit/filename)!=h:raise ValueError('Audit artifact damaged')
                        else:
                            sub=dict(request,manifest=dict(manifest,scenarios=[scene]),indices=[0],
                                     audit=True,audit_output=str(audit))
                            checked=evaluate(sub)['rollouts'][0]
                            if digest({k:v for k,v in checked.items() if k!='wall_seconds'})!=expected:
                                raise ValueError('Audit mode changed evaluation metrics')
                            atomic_json(certificate,dict(identity=row_identity,metrics_hash=expected,
                                files={p.name:file_hash(p) for p in audit.glob('*.npz')}))
            # Rows are the canonical data; the index contains hashes, not duplicated traces.
            index={p.name:file_hash(p) for p in sorted((out/'rows').glob('*.json.gz'))}
            atomic_json(out/'result.json',dict(status='complete',score=float(np.mean([r['mean_queue'] for r in rows])),
                        rollouts=len(rows),shards=index,metrics_schema=METRICS_SCHEMA))
            return dict(status='complete',rollouts=len(rows),output=str(out))
        except BaseException as exc:
            atomic_json(out/'result.json',dict(status='failed',error=repr(exc)));raise


def read_evaluation(path):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    result=json.loads((path/'result.json').read_text())
    if result['status']!='complete' or manifest.get('metrics_schema')!=METRICS_SCHEMA:
        raise ValueError('Only complete current-schema evaluations may enter a report')
    if manifest.get('selection') not in ('best','final'):raise ValueError('Unknown evaluation selection')
    rows=[]
    for name,h in result['shards'].items():
        target=path/'rows'/name
        if target.resolve().parent!=(path/'rows').resolve() or file_hash(target)!=h:
            raise ValueError('Damaged evaluation shard')
        saved=read_gzip_json(target)
        if saved['hash']!=digest(saved['row']):raise ValueError('Damaged rollout payload')
        row=saved['row']
        identity=digest(dict(checkpoint=manifest['checkpoint_hash'],test=manifest['test_hash'],scene=row['scenario'],
            source=manifest['source'],runtime=manifest['runtime'],metrics=METRICS_SCHEMA))
        rows.append(checked_row(target,identity,row['scenario'],row['realization']))
    expected={(s,i) for s in manifest['scenarios'] for i in range(manifest['rollouts'])}
    observed={(r['scenario'],r['realization']) for r in rows}
    if observed!=expected or len(rows)!=len(expected) or result['rollouts']!=len(rows):
        raise ValueError('Incomplete/duplicate scenario shards')
    return manifest,rows
