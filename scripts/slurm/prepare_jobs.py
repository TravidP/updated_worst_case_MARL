#!/usr/bin/env python3
"""Create publication task arrays only after explicit stage prerequisites pass."""
import argparse
import json
import sys
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.checkpoint import Checkpoints,inspect_checkpoint
from experiments.core import file_hash
from experiments.storage import atomic_json
from experiments.validation import load_manifest

p=argparse.ArgumentParser(description=__doc__)
for key in ['stage','image','code-release','data-root','run-root','cache-root','validation-grid','validation-monaco','gate','output']:
    p.add_argument('--'+key,required=True)
a=p.parse_args()
if a.stage not in ['parent','wce','continue']:p.error('Invalid stage')
image=Path(a.image).resolve();image_hash=file_hash(image);run_root=Path(a.run_root).resolve();jobs=[]
def parent(path):
    result=json.loads((path/'result.json').read_text())
    if result['status']!='early_stopped':raise ValueError('Parent has not completed patience stopping: '+str(path))
    receipt=json.loads((path/'scheduler.json').read_text())
    if receipt['exit_code']!=0 or not receipt['job_id']:raise ValueError('Missing successful scheduler receipt')
    account=subprocess.check_output(['sacct','-n','-X','-P','-j',receipt['job_id'],
                                     '--format=JobIDRaw,State,ExitCode'],text=True)
    rows=[line.split('|') for line in account.splitlines() if line.strip()]
    if not rows or any(r[1]!='COMPLETED' or r[2]!='0:0' for r in rows):
        raise ValueError('Slurm accounting has not confirmed successful completion: '+receipt['job_id'])
    cp=Checkpoints(path).role('best');inspect_checkpoint(cp);return str(cp)

for network in ['grid','monaco']:
    validation=str(Path(getattr(a,'validation_'+network)).resolve());load_manifest(validation,network,True)
    for family in ['ma2c','dql','ppo']:
        for seed in [101,202,303]:
            base=run_root/network/family/('seed_%d'%seed)
            methods=['baseline','random_group','domain_randomization','fixed_wce','online_wce'] if a.stage=='continue' else ['baseline']
            for method in methods:
                output=base/(method if a.stage=='continue' else a.stage)
                argv=[a.stage,'--network',network,'--controller',family,'--seed',str(seed),
                      '--validation',validation,'--gate',str(Path(a.gate).resolve()),'--output',str(output)]
                if a.stage!='parent':argv+=['--parent',parent(base/'parent')]
                if a.stage=='continue':
                    argv+=['--method',method]
                    if method.endswith('wce'):argv+=['--wce',parent(base/'wce')]
                if output.exists():
                    result=json.loads((output/'result.json').read_text()) if (output/'result.json').exists() else {}
                    if result.get('status')=='interrupted':argv+=['--resume']
                    elif result.get('status') in ('early_stopped','max_budget_reached','no_learning'):continue
                    else:raise ValueError('Existing failed/active run needs inspection: '+str(output))
                jobs.append(dict(image=str(image),image_sha256=image_hash,code_release=str(Path(a.code_release).resolve()),
                                 data_root=str(Path(a.data_root).resolve()),run_root=str(run_root),cache_root=str(Path(a.cache_root).resolve()),argv=argv))
if Path(a.output).exists():raise FileExistsError(a.output)
atomic_json(a.output,jobs)
print('Prepared %d tasks. No jobs submitted.'%len(jobs))
if jobs:print('Submit with --array=0-%d%%4'%(len(jobs)-1))
