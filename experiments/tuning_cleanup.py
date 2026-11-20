"""Retire development checkpoint directories after verifying the exported winners."""
import fcntl
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

from experiments.campaign import load_campaign, tuning_winners
from experiments.core import file_hash
from experiments.storage import atomic_json


def prune_tuning(directory, selection):
    campaign_dir = Path(directory).resolve()
    campaign = load_campaign(campaign_dir)
    if campaign.get('purpose') != 'development':
        raise ValueError('Only development candidates can be retired')
    export = Path(selection).resolve()
    manifest = export / 'selection.json'
    selected = json.loads(manifest.read_text())
    if selected['campaign_hash'] != campaign['hash']:
        raise ValueError('Selection belongs to a different campaign')
    root = Path(campaign['run_root']).resolve()
    journal = campaign_dir / 'tuning_cleanup.json'
    with ExitStack() as locks:
        def lock(path):
            handle = locks.enter_context(path.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

        lock(root / '.local_scheduler.lock')
        state = json.loads((campaign_dir / 'state.json').read_text())
        if (state['campaign_hash'] != campaign['hash'] or
                any(state['tasks'][t['id']]['status'] != 'complete' for t in campaign['tasks'])):
            raise ValueError('All candidates must be complete in the queue')
        runs = []
        for t in campaign['tasks']:
            run = root / 'development' / t['network'] / t['controller'] / ('candidate_%d' % t['candidate_rank'])
            if run.resolve() != run or run != Path(t['output']) or not run.is_dir():
                raise ValueError('Unexpected candidate directory')
            lock(run / '.run.lock')
            lock(run / '.references.lock')
            if json.loads((run / 'references.json').read_text())['pins']:
                raise ValueError('Candidate contains pinned checkpoints: ' + str(run))
            for target in (run / 'checkpoints', run / '.checkpoints-retiring'):
                if target.is_symlink() or any(path.is_relative_to(target) for path in (export, campaign_dir)):
                    raise ValueError('Unsafe checkpoint or selection directory')
            for name in ('episodes.jsonl.gz', 'blocks.jsonl.gz', 'validation.jsonl.gz'):
                if not (run / name).stat().st_size:
                    raise ValueError('Missing training history: ' + str(run / name))
            runs.append(run)
        for row in selected['selected']:
            filename = row['config']
            if Path(filename).name != filename or file_hash(export / filename) != row['sha256']:
                raise ValueError('Winner INI mismatch')
        identity = dict(campaign_hash=campaign['hash'], selection_hash=file_hash(manifest))
        if journal.exists():
            record = json.loads(journal.read_text())
            if record['identity'] != identity:
                raise ValueError('Cleanup selection changed')
            if record['status'] == 'complete':
                return record
        else:
            if any((run / '.checkpoints-retiring').exists() for run in runs):
                raise ValueError('Unexpected previous cleanup directory')
            if tuning_winners(campaign) != selected['selected']:
                raise ValueError('Exported winners differ from verified validation scores')
            size = sum(f.stat().st_size for run in runs for f in (run / 'checkpoints').rglob('*') if f.is_file())
            record = dict(identity=identity, status='retiring', candidates=len(runs), checkpoint_bytes=size)
            atomic_json(journal, record)
        for run in runs:
            live = run / 'checkpoints'
            retiring = run / '.checkpoints-retiring'
            if live.exists():
                live.rename(retiring)
            if retiring.exists():
                shutil.rmtree(retiring)
        record['status'] = 'complete'
        atomic_json(journal, record)
        return record
