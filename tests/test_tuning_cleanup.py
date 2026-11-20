import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.campaign import load_campaign, prepare_tuning, select_tuning
from experiments.checkpoint import Checkpoints, inspect_checkpoint
from experiments.storage import append_record, atomic_json
from experiments.tuning_cleanup import prune_tuning
from experiments.validation import validation_manifest


class TuningRetentionTests(unittest.TestCase):
    def test_validate_all_exports_before_retiring_then_recover_interrupted_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            queue, export = root/'queue', root/'selected'
            validation = {}
            for network in ('grid','monaco'):
                validation[network] = root/(network+'.json')
                atomic_json(validation[network], validation_manifest(network, 9001))
            prepare_tuning(queue, validation, root/'runs')
            campaign = load_campaign(queue)
            state = json.loads((queue/'state.json').read_text())
            protocol = copy.deepcopy(campaign['protocol'])
            protocol['max_episodes']['parent'] = 1000
            runs = []
            for task in campaign['tasks']:
                run = Path(task['output'])
                runs.append(run)
                store = Checkpoints(run)
                store.publish('best', {}, {}, 'full')
                store.update('best','best')
                store.publish('inference', {}, {}, 'inference')
                store.update('best_inference','inference')
                h = inspect_checkpoint(store.role('best_inference'))['hash']
                append_record(run/'validation.jsonl.gz',dict(score=10-task['candidate_rank'],checkpoint_hash=h))
                for name in ('episodes','blocks'):
                    append_record(run/(name+'.jsonl.gz'),dict(episode=1000))
                atomic_json(run/'result.json',dict(status='max_budget_reached',completed_episodes=1000))
                identity = {k:task[k] for k in ('network','controller','seed','method','stage','config_hash')}
                atomic_json(run/'manifest.json',dict(identity, pilot=True, protocol=protocol,
                    validation_hash=campaign['validation'][task['network']]['hash'],source=campaign['source']))
                atomic_json(run/'local_receipt.json',dict(exit_code=0,source=campaign['source']))
                state['tasks'][task['id']]['status'] = 'complete'
            atomic_json(queue/'state.json',state)
            select_tuning(queue,export)
            ini = next(export.glob('*.ini'))
            original = ini.read_bytes()
            ini.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'INI mismatch'):
                prune_tuning(queue,export)
            ini.write_bytes(original)
            self.assertTrue(all((r/'checkpoints/best').exists() for r in runs))
            refs = runs[-1]/'references.json'
            data = json.loads(refs.read_text())
            atomic_json(refs, dict(data,pins=['best']))
            with self.assertRaisesRegex(ValueError,'pinned'):
                prune_tuning(queue,export)
            atomic_json(refs,data)
            remove = shutil.rmtree
            def interrupt(path):
                if path.parent == runs[1]:
                    raise InterruptedError
                remove(path)
            with patch('experiments.tuning_cleanup.shutil.rmtree',side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    prune_tuning(queue,export)
            receipt = prune_tuning(queue,export)
            self.assertEqual(receipt,prune_tuning(queue,export))
            self.assertEqual(receipt['candidates'],24)
            self.assertGreater(receipt['checkpoint_bytes'],0)
            for run in runs:
                self.assertFalse((run/'checkpoints').exists())
                self.assertFalse((run/'.checkpoints-retiring').exists())
                self.assertTrue((run/'episodes.jsonl.gz').exists())
            self.assertEqual(len(list(export.glob('*.ini'))),6)
