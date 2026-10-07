"""Replay audited legacy 600-second demand blocks with current frozen policies."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.experiment_env import RevisedMixin
from experiments.core import write_json

original_inject = RevisedMixin.inject
original_simulate = RevisedMixin._simulate


def inject(self, vehicles):
    if self.network != 'monaco':
        raise ValueError('Legacy replay is Monaco only')
    self.legacy_blocks = [[v for v in vehicles if v['legacy_block'] == i] for i in range(6)]
    self.legacy_injections = []


def simulate(self, seconds):
    if self.cur_sec % 600 == 0 and self.cur_sec < 3600:
        block = self.cur_sec // 600
        original_inject(self, self.legacy_blocks[block])
        self.legacy_injections.append(dict(block=block, time=self.cur_sec,
                                           scheduled=len(self.legacy_blocks[block])))
        write_path = self.work.parent / 'legacy_injections.json'
        # The runtime directory's parent is the immutable evaluation attempt.
        write_path.write_text(json.dumps(self.legacy_injections, indent=2) + '\n')
    return original_simulate(self, seconds)


if __name__ == '__main__':
    RevisedMixin.inject = inject
    RevisedMixin._simulate = simulate
    from experiments.cli import main
    main(sys.argv[1:])
