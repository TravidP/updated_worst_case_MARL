"""Task labels and persistent operator permissions shared by the scheduler and dashboard."""
import json
import os
from pathlib import Path

STAGES = ('inventory', 'tuning', 'freeze', 'verify', 'parent', 'wce', 'baseline',
          'random_group', 'domain_randomization', 'fixed_wce', 'online_wce', 'evaluate', 'report')
TRAIN_STAGES = STAGES[4:11]


def stage(task):
    if 'candidate_rank' in task:
        return 'tuning'
    if task['kind'] == 'evaluate':
        return 'evaluate'
    return task['method'] if task.get('stage') == 'continue' else task.get('stage', 'parent')


def task_number(task):
    s = stage(task)
    if s not in TRAIN_STAGES or task.get('network') not in ('grid', 'monaco'):
        return ''
    lane = ('grid', 'monaco').index(task['network']) * 3 + ('ma2c', 'ppo', 'dql').index(task['controller'])
    return f'{TRAIN_STAGES.index(s) * 6 + lane + 1:02d}'


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_bytes())
    except FileNotFoundError:
        return default


def permitted(control, group, task_id=''):
    if control.get('paused') or task_id in control.get('held', []):
        return False
    return control.get('auto', False) or group in control.get('stages', []) or task_id in control.get('selected', [])


def pause_requested():
    path = os.environ.get('CBWCE_PAUSE_FILE')
    return bool(path and Path(path).is_file())
