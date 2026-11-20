"""Local CPU and memory budgets, accounting for process affinity and Linux cgroup v2."""
import math
import os
from pathlib import Path

GIB = 1024 ** 3
TUNING_SLOTS = 12
FORMAL_SLOTS = 24


def workflow_reservation(observed, supplied):
    """Reserve one single-thread worker slot and track memory/disk growth while it runs."""
    profiles = [profile for profile in (observed, supplied) if profile]
    rss = 2 * GIB
    disk = GIB
    for profile in profiles:
        peak, headroom = profile['peak_rss_bytes'], profile['checkpoint_headroom_bytes']
        if not math.isfinite(peak) or peak < 0 or not math.isfinite(headroom) or headroom <= 0:
            raise ValueError('Invalid workflow memory or checkpoint estimate')
        rss = max(rss, peak)
        disk = max(disk, headroom)
    return dict(cpu_cores=1, peak_rss_bytes=rss, checkpoint_headroom_bytes=disk)


def cgroup_dirs():
    mount = Path('/sys/fs/cgroup')
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            group = Path(line[3:])
            for item in (group, *group.parents):
                if '..' not in item.parts:
                    yield mount / str(item).lstrip('/')
            break


def cpu_capacity():
    allowed = os.sched_getaffinity(0)
    cores = set()
    for cpu in allowed:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            cores.add(tuple((topology / key).read_text().strip() for key in ('physical_package_id', 'core_id')))
        except OSError:
            cores.add(('cpu', str(cpu)))
    capacity = float(len(cores))
    for group in cgroup_dirs():
        limit = group / 'cpu.max'
        if limit.is_file():
            quota, period = limit.read_text().split()
            if quota != 'max':
                capacity = min(capacity, int(quota) / int(period))
    return dict(physical=len(cores), logical=len(allowed), effective=capacity,
                default_budget=min(capacity, max(1, math.floor(capacity * .8))))


def memory_info():
    fields = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemTotal:', 'MemAvailable:'))}
    total, available = fields['MemTotal'], fields['MemAvailable']
    for group in cgroup_dirs():
        limit = group / 'memory.max'
        if limit.exists() and (value := limit.read_text().strip()) != 'max':
            total = min(total, int(value))
            available = min(available, max(0, int(value) - int((group / 'memory.current').read_text())))
    return total, available


def admission(active, job, ram, cpu, available, disk, reserve):
    """Return (display state, reason). Unknown resource classes require an exclusive slot."""
    if job is None:
        if active:
            return 'waiting_resources', 'Exclusive resource measurement needs an empty pool'
        if available < GIB or disk <= reserve:
            return 'waiting_resources', 'Insufficient RAM or disk for exclusive measurement'
        return 'ready', ''
    memory = job['peak_rss_bytes'] * 1.3
    if job['cpu_cores'] > cpu or memory > ram:
        return 'needs_attention', f"Job needs {job['cpu_cores']:.1f} cores and {memory/GIB:.1f} GiB; pool has {cpu:.1f} cores and {ram/GIB:.1f} GiB"
    if None in active:
        return 'waiting_resources', 'Exclusive resource measurement in progress'
    remaining = cpu - sum(entry['cpu_cores'] for entry in active)
    if job['cpu_cores'] > remaining:
        return 'waiting_resources', f"Needs {job['cpu_cores']:.1f} cores; {remaining:.1f} free"
    free_ram = min(available, ram - sum(entry['peak_rss_bytes'] * 1.3 for entry in active))
    if memory > free_ram:
        return 'waiting_resources', f'Needs {memory/GIB:.1f} GiB RAM; {free_ram/GIB:.1f} GiB free'
    disk_needed = reserve + sum(entry['checkpoint_headroom_bytes'] for entry in [*active, job])
    if disk < disk_needed:
        return 'waiting_resources', f'Needs {disk_needed/GIB:.1f} GiB disk including reserve; {disk/GIB:.1f} GiB free'
    return 'ready', ''


def observe(task, row, estimates, key, progress):
    """Learn per-config envelopes in place. Inspect a new recovery directory once."""
    from experiments.dispatch import read_json
    entry = estimates.setdefault(key, dict(peak_rss_bytes=0, cpu_cores=1.,
        checkpoint_headroom_bytes=GIB, coverage=[]))
    entry['peak_rss_bytes'] = max(entry['peak_rss_bytes'], row.get('peak_rss_bytes', 0))
    entry['cpu_cores'] = max(entry['cpu_cores'], row.get('cpu_cores', 0))
    coverage = set(entry['coverage'])
    output = Path(task['output'])
    if task['kind'] == 'evaluate':
        if row['status'] == 'complete':
            coverage.add('evaluation')
    else:
        if progress.get('phase') in ('training', 'validation') and row.get('rss_bytes'):
            coverage.add(progress['phase'])
        name = read_json(output/'references.json', {}).get('roles', {}).get('latest')
        if name and row.get('measured_checkpoint') != name:
            cp = output/'checkpoints'/name
            try:
                manifest = read_json(cp/'manifest.json')
                if not manifest:
                    return
                size = sum((cp/f).stat().st_size for f in manifest['files'])
                replay = read_json(cp/'controller/replay/index.json')
            except FileNotFoundError:
                return  # Rotated during inspection; retry on the next sample.
            if manifest['kind'] == 'full':
                coverage.add('checkpoint')
                entry['checkpoint_headroom_bytes'] = max(entry['checkpoint_headroom_bytes'], size * 2)
                if replay and replay['size'] == replay['capacity']:
                    coverage.add('replay_full')
                row['measured_checkpoint'] = name
    entry['coverage'] = sorted(coverage)
