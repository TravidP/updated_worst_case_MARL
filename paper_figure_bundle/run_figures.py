"""Run the archived plotting code with local paths, preserving the source copies."""
from pathlib import Path
import argparse
import importlib.util
import os

ROOT = Path(__file__).resolve().parent
SCRIPTS = ['heatmap_optimized_unseen.py', 'boxplot_comparison_2in1.py', 'Worst_case_Multi_plot.py']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check dependencies without drawing')
    args = parser.parse_args()
    missing = [name for name in ['numpy', 'pandas', 'matplotlib', 'seaborn', 'openpyxl']
               if importlib.util.find_spec(name) is None]
    if missing:
        print('Missing dependencies: ' + ', '.join(missing))
        print('Install with: python -m pip install -r requirements.txt')
        raise SystemExit(1)
    if args.check:
        print('All plotting dependencies available.')
        return
    os.chdir(ROOT)
    cache = ROOT / 'generated' / 'matplotlib_cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    for folder in ['manual_comparisons', 'manual_comparisons_real', 'ieee_comparisons_stacked']:
        (ROOT / 'runs_eval' / folder).mkdir(parents=True, exist_ok=True)
    for script in SCRIPTS:
        path = ROOT / script
        source = path.read_text(encoding='utf-8')
        # The original scripts use a Linux-only /tmp cache path. Adapt it in
        # memory only; archived scripts and paper images remain byte-identical.
        source = source.replace('Path("/tmp/matplotlib")', 'Path(' + repr(str(cache)) + ')')
        print('Running ' + script, flush=True)
        exec(compile(source, str(path), 'exec'), {'__file__': str(path), '__name__': '__main__'})


if __name__ == '__main__':
    main()
