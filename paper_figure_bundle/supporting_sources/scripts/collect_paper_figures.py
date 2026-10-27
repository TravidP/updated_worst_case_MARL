"""Collect paper figures and their available provenance without changing sources."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper_figure_bundle'
OUT.mkdir(exist_ok=True)
records = []


def copy(src, dest=None, role='source'):
    src = ROOT / src
    dst = OUT / (dest or src.relative_to(ROOT))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    assert digest == hashlib.sha256(dst.read_bytes()).hexdigest()
    records.append(dict(source_path=src.relative_to(ROOT).as_posix(),
                        bundle_path=dst.relative_to(OUT).as_posix(), role=role,
                        bytes=src.stat().st_size, sha256=digest))


for p in sorted((ROOT / 'paper').iterdir()):
    if p.is_file():
        copy(p.relative_to(ROOT), role='paper_image' if p.suffix == '.png' else 'paper_document')

for name in ['Worst_case_Multi_plot.py', 'heatmap_optimized_unseen.py', 'boxplot_comparison_2in1.py']:
    copy(name, role='direct_plot_program')

for scenario in ['signal_controller_benchmark', 'signal_controller_benchmark_real']:
    d = ROOT / 'runs_eval' / scenario
    for p in sorted(d.iterdir()):
        if p.suffix in {'.csv', '.xlsx'}:
            copy(p.relative_to(ROOT), role='summary_or_manifest')
    for algo in ['ia2c', 'ma2c', 'iqll', 'ppo']:
        for p in sorted((d / algo).glob('*_raw.csv')):
            copy(p.relative_to(ROOT), role='raw_rollout_data')

for directory in ['data_traffic', 'real_net_subnet/demand_groups', 'real_net_subnet/data', 'large_grid/data']:
    for p in sorted((ROOT / directory).rglob('*')):
        if p.is_file() and p.suffix in {'.csv', '.json', '.xml', '.sumocfg', '.netccfg', '.ini', '.pdf', '.png', '.txt', '.py'}:
            copy(p.relative_to(ROOT), 'supporting_sources/' + p.relative_to(ROOT).as_posix(), 'related_network_or_demand')

for name in ['eval_signal_controllers.py', 'eval_signal_controllers_real.py',
             'compute_horizon_averages_from_raw.py', 'generate_traffic_flow.py',
             'generate_traffic_OD.py', 'generate_traffic.py', 'environment.yml', 'LICENSE']:
    copy(name, 'supporting_sources/' + name, 'upstream_reference')
copy('scripts/collect_paper_figures.py', 'supporting_sources/scripts/collect_paper_figures.py', 'collection_program')

# Check the exact CSV selection used by the stacked plots, without importing plotting packages.
tree = ast.parse((ROOT / 'Worst_case_Multi_plot.py').read_text(encoding='utf-8'))
configs = next(ast.literal_eval(n.value) for n in tree.body
               if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PLOT_CONFIGS' for t in n.targets))
selections = []
for cfg in configs:
    for pair in cfg['data_pairs']:
        for variant in ['baseline', 'retrained']:
            ref = pair[variant + '_csv']
            matches = sorted((OUT / 'runs_eval').rglob(Path(ref).name))
            assert len(matches) == 1, (ref, matches)
            selections.append(dict(scenario='Monaco' if '_real/' in ref else '5x5',
                                   metric=cfg['plot_title'], algorithm=pair['algo_name'], variant=variant,
                                   paper_group=int(re.search(r'group(\d+)', ref).group(1)) + 1,
                                   csv_path=matches[0].relative_to(OUT).as_posix()))

with (OUT / 'stacked_plot_inputs.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(selections[0])); w.writeheader(); w.writerows(selections)

# XLSX read-only extraction using standard ZIP/XML, preserving all originals.
ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
tables = {}
for p in (OUT / 'runs_eval').rglob('*.xlsx'):
    with zipfile.ZipFile(p) as z:
        strings = [''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rows = []
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:row', ns):
            vals = [''] * 25
            for cell in row:
                letters = re.match(r'[A-Z]+', cell.get('r')).group()
                col = 0
                for c in letters: col = col * 26 + ord(c) - 64
                v = cell.find('s:v', ns)
                val = v.text if v is not None else ''.join(cell.itertext())
                if cell.get('t') == 's': val = strings[int(val)]
                vals[col - 1] = val
            rows.append(vals)
        assert len(rows) == 15 and [r[0] for r in rows[2:14]] == [str(i) for i in range(12)]
        tables[p.parent.name] = rows

# Inspect all rollout CSVs and compare their averages with the plot-input workbooks.
checks, comparisons = [], []
for rec in records:
    if rec['role'] != 'raw_rollout_data': continue
    p = OUT / rec['bundle_path']
    with p.open(newline='') as f:
        rd = csv.reader(f); header = next(rd)
        assert header[0] == 't' and all(c.startswith('rollout_') for c in header[1:])
        sums = 0.0; count = 0; times = []
        for row in rd:
            assert len(row) == len(header)
            times.append(float(row[0])); sums += sum(map(float, row[1:])) / (len(header) - 1); count += 1
        assert times == list(range(1, 3601)), p
    checks.append(dict(path=rec['bundle_path'], time_rows=count, rollouts=len(header)-1, horizon_average=sums/count))
    m = re.match(r'(ia2c|ma2c|iqll|ppo)_(marl|real|retrained)_group(\d+)_.*_(queue|speed)_raw.csv', p.name)
    assert m, p
    alg, variant, group, metric = m.groups()
    col = 1 + ['ia2c','ma2c','iqll','ppo'].index(alg)*6 + (3 if metric == 'speed' else 0) + (1 if variant == 'retrained' else 0)
    scenario = p.parents[1].name
    workbook_val = float(tables[scenario][int(group)+2][col])
    comparisons.append(dict(path=rec['bundle_path'], raw_average=sums/count,
                            workbook_value=workbook_val, difference=sums/count-workbook_val))

for filename, data in [('raw_data_checks.csv', checks), ('raw_vs_workbook.csv', comparisons), ('file_manifest.csv', records)]:
    with (OUT / filename).open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

text = (ROOT / 'paper/main.tex').read_text(encoding='utf-8')
references=[]
for line_no, line in enumerate(text.splitlines(), 1):
    # Strip LaTeX comments, including inline comments.
    line = re.split(r'(?<!\\)%', line)[0]
    for ref in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', line):
        name=Path(ref).name
        p=OUT/'paper'/name
        substitute=OUT/'paper'/(Path(name).stem+'.png')
        references.append(dict(tex_line=line_no, latex_reference=ref,
                               available_file=p.relative_to(OUT).as_posix() if p.exists() else substitute.relative_to(OUT).as_posix() if substitute.exists() else '',
                               status='available_by_basename' if p.exists() else 'png_available_pdf_missing' if substitute.exists() else 'missing'))
with (OUT/'latex_image_references.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(references[0]));w.writeheader();w.writerows(references)

report=dict(source_file_count=len(records), raw_csv_count=len(checks), total_source_bytes=sum(r['bytes'] for r in records),
            raw_shapes=sorted({(r['time_rows'],r['rollouts']) for r in checks}),
            mismatches_over_0_011=[r for r in comparisons if abs(r['difference'])>0.011],
            missing_latex_references=[r for r in references if r['status']=='missing'])
(OUT/'collection_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
