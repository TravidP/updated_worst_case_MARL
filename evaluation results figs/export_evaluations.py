#!/usr/bin/env python3
"""Reproduce the English evaluation archive from the results-site snapshot."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

CONTROLLERS = ['ia2c', 'ma2c', 'iqll', 'ppo']
METHODS = ['baseline', 'random_group', 'domain_randomization', 'fixed_wce', 'online_wce']
LABELS = ['Baseline', 'Random grouping', 'Domain randomization', 'Fixed WCE', 'Online WCE']
COLORS = ['#244a75', '#d6781f', '#2d876e', '#985da5', '#c3423f']
HEADERS = ['Baseline', 'Random\ngrouping', 'Domain\nrandomization', 'Fixed WCE', 'Online WCE']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.titlesize': 14, 'axes.labelsize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False})


def ema(values):
    out = np.array(values, dtype=float, copy=True)
    for i in range(1, len(out)):
        out[i] = .9 * out[i-1] + .1 * values[i]
    return out


class Exporter:
    def __init__(self, source, output):
        self.source, self.output = source, output
        self.sources, self.checks, self.figures = {}, [], []

    def record(self, path):
        path = Path(path)
        self.sources[str(path.relative_to(self.source))] = hashlib.sha256(path.read_bytes()).hexdigest()

    def copy(self, source, target):
        self.record(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def save(self, fig, stem):
        stem.parent.mkdir(parents=True, exist_ok=True)
        network = next(part for part in stem.relative_to(self.output).parts if part in ['grid','monaco'])
        width, old_height = fig.get_size_inches()
        new_height = old_height + .4
        scale = old_height / new_height
        axes_positions = [(ax, ax.get_position().bounds) for ax in fig.axes]
        legend_positions = [(legend, legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).bounds)
                            for legend in fig.legends]
        fig.set_size_inches(width, new_height)
        for ax, (x,y,w,h) in axes_positions:
            ax.set_position([x,y*scale,w,h*scale])
        for artist in fig.texts:
            x,y = artist.get_position()
            artist.set_position((x,y*scale))
        for legend, (x,y,w,h) in legend_positions:
            legend.set_bbox_to_anchor((x,y*scale,w,h*scale),transform=fig.transFigure)
        fig.text(.5,1-.10/new_height,network.capitalize()+' Network',
                 ha='center',va='top',fontsize=12,weight='bold')
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box = fig.bbox
        text_bounds = []
        for artist in fig.findobj(Text):
            if not artist.get_visible() or not artist.get_text().strip():
                continue
            b = artist.get_window_extent(renderer)
            if b.x0 < -1 or b.y0 < -1 or b.x1 > box.x1+1 or b.y1 > box.y1+1:
                raise ValueError(f'Text outside figure: {stem}: {artist.get_text()}')
            for other, previous in text_bounds:
                width = min(b.x1,previous.x1)-max(b.x0,previous.x0)
                height = min(b.y1,previous.y1)-max(b.y0,previous.y0)
                if width > 1 and height > 1:
                    raise ValueError(f'Overlapping text: {stem}: {artist.get_text()} / {other}')
            text_bounds.append((artist.get_text(),b))
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(stem.with_suffix('.'+extension), dpi=300, facecolor='white')
        with Image.open(stem.with_suffix('.png')) as im:
            expected = tuple(int(round(x*300)) for x in fig.get_size_inches())
            assert all(abs(a-b) <= 1 for a,b in zip(im.size, expected))
        self.figures.append(str(stem.relative_to(self.output)))
        plt.close(fig)

    def table(self, values, directory, title, note, stem='mean_queue'):
        directory.mkdir(parents=True, exist_ok=True)
        winners = values == values.min(axis=1, keepdims=True)
        frame = pd.DataFrame(values, index=[c.upper() for c in CONTROLLERS], columns=LABELS)
        frame.index.name = 'Controller'
        frame.to_csv(directory/(stem+'.csv'), float_format='%.9g')
        md = [f'# {title}', '', 'Mean queue (vehicles; lower is better)', '',
              '| Controller | '+' | '.join(LABELS)+' |', '|---|'+'---:|'*5]
        tex = [r'\begin{tabular}{lrrrrr}', r'\hline', 'Controller & '+' & '.join(LABELS)+r' \\', r'\hline']
        for i,c in enumerate(CONTROLLERS):
            cells = [f'{v:.2f}' for v in values[i]]
            md.append('| '+c.upper()+' | '+' | '.join('**'+v+'**' if winners[i,j] else v for j,v in enumerate(cells))+' |')
            tex.append(c.upper()+' & '+' & '.join(r'\textbf{'+v+'}' if winners[i,j] else v for j,v in enumerate(cells))+r' \\')
        md += ['', note, '', 'Bold: lowest unrounded mean within each controller; exact ties included.']
        tex += [r'\hline', r'\end{tabular}', '% '+title, '% Mean queue (vehicles; lower is better)', '% '+note]
        (directory/(stem+'.md')).write_text('\n'.join(md)+'\n')
        (directory/(stem+'.tex')).write_text('\n'.join(tex)+'\n')
        fig, ax = plt.subplots(figsize=(11, 3.5)); ax.axis('off')
        fig.suptitle(title, y=.95, fontsize=14)
        fig.text(.5,.79,'Mean queue (vehicles; lower is better)',ha='center',fontsize=12)
        tbl = ax.table(cellText=[[c.upper()]+[f'{v:.2f}' for v in values[i]] for i,c in enumerate(CONTROLLERS)],
                       colLabels=['Controller']+HEADERS, cellLoc='center', bbox=[0,.12,1,.77])
        tbl.auto_set_font_size(False); tbl.set_fontsize(11)
        for (r,c),cell in tbl.get_celld().items():
            cell.set_edgecolor('#d4d9df'); cell.set_linewidth(.5)
            cell.set_facecolor('#edf1f5' if r == 0 else ('#f8fafc' if r%2 == 0 else 'white'))
            if r == 0 or c == 0 or (r > 0 and winners[r-1,c-1]): cell.set_text_props(weight='bold')
        fig.subplots_adjust(left=.03,right=.97,top=.78,bottom=.18)
        fig.text(.03,.07,note,fontsize=9,va='bottom',linespacing=1.5)
        self.save(fig,directory/stem)

    def draw_axes(self, ax, curves, title):
        ymax = max(1., max(float(curves[m][3].max()) for m in METHODS) * 1.05)
        for m,label,color in zip(METHODS,LABELS,COLORS):
            t,lo,mean,hi = curves[m]
            ax.fill_between(t,lo,hi,color=color,alpha=.12,linewidth=0)
            ax.plot(t,mean,color=color,lw=1.8,label=label)
        ax.set(xlim=(0,3600),ylim=(0,ymax),xlabel='Simulation time (s)',ylabel='Network queue (vehicles)')
        ax.set_yticks([tick for tick in ax.get_yticks() if 0 <= tick <= ymax])
        ax.set_title(title,pad=12); ax.set_xticks(np.arange(0,3601,600))
        ax.grid(axis='y',alpha=.2,linewidth=.6); ax.set_axisbelow(True)

    def dataset(self, base, network, split):
        catalog = json.loads((base/'catalog.json').read_text())
        self.record(base/'catalog.json')
        summary = pd.read_csv(base/'metrics_summary.csv')
        rollouts = pd.read_csv(base/'rollout_metrics.csv')
        scenarios = [s for s in catalog['scenarios'] if s['split'] == ('external' if split == 'real_world' else split)]
        root = self.output/split/network
        for filename in ['catalog.json','metrics_summary.csv','rollout_metrics.csv','validation.json','paired_statistics.json','queue_comparisons.csv']:
            if (base/filename).exists(): self.copy(base/filename,root/'data'/filename)
        if split == 'real_world':
            caveat = ''
        else: caveat = 'Frozen '+('in-distribution' if split == 'seen' else 'test')+' evaluation.'
        note = 'n = 10 per method; training seed = 101; table statistics are unsmoothed.'
        if caveat: note += '\n'+caveat
        values_by_scenario = []
        for scenario in scenarios:
            sid = scenario['id']; dest = root/sid
            title = network.capitalize()+' | '+scenario['label']['en'].replace(' · ',' / ')
            if split == 'real_world':
                title = 'Hangzhou realworld' if network == 'grid' else 'Monaco real world data'
            group = summary[summary.scenario == sid]
            rg = rollouts[rollouts.scenario == sid]
            assert len(group) == 20 and len(rg) == 200
            assert not group.duplicated(['controller','method']).any()
            values = np.zeros((4,5)); curves = {}
            group.to_csv(dest_mkdir(dest/'data')/'metrics_summary.csv',index=False)
            rg.to_csv(dest/'data'/'rollout_metrics.csv',index=False)
            for i,c in enumerate(CONTROLLERS):
                curves[c] = {}
                for j,m in enumerate(METHODS):
                    stat = group[(group.controller == c)&(group.method == m)].iloc[0]
                    rr = rg[(rg.controller == c)&(rg.method == m)]
                    assert stat['n'] == 10 and len(rr) == 10 and rr.rollout.nunique() == 10
                    assert set(rr.arrival_seed) == set(range(51001,51011))
                    p = base/'series'/c/m/scenario['split']/(sid+'.csv')
                    raw = pd.read_csv(p); self.copy(p,dest/'data'/'series'/c/(m+'.csv'))
                    data = raw[['queue_min','queue_mean','queue_max']].to_numpy()
                    assert len(raw) == 3600 and np.array_equal(raw.time,np.arange(1,3601))
                    assert (raw.n == 10).all() and np.isfinite(data).all() and (data >= 0).all()
                    assert (data[:,0] <= data[:,1]).all() and (data[:,1] <= data[:,2]).all()
                    q = rr.mean_queue.to_numpy(); assert np.isfinite(q).all()
                    for field,expected in [('mean',q.mean()),('min',q.min()),('max',q.max()),('sd',q.std(ddof=1))]:
                        assert np.isclose(stat['mean_queue_'+field],expected,atol=2e-5,rtol=1e-7)
                    assert np.isclose(data[:,1].mean(),q.mean(),atol=2e-5,rtol=1e-7)
                    smooth = np.column_stack([ema(data[:,k]) for k in range(3)])
                    assert (smooth[:,0] <= smooth[:,1]+1e-9).all() and (smooth[:,1] <= smooth[:,2]+1e-9).all()
                    assert np.allclose(smooth[1:],.9*smooth[:-1]+.1*data[1:],atol=1e-12)
                    curves[c][m] = (raw.time.to_numpy(),smooth[:,0],smooth[:,1],smooth[:,2])
                    values[i,j] = stat.mean_queue_mean
            for c in CONTROLLERS:
                fig,ax = plt.subplots(figsize=(9,5.2))
                self.draw_axes(ax,curves[c],c.upper()+' | '+title)
                fig.subplots_adjust(left=.10,right=.98,bottom=.29,top=.86)
                fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.54,.10),ncol=3,frameon=False,fontsize=10)
                caption = 'Solid: mean; shading: min–max across 10 runs; EMA = 0.9; training seed = 101.'
                if caveat: caption += '\n'+caveat
                fig.text(.10,.025,caption,fontsize=8.5)
                self.save(fig,dest/'figures'/(c+'_queue'))
            if split == 'real_world':
                fig,axes = plt.subplots(2,2,figsize=(14,9))
                for c,ax in zip(CONTROLLERS,axes.flat): self.draw_axes(ax,curves[c],c.upper())
                fig.suptitle(title,fontsize=16,y=.98)
                fig.subplots_adjust(left=.07,right=.98,bottom=.19,top=.90,hspace=.40,wspace=.22)
                fig.legend([Line2D([],[],color=x,lw=2) for x in COLORS],LABELS,loc='lower center',bbox_to_anchor=(.5,.08),ncol=5,frameon=False)
                fig.text(.07,.025,caption+'\nEach panel uses its own adaptive y-axis range.',fontsize=10)
                self.save(fig,dest/'figures'/'all_controllers_queue')
            self.table(values,dest/'tables',title,note)
            values_by_scenario.append(values)
            self.checks.append({'network':network,'split':split,'scenario':sid,'groups':20,'rollouts':200,'series_rows':3600,'status':'passed'})
            print(f'Completed {split}/{network}/{sid}',flush=True)
        if split != 'real_world':
            stack = np.stack(values_by_scenario)
            self.table(stack.mean(axis=0),root/'summary',network.capitalize()+' | '+split.capitalize()+' summary',
                       'Equal weight per scenario; '+str(len(scenarios))+' scenarios; n = 10 per scenario/method; training seed = 101.\nUnsmoothed Mean queue; best values compared within controller.')
            if split == 'test':
                for family in ['redistribution','mixture','temporal','peak']:
                    indices = [i for i,s in enumerate(scenarios) if s['family'] == family]
                    self.table(stack[indices].mean(axis=0),root/'summary',network.capitalize()+' | Test | '+family.capitalize(),
                               f'Equal weight per scenario; {len(indices)} scenarios; n = 10 per scenario/method; training seed = 101.\nUnsmoothed Mean queue; best values compared within controller.',family+'_mean_queue')
            self.heatmap(stack,scenarios,root/'summary',network,split)

    def heatmap(self, stack, scenarios, directory, network, split):
        fig,axes = plt.subplots(2,2,figsize=(18,12))
        low,high = float(stack.min()),float(stack.max())
        for i,(c,ax) in enumerate(zip(CONTROLLERS,axes.flat)):
            values = stack[:,i,:]; image = ax.imshow(values,cmap='viridis_r',vmin=low,vmax=high,aspect='auto')
            ax.set_title(c.upper()); ax.set_xticks(range(5),HEADERS,fontsize=9)
            ax.set_yticks(range(len(scenarios)),[s['label']['en'] for s in scenarios],fontsize=9)
            ax.tick_params(length=0)
            for row in range(len(scenarios)):
                for col in range(5):
                    v = values[row,col]; rgba = image.cmap(image.norm(v))
                    luminance = .2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                    ax.text(col,row,f'{v:.1f}',ha='center',va='center',color='black' if luminance>.5 else 'white',fontsize=9)
        fig.suptitle(network.capitalize()+' | '+split.capitalize()+' | Mean queue by scenario',fontsize=16,y=.98)
        fig.subplots_adjust(left=.18,right=.90,bottom=.13,top=.92,wspace=.72,hspace=.30)
        cax = fig.add_axes([.93,.17,.015,.70]); fig.colorbar(image,cax=cax,label='Mean queue (vehicles; lower is better)')
        fig.text(.18,.025,'Unsmoothed statistics; common color scale; n = 10 per scenario/method; training seed = 101.',fontsize=10)
        self.save(fig,directory/'mean_queue_heatmap')


def dest_mkdir(path):
    path.mkdir(parents=True,exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parent.parent/'docs/evaluation_workbook/grid_results_site/dist/data')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args = parser.parse_args(); output = args.output.resolve()
    if (output/'manifest.json').exists() or (output/'real_world').exists():
        output = output/('version_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    build_archive(args.source.resolve(), output)


def build_archive(source, output):
    output.mkdir(parents=True,exist_ok=True)
    script = Path(__file__).resolve()
    if script.parent != output: shutil.copy2(script,output/script.name)
    e = Exporter(source,output)
    registry = json.loads((e.source/'evaluation_sets.json').read_text())
    e.copy(e.source/'evaluation_sets.json',output/'data'/'evaluation_sets.json')
    sets = {s['id']:s for s in registry['evaluationSets']}
    for network,set_id in [('grid','external_group12'),('monaco','monaco_repaired_full14')]:
        n = next(n for n in sets[set_id]['networks'] if n['id'] == network)
        assert n['available']
        e.dataset(e.source/Path(n['base']).relative_to('data'),network,'real_world')
    for split in ['seen','test']:
        for network in ['grid','monaco']:
            n = next(n for n in sets['main_v7']['networks'] if n['id'] == network)
            e.dataset(e.source/Path(n['base']).relative_to('data'),network,split)
    assert len(e.checks) == 48
    (output/'validation_report.json').write_text(json.dumps({'status':'passed','scenario_count':48,'groups':960,'rollouts':9600,
        'curve_figures':192,'rendered_figures_and_tables':len(e.figures),'checks':e.checks,
        'checks_per_group':['coverage','finite_nonnegative','time_1_to_3600','min_mean_max_order','rollout_mean_min_max_sd','timeseries_mean','EMA_recurrence','text_within_canvas','no_text_overlap','PNG_dimensions']},indent=2)+'\n')
    (output/'README.md').write_text('''# Evaluation results figures

English figures and Mean queue tables reproduced from the local results-site snapshot.

- Real world: 8 single-controller curves, 2 four-controller overviews, 2 scenario tables.
- Seen: 88 curves, 22 scenario tables, 2 network summaries and 2 heatmaps.
- Test: 96 curves, 24 scenario tables, 2 network summaries, 8 family summaries and 2 heatmaps.
- Figures and table images: PNG at 300 dpi, PDF, SVG. Text tables: Markdown, CSV, LaTeX.

## Statistical definitions

Network queue is the sum of queued vehicles over all monitored lanes at each second.
Solid curves show the mean across 10 paired evaluation runs; shading is the pointwise
minimum–maximum range, not a confidence interval. The mean and both band bounds use
EMA smoothing: s[t] = 0.9 s[t-1] + 0.1 x[t], initialized at the first observation.
Tables and heatmaps use unsmoothed Mean queue (vehicles; lower is better).
Each curve figure and overview panel uses an independent y-axis range starting at
zero and ending 5% above its largest smoothed min–max upper bound (at least 1 vehicle).
Real-world display titles are Hangzhou realworld and Monaco real world data.
All image exports include a Grid Network or Monaco Network heading above the title.
Tables display two decimals; bold indicates the lowest unrounded mean within a controller,
including exact ties. CSV retains export precision. Summary tables weight scenarios equally.
All policies use training seed 101; these results do not measure across-training-seed variability.

## Dataset provenance

Grid uses external_group12 (grid_sparse_native_2983), labeled Hangzhou realworld data
for grid by the site. The upstream source mapping is unverified; the local sparse OD
input must not be presented as a verified Hangzhou mapping.
Monaco uses monaco_repaired_full14 (monaco_repaired_full14_v1): all 14 OD pairs
on repaired topology, with frozen policies. This is a repaired-map transfer evaluation.
The older partial-demand Monaco legacy export is excluded.
Seen and Test use main_v7. Supplementary Monaco and main Monaco have different
network topology provenance and should not be pooled as one experiment.

## Files and reproduction

Each split/network/scenario contains figures/, tables/, and data/. Network data/
preserves complete exported metrics and metadata; scenario data/ contains raw series
and selected metrics. summary/ contains equal-scenario tables and heatmaps.
manifest.json records source/output SHA-256 hashes and configuration.
validation_report.json records numerical and render checks; file_index.csv indexes outputs.

Run from the repository root:

```bash
python "evaluation results figs/export_evaluations.py"
```

An existing delivery is preserved; another invocation creates a version directory.
For a relocated script, pass --source /path/to/site/dist/data --output /path/to/archive.
No training, simulation, site mutation, or publication is performed.
''')
    # Visual-review sheets preserve every real-world artifact and representative main scenarios.
    real = sorted((output/'real_world').rglob('*.png'))
    sample = []
    for split in ['seen','test']:
        for network in ['grid','monaco']:
            choices = ['seen_Uniform'] if split == 'seen' else ['mixture_1','peak_1.5','redistribution_0.75','switch_300']
            for sid in choices: sample.append(output/split/network/sid/'figures'/'ia2c_queue.png')
            sample.append(output/split/network/'summary'/'mean_queue_heatmap.png')
    for label,paths in [('real_world',real),('representative',sample)]:
        for start in range(0,len(paths),6):
            batch = paths[start:start+6]; sheet = Image.new('RGB',(1500,540*((len(batch)+1)//2)),'white'); draw = ImageDraw.Draw(sheet)
            for i,p in enumerate(batch):
                with Image.open(p) as im: thumb = ImageOps.contain(im,(740,505)); sheet.paste(thumb,((i%2)*750,(i//2)*540+25))
                draw.text(((i%2)*750+5,(i//2)*540+5),str(p.relative_to(output)),fill='black')
            dest_mkdir(output/'qa'); sheet.save(output/'qa'/f'{label}_{start//6+1}.jpg')
    files = sorted(p for p in output.rglob('*') if p.is_file() and p.name not in ['manifest.json','file_index.csv'])
    records = [{'path':str(p.relative_to(output)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    pd.DataFrame(records).to_csv(output/'file_index.csv',index=False)
    manifest = {'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'source_root':str(e.source),
        'settings':{'training_seed':101,'evaluation_runs':10,'smoothing':.9,'smoothing_initialization':'first observation',
                    'table_statistics':'unsmoothed','summary_weighting':'equal scenarios','methods':METHODS,'colors':COLORS,
                    'curve_y_axis':'independent per controller, zero baseline, 5% upper-band padding',
                    'real_world_titles':{'grid':'Hangzhou realworld','monaco':'Monaco real world data'}},
        'source_sha256':e.sources,'outputs':records,'software':{'matplotlib':matplotlib.__version__,'numpy':np.__version__,'pandas':pd.__version__}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Archive complete: '+str(output),flush=True)


if __name__ == '__main__': main()
