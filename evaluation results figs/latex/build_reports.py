#!/usr/bin/env python3
"""Build and audit two A3 landscape LaTeX evaluation reports."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / 'assets'
BUILD = HERE / 'build'
QA = HERE / 'qa'
SOURCES = {}
PAGES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    SOURCES[str(path.relative_to(ROOT))] = sha(path)


def tex(text):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
                    '^': r'\textasciicircum{}', 'σ': r'\ensuremath{\sigma}',
                    '×': r'\ensuremath{\times}', '–': '-', '—': '-', '·': '/'}
    return ''.join(replacements.get(c,c) for c in text)


def table_body(path):
    record(path)
    content = path.read_text()
    match = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}',content,re.S)
    if not match:
        raise ValueError(f'Missing tabular body: {path}')
    return match.group()


PREAMBLE = r'''\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[a3paper,landscape,left=12mm,right=12mm,top=12mm,bottom=14mm,includefoot]{geometry}
\usepackage{graphicx,booktabs,caption,multicol,fancyhdr,lastpage}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdfauthor={Evaluation results archive},pdfsubject={Traffic signal control evaluation},bookmarksnumbered=true}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[L]{\small CBWCE evaluation results}
\fancyfoot[R]{\small Page \thepage\ of \pageref*{LastPage}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
\setlength{\tabcolsep}{16pt}
\renewcommand{\arraystretch}{1.20}
\captionsetup{font=small,labelfont=bf,skip=3pt}
\newcommand{\PageHeading}[3]{\normalsize\phantomsection\label{#1}\pdfbookmark[1]{#2}{bookmark-#1}
{\Large\bfseries #2}\hfill {\large #3}\par\vspace{2pt}}
\newcommand{\Chart}[3]{\begin{center}\includegraphics[width=#3]{#1}
\captionof{figure}{#2}\end{center}}
\newcommand{\QueueTable}[2]{\begin{center}\captionof{table}{#1}
\input{#2}\par\vspace{2pt}{\small Mean queue (vehicles; lower is better). Bold: lowest unrounded mean within each controller.}\end{center}}
\begin{document}
'''


def add_page(pages, identifier, title, network, content, kind, scenario=None):
    body = r'\PageHeading{'+identifier+'}{'+tex(title)+'}{'+tex(network)+'}\n'+content
    pages.append({'id':identifier,'title':title,'network':network,'body':body,'kind':kind,'scenario':scenario})


def materialize_table(path):
    relative = path.relative_to(ROOT)
    dest = ASSETS/'tables'/relative
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(table_body(path)+'\n')
    return str(dest.relative_to(HERE)).replace(os.sep,'/')


def graph(path, caption, width):
    # Graphic paths are relative to the build working directory (latex/).
    if path.is_relative_to(ROOT) and not path.is_relative_to(ASSETS): record(path)
    name = os.path.relpath(path,HERE).replace(os.sep,'/')
    return r'\Chart{'+name+'}{'+tex(caption)+'}{'+width+'}\n'


def queue_table(path, caption):
    return r'\QueueTable{'+tex(caption)+'}{'+materialize_table(path)+'}\n'


def overview(export, split, network, scenario):
    sid = scenario['id']; dest = ROOT/split/network/sid
    fig,axes = plt.subplots(2,2,figsize=(14,8.0))
    for controller,ax in zip(export.CONTROLLERS,axes.flat):
        curves = {}
        for method in export.METHODS:
            path = dest/'data'/'series'/controller/(method+'.csv'); record(path)
            frame = pd.read_csv(path)
            data = frame[['queue_min','queue_mean','queue_max']].to_numpy()
            smooth = np.column_stack([export.ema(data[:,i]) for i in range(3)])
            curves[method] = (frame.time.to_numpy(),smooth[:,0],smooth[:,1],smooth[:,2])
        export.Exporter(ROOT,ROOT).draw_axes(ax,curves,controller.upper())
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(12); ax.yaxis.label.set_size(12)
    title = network.capitalize()+' Network | '+split.capitalize()+' | '+scenario['label']['en']
    fig.suptitle(title,fontsize=16,y=.98)
    fig.subplots_adjust(left=.07,right=.985,bottom=.17,top=.88,hspace=.42,wspace=.22)
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.025),ncol=5,frameon=False,fontsize=12)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = []
    for artist in fig.findobj(export.Text):
        if not artist.get_visible() or not artist.get_text().strip(): continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1+1 and box.y1 <= fig.bbox.y1+1
        for label,previous in bounds:
            assert min(box.x1,previous.x1)-max(box.x0,previous.x0)<=1 or min(box.y1,previous.y1)-max(box.y0,previous.y0)<=1,(title,artist.get_text(),label)
        bounds.append((artist.get_text(),box))
    path = ASSETS/'overviews'/split/network/(sid+'.pdf')
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,facecolor='white')
    fig.savefig(path.with_suffix('.png'),dpi=200,facecolor='white')
    plt.close(fig)
    return path


def report_heatmap(export, split, network, scenarios):
    path = ROOT/split/network/'data'/'metrics_summary.csv'; record(path)
    rows = pd.read_csv(path)
    values = np.array([[rows[(rows.scenario==s['id']) & (rows.controller==c) & (rows.method==m)].iloc[0].mean_queue_mean
                        for c in export.CONTROLLERS for m in export.METHODS] for s in scenarios])
    fig,ax = plt.subplots(figsize=(14.8,6.7))
    image = ax.imshow(values,cmap='viridis_r',aspect='auto')
    ax.set_yticks(range(len(scenarios)),[s['label']['en'] for s in scenarios],fontsize=12)
    ax.set_xticks(range(20),['B','RG','DR','FW','OW']*4,fontsize=12)
    ax.tick_params(length=0)
    for i,c in enumerate(export.CONTROLLERS):
        ax.text(i*5+2,-1.1,c.upper(),ha='center',va='center',fontsize=14,weight='bold',clip_on=False)
        if i:ax.axvline(i*5-.5,color='white',lw=2)
    for i,row in enumerate(values):
        for j,value in enumerate(row):
            rgba=image.cmap(image.norm(value));lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
            ax.text(j,i,f'{value:.1f}',ha='center',va='center',fontsize=10,
                    color='black' if lum>.5 else 'white')
    fig.subplots_adjust(left=.23,right=.92,top=.86,bottom=.15)
    cax=fig.add_axes([.94,.15,.015,.71]);fig.colorbar(image,cax=cax,label='Mean queue (vehicles)')
    cax.tick_params(labelsize=11);cax.yaxis.label.set_size(12)
    fig.text(.23,.04,'B: Baseline   RG: Random grouping   DR: Domain randomization   FW: Fixed WCE   OW: Online WCE',fontsize=12)
    out=ASSETS/'heatmaps'/split/(network+'.pdf');out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,facecolor='white');fig.savefig(out.with_suffix('.png'),dpi=200,facecolor='white');plt.close(fig)
    return out


def scenario_pages(export):
    pages = []; standalone = []
    for network,sid in [('grid','grid_sparse_native_2983'),('monaco','monaco_repaired_full14_v1')]:
        root = ROOT/'real_world'/network/sid
        title = 'Hangzhou realworld' if network=='grid' else 'Monaco real world data'
        content = graph(root/'figures'/'all_controllers_queue.png',title+' - four controllers and five methods.','11.2in')
        content += queue_table(root/'tables'/'mean_queue.tex',title+' - unsmoothed Mean queue.')
        content += r'\small n = 10 evaluation runs per controller/method; training seed = 101. Solid: mean; shaded bands: pointwise min-max; EMA = 0.9. Independent y-axis ranges.'+'\n'
        add_page(pages,'real-'+network,title,network.capitalize()+' Network',content,'scenario',sid)
        standalone.append(pages[-1])
    for split in ['seen','test']:
        for network in ['grid','monaco']:
            path = ROOT/split/network/'data'/'catalog.json';record(path)
            catalog = json.loads(path.read_text())
            scenarios = [s for s in catalog['scenarios'] if s['split']==split]
            summary = ROOT/split/network/'summary'
            heading = split.capitalize()+' evaluation - network summary'
            heatmap = report_heatmap(export,split,network,scenarios)
            content = graph(heatmap,heading+' - unsmoothed Mean queue by scenario.','14.0in')
            content += queue_table(summary/'mean_queue.tex',heading+' - equal weight per scenario.')
            content += r'\small Each scenario has 10 evaluation runs per controller/method; training seed = 101. Summary values weight scenarios equally.'+'\n'
            add_page(pages,split+'-'+network+'-summary',heading,network.capitalize()+' Network',content,'network_summary')
            if split=='test':
                content = ''
                for family in ['redistribution','mixture','temporal','peak']:
                    content += queue_table(summary/(family+'_mean_queue.tex'),family.capitalize()+' - equal weight across the three scenarios.')+r'\vspace{8mm}'+'\n'
                content += r'\small Each family contains three frozen test scenarios. n = 10 per scenario/controller/method; training seed = 101.'+'\n'
                add_page(pages,'test-'+network+'-families','Test evaluation - demand-family summaries',network.capitalize()+' Network',content,'family_summary')
            for scenario in scenarios:
                root = ROOT/split/network/scenario['id']
                title = scenario['label']['en']
                image = overview(export,split,network,scenario)
                content = graph(image,split.capitalize()+' evaluation: '+title+' - four controllers and five methods.','13.4in')
                content += queue_table(root/'tables'/'mean_queue.tex',title+' - unsmoothed Mean queue.')
                content += r'\small n = 10 evaluation runs per controller/method; training seed = 101. Solid: mean; shaded bands: pointwise min-max; EMA = 0.9. Independent y-axis ranges.'+'\n'
                add_page(pages,split+'-'+network+'-'+scenario['id'],title,network.capitalize()+' Network / '+split.capitalize(),content,'scenario',scenario['id'])
            print(f'Prepared {split}/{network}: {len(scenarios)} scenario pages',flush=True)
    return standalone,pages


def intro(pages):
    return r'''\pdfbookmark[0]{Complete evaluation report}{complete-report}
{\Huge\bfseries Complete evaluation report}\par\vspace{8mm}
{\Large Real-world, Seen and Test results / Grid and Monaco networks}\par\vspace{8mm}
\begin{minipage}[t]{0.57\textwidth}
\section*{How to read this report}
Each scenario page combines four controller panels (IA2C, MA2C, IQLL and PPO)
with a selectable-text LaTeX table. The five methods are Baseline, Random grouping,
Domain randomization, Fixed WCE and Online WCE.

Network queue is the sum of queued vehicles across monitored lanes at each second.
Solid curves are 10-run means; shaded bands are pointwise minimum-to-maximum ranges,
not confidence intervals. Curves and band boundaries use EMA = 0.9, initialized at
the first observation. Each panel starts at zero and uses its own adaptive y-axis.

Tables and heatmaps use unsmoothed Mean queue (vehicles; lower is better).
Bold values identify the lowest unrounded mean within each controller, including
exact ties. Values are displayed to two decimal places. Network and family summaries
give equal weight to scenarios.

All results use training seed 101 and 10 evaluation runs per scenario/controller/method.
The sampling ranges measure evaluation variability; they do not measure variability
across independently trained policies.

\section*{Coverage}
48 scenarios, 192 controller panels, 60 Mean queue tables and four heatmaps.
Real-world: two scenarios. Seen: 11 scenarios per network. Test: 12 scenarios per
network, organized into redistribution, mixture, temporal switching and peak demand.
The report contains 9,600 evaluation runs across 960 controller/method/scenario groups.
\end{minipage}\hfill
\begin{minipage}[t]{0.38\textwidth}
\section*{Report structure}
\begin{tabular}{ll}
Real-world & Grid and Monaco scenario pages\\
Seen & Network summaries and scenario pages\\
Test & Network summaries, family tables and scenarios
\end{tabular}
\section*{Navigation and sources}
Use the clickable contents or PDF bookmarks to jump to a network, summary or scenario.
The adjacent real-world report contains the two real-world pages as a standalone document.

Figures reproduce the archived evaluation series. Real-world pages embed the specified
original overview PNGs; their tables are taken directly from the corresponding LaTeX
tabular environments. Seen/Test overview graphics preserve all four controllers and
five methods while avoiding repeated single-controller pages.

The report archive includes editable LaTeX sources, generated overview graphics,
a build script, an input/output hash manifest and a validation report. Numerical values
are carried over from the existing archive, without running training or simulation.
\end{minipage}
\vfill{\small A3 landscape / English / Compact scenario overviews}
'''


def contents(pages):
    body = r'\pdfbookmark[0]{Contents}{contents}\section*{Contents}\begin{multicols}{3}\small'+'\n'
    for p in pages:
        title = p['network']+' / '+p['title']
        body += r'\noindent\hyperref['+p['id']+']{'+tex(title)+r'}\hfill\pageref*{'+p['id']+r'}\par\vspace{3pt}'+'\n'
    return body+r'\end{multicols}'+'\n'


def compile_report(name, pages, complete):
    source = PREAMBLE.replace(r'\begin{document}',r'\hypersetup{pdftitle={'+('Complete evaluation report' if complete else 'Real-world evaluation report')+r'}}\begin{document}')
    bodies = ([intro(pages),contents(pages)] if complete else [])+[p['body'] for p in pages]
    source += '\n\\newpage\n'.join(bodies)+r'\end{document}'+'\n'
    path = HERE/(name+'.tex');path.write_text(source)
    for attempt in range(3):
        result = subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error',
                                  '-output-directory='+str(BUILD),path.name],cwd=HERE,capture_output=True,text=True)
        (BUILD/(name+f'.pass{attempt+1}.txt')).write_text(result.stdout)
        if result.returncode: raise RuntimeError(result.stdout[-5000:])
    log = (BUILD/(name+'.log')).read_text(errors='replace')
    for bad in ['Overfull \\hbox','Overfull \\vbox','Missing character:','There were undefined references','Rerun to get cross-references right']:
        if bad in log: raise RuntimeError(f'{name}: {bad}; inspect {name}.log')
    shutil.copy2(BUILD/(name+'.pdf'),ROOT/(name+'.pdf'))
    PAGES[name] = ([{'id':'reading-guide','kind':'intro'},{'id':'contents','kind':'contents'}] if complete else [])+[{k:v for k,v in p.items() if k!='body'} for p in pages]
    print('Compiled '+name,flush=True)


def audit_and_render(name):
    pdf = ROOT/(name+'.pdf')
    info = subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    page_count = int(re.search(r'Pages:\s+(\d+)',info).group(1))
    assert page_count == len(PAGES[name]),(name,page_count,len(PAGES[name]))
    size = tuple(map(float,re.search(r'Page size:\s+([\d.]+) x ([\d.]+)',info).groups()))
    assert np.allclose(size,(1190.55,841.89),atol=1),size
    text = subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
    extracted = text.split('\f')[:page_count]
    assert len(extracted)==page_count
    for i,metadata in enumerate(PAGES[name]):
        page = extracted[i]
        assert f'Page {i+1} of {page_count}' in page
        if metadata['kind'] in ['scenario','network_summary','family_summary']:
            for label in ['IA2C','MA2C','IQLL','PPO','Baseline','Mean queue']: assert label in page,(i,label)
        if metadata['kind'] in ['scenario','network_summary','family_summary']:
            split = 'real_world' if metadata['id'].startswith('real-') else metadata['id'].split('-')[0]
            network = 'grid' if 'Grid Network' in metadata['network'] else 'monaco'
            if metadata['kind']=='scenario':
                tables = [ROOT/split/network/metadata['scenario']/'tables'/'mean_queue.csv']
            elif metadata['kind']=='network_summary':
                tables = [ROOT/split/network/'summary'/'mean_queue.csv']
            else:
                tables = [ROOT/split/network/'summary'/(f+'_mean_queue.csv') for f in ['redistribution','mixture','temporal','peak']]
            for table in tables:
                record(table)
                values = pd.read_csv(table).iloc[:,1:].to_numpy(dtype=float)
                for value in values.flat: assert f'{value:.2f}' in page,(metadata['id'],value)
    bbox_path = QA/(name+'_text_bounds.xhtml')
    subprocess.run(['pdftotext','-bbox',str(pdf),str(bbox_path)],check=True)
    tree = ET.parse(bbox_path)
    for page in [n for n in tree.getroot().iter() if n.tag.endswith('}page')]:
        width,height = float(page.attrib['width']),float(page.attrib['height'])
        for word in [n for n in page.iter() if n.tag.endswith('}word')]:
            bounds = [float(word.attrib[k]) for k in ['xMin','yMin','xMax','yMax']]
            assert bounds[0]>=0 and bounds[1]>=0 and bounds[2]<=width and bounds[3]<=height,(name,word.text,bounds)
    render = QA/name;render.mkdir(parents=True,exist_ok=True)
    subprocess.run(['pdftoppm','-r','60','-png',str(pdf),str(render/'page')],check=True,capture_output=True)
    images = sorted(render.glob('page-*.png'),key=lambda p:int(re.search(r'(\d+)\.png$',p.name).group(1)))
    assert len(images)==page_count
    for start in range(0,len(images),6):
        batch = images[start:start+6];sheet = Image.new('RGB',(1800,1280),'#e4e7eb');draw=ImageDraw.Draw(sheet)
        for i,p in enumerate(batch):
            with Image.open(p) as im: thumb=ImageOps.contain(im,(890,397));sheet.paste(thumb,((i%2)*900+5,(i//2)*426+24))
            draw.text(((i%2)*900+10,(i//2)*426+6),f'{name} - page {start+i+1}',fill='black')
        sheet.save(QA/(name+f'_contact_{start//6+1:02}.jpg'),quality=95)
    return {'pages':page_count,'page_size_points':size,'page_content_checks':'passed',
            'text_bounds_checks':'passed','rendered_pages':len(images),'warnings':'No overfull boxes, missing glyphs or unresolved references.'}


def main():
    for p in [ASSETS,BUILD,QA]:p.mkdir(parents=True,exist_ok=True)
    module = ROOT/'export_evaluations.py';record(module)
    spec = importlib.util.spec_from_file_location('evaluation_export',module)
    export = importlib.util.module_from_spec(spec);spec.loader.exec_module(export)
    standalone,pages = scenario_pages(export)
    assert len(pages)==54 and sum(p['kind']=='scenario' for p in pages)==48
    compile_report('real_world_evaluation_report',standalone,False)
    compile_report('complete_evaluation_report',pages,True)
    checks = {name:audit_and_render(name) for name in PAGES}
    (HERE/'report_pages.json').write_text(json.dumps(PAGES,indent=2)+'\n')
    (HERE/'validation_report.json').write_text(json.dumps({'status':'passed','reports':checks,
        'coverage':{'scenarios':48,'controller_panels':192,'tables':60,'heatmaps':4},
        'visual_review':'Rendered pages and contact sheets ready for visual inspection.'},indent=2)+'\n')
    (HERE/'README.md').write_text('''# English LaTeX evaluation reports

Both reports use A3 landscape pages. The real-world report has one page per network.
The complete report includes a reading guide, clickable contents, 48 scenario pages,
four network-summary pages and two Test family-summary pages (56 pages total).

Each scenario contains all four controllers, five methods and an unsmoothed Mean queue
table. The real-world report uses the two specified original overview PNGs and the
two original LaTeX table bodies. Table numbers and bold formatting are preserved.
Sources and hashes are recorded in report_manifest.json. report_pages.json maps each
page to its content. validation_report.json records compile and content checks.

Rebuild from the repository root:

```bash
MPLCONFIGDIR=/tmp/evaluation-matplotlib python -B "evaluation results figs/latex/build_reports.py"
```

Requirements: Python with NumPy, pandas, Matplotlib and Pillow; pdflatex; Poppler
pdfinfo, pdftotext and pdftoppm. LaTeX uses Latin Modern, geometry, graphicx, caption,
multicol, fancyhdr, lastpage and hyperref. No simulation or training is run.
Generated assets are in assets/. Editable report sources are the two root .tex files.
QA renders and contact sheets are in qa/. Temporary compiler outputs are in build/.
The original standalone evaluation figures and tables are unchanged.
''')
    refresh_manifest()
    print('Reports ready in '+str(ROOT),flush=True)


def refresh_manifest():
    outputs = [ROOT/(name+'.pdf') for name in ['real_world_evaluation_report','complete_evaluation_report']]
    outputs += sorted(p for p in HERE.rglob('*') if p.is_file() and 'build' not in p.relative_to(HERE).parts
                      and p.name not in ['report_manifest.json','file_index.csv'] and '__pycache__' not in p.parts)
    records = [{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in outputs]
    pd.DataFrame(records).to_csv(HERE/'file_index.csv',index=False)
    (HERE/'report_manifest.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
        'settings':{'page_size':'A3 landscape','language':'English','layout':'compact scenario overviews',
                    'training_seed':101,'runs_per_group':10,'EMA':.9},
        'source_sha256':SOURCES,'outputs':records},indent=2)+'\n')


if __name__=='__main__':main()
