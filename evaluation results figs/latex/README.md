# English LaTeX evaluation reports

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
