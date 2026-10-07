# Evaluation results figures

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
