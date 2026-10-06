# Restore results / 恢复结果数据

The repository contains bilingual source, catalogs, validation records and 920
summary rows for Grid and Monaco. The portable archive supplies 920 per-second
CSVs (3,600 rows each) and 9,200 per-rollout metric rows. This is a preliminary
Protocol v7 / training seed 101 snapshot, not a formal Phase F report.

仓库保留双语网站和两套路网的精简结果。完整时序通过 ZIP 恢复，不启动评估。

## Dependencies / 依赖

Python 3.10+ and a modern browser; no third-party packages for restore/serving.
Raw NPZ export additionally needs NumPy and complete original publication data.
Do not use the legacy Python 3.6 training environment for this website.
恢复和查看不需要 SUMO 或 TensorFlow；重新导出才需要 NumPy 和原始评估数据。

## Restore / 恢复

Obtain CBWCE_Grid_Monaco_Portable_20261006.zip from the maintainer; no Release
has been published yet. Version, size and SHA-256 are in release.json.
先从维护者取得 ZIP（尚无公开下载链接），然后在仓库根目录执行：

```bash
python3 docs/evaluation_workbook/grid_results_site/restore_results.py --zip /path/to/CBWCE_Grid_Monaco_Portable_20261006.zip
python3 docs/evaluation_workbook/grid_results_site/server.py --host 127.0.0.1 --port 8878
```

Open <http://127.0.0.1:8878/>. Select Grid or Monaco and use the CSV download
controls for the current selection. CSV paths are /data/series/ and
/data/networks/monaco/series/. 已有相同文件跳过，不同文件拒绝覆盖。

Restoration verifies the entire ZIP and file manifest before writing; it only
restores display CSVs and never overwrites site source or conflicting data.
恢复会检查归档和成员校验值，只补齐展示数据，不覆盖源码或冲突文件。

Without a clone, extract the ZIP and follow README_FIRST.md and
START_WINDOWS.bat / START_MAC_LINUX.sh. Python is still required; this is not a
compiled Windows EXE. Keep the ZIP locally for a future Release, not in Git.
