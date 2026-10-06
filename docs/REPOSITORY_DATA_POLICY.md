# Repository data policy / 仓库数据管理

This repository keeps research source, network inputs, configuration, tests,
reproduction instructions, final selection provenance, and compact website results.
Training checkpoints, runtime output, frozen generated demand and full time series
remain local. Removing files from the Git index does not delete their disk copies.

仓库保留研究源码、路网输入、配置、测试、复现说明、最终 selection 溯源信息和
精简网站结果。训练 checkpoint、运行输出、生成的冻结需求和完整时序保留在本地。
取消 Git 跟踪不会删除磁盘文件。

## Retain / 保留

- `agents/`, `envs/`, `experiments/`, `config/`, `tests/`, `scripts/` and entry points.
- Original network/demand inputs and generators, including legacy experimental inputs.
- Historical run INI configurations and output directory README files.
- `output_result/revised/tensorboard_monitor.py` (source, not generated output).
- Protocol v7 scenario definitions, final selections, and archived verification gates.
- Website HTML/CSS/JS, bilingual catalogs, validation records and metric summaries.

## Local only / 仅本地保留

- Checkpoints, TensorBoard events, logs, runtime rollout files, old documentation snapshots.
- Generated demand artifacts and final model bundles referenced by the selections.
- Website per-second CSVs and per-rollout metric tables; portable ZIP packages.
- Local editor/session markers and notebook checkpoints.
- Unreviewed `paper/` drafts: pending author/publication clearance, not staged.

The exact formerly tracked files, byte sizes and SHA-256 values are recorded in
`repository_cleanup/untracked_artifacts.csv`. Input networks are not removed.
Old Git history still contains large files; this cleanup does not reduce historical
clone size and does not rewrite history.

精确取消跟踪清单、字节数与 SHA-256 见 `repository_cleanup/untracked_artifacts.csv`。
旧历史仍含大文件，本次不改写历史，也不承诺缩小历史克隆体积。

## Portable results / 便携结果

See `evaluation_workbook/grid_results_site/RESTORE.md` for archive verification,
restoration, dependencies, and local viewing. The ZIP is retained locally, not
uploaded or released by this change. Full model bundles and frozen demand are
not included in the display ZIP: obtain the matching local archive from the
maintainer and validate the hashes in the selection before reproducing evaluation.
There is currently no public download URL for these research artifacts.

显示 ZIP 不含最终模型或冻结需求；复现实验需向维护者获取 selection 对应的本地
归档并核对 hash。目前没有公开下载地址，不要用旧模型替代冻结 selection。

## Future publication / 后续发布

1. Review `git log origin/revision..HEAD` and the pending-publication audit.
2. After authorization, run `git push origin revision` (not performed here).
3. Choose a release tag and publish the existing ZIP with its committed SHA-256
   metadata; do not upload the full training tree. Release creation is separate.
4. Add the real download URL to RESTORE.md only after the archive is published.

推送和 Release 均需后续单独执行；本次只创建本地提交。
