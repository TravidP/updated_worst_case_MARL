# 存储迁移交接记录 — 2026-11-16

## 新训练入口

- 本地：`cbwce_standalone/`，独立 Git 仓库。
- 分支：`training/local-v8.3`。
- 目标远端：https://github.com/TravidP/updated_worst_case_MARL/tree/training/local-v8.3 （已发布，2026-11-16 核验）。
- 起点：`a367a6615a99a1e9f3c1a715248ade461c3bfcbf`，不继承旧仓库的大型历史。
- 旧 `main`、`revision` 不改写，不 force push。
- 已保留已有 README、中英文计划和发布白名单的工作区修改；新增训练输出忽略规则。
- 中文计划 v8.3 面向本地 CPU 多核；英文仍为 v8.2。训练代码仍有 v8.1 的 3600 秒验证协议，不能认为 v8.3 已实现。此次不启动训练。

## 云端归档位置

[Google Drive 归档根目录](https://drive.google.com/drive/folders/1LsS460qo-oZWThkFy2Qh8Ya2EuGvCm_G)。创建后权限检查仅显示所有者；未增加共享权限。

| 分类 | Drive 文件夹 ID |
|---|---|
| 00_handover | 1GsiwC6xHoWJ70qETOan1rc-lM1DD5t5j |
| 01_legacy_source | 1IyIz1uSNqeUbWOkcfWSL_kWAUIj-LhWh |
| 02_legacy_results | 1oQ8RhjI4ZuGqZwqLN7dzrqJm5W6crK9d |
| 03_figures_reports | 1g86WKGIcAL1TBhJXBr9AEilG8aWmBn8_ |
| 04_standalone | 1qSCdogPsqJi6nsKgWsReg8X62YhdK35m |

## 当前状态

- 磁盘开始时：916 GiB 总容量，约 700 GiB 已用、170 GiB 可用。
- 清单：`../cbwce_archive/catalogs/inventory.jsonl`，388,224 个文件，178,405,110,244 逻辑字节。逻辑字节与 `du` 的实际占用可能不同。
- 旧仓库状态、refs、未提交补丁分别保存在同目录 `legacy_status_before.txt`、`legacy_refs_before.txt`、`legacy_uncommitted.patch`。
- 42 字节探针上传并下载核验成功；SHA-256 `910723369fa98bc9ba84fecb1cfaefa6b12e1f62f681c6a3dfc7580caa02987a`。
- 真实结果批次：已完成上传、完整下载和 860 个成员逐项 SHA-256 核验。
- 已删除该批 859 个未跟踪结果，保留 1 个受 Git 跟踪的 README；旧 Git 与 LFS 均保留。
- Drive 总配额尚未核实，不能保证完整历史数据可容纳。

## 归档与恢复规则

清单中的 size/mtime 是初始盘点；打包时才计算内容 SHA-256。每批生成独立压缩包与逐文件 manifest，重新读取包内成员验证哈希。上传后须下载完整包并匹配哈希、检查成员，才允许删除该批精确列出的源文件。任何不匹配均保留原件。

`python scripts/storage/archive_batch.py --help` 显示显式清单打包方式。该工具只打包、不删除，默认每批 256 MiB 源数据；超过单批上限的文件跳过并留在清单，必须另行分片，不能算已归档。完整清单、上传 ID、下载校验和删除记录放在项目外的 `cbwce_archive/catalogs/`。

恢复时先下载归档和 manifest，校验压缩包 SHA-256，再在空目录检查 tar 路径并解包，逐个匹配成员 SHA-256。Git bundle 不包含 Git LFS 的真实对象，旧 `.git` 不得仅凭 bundle 存在就删除。

新训练分支推荐用 `git clone --single-branch --branch training/local-v8.3 --depth 1 https://github.com/TravidP/updated_worst_case_MARL.git` 获取；不要默认拉取全部旧分支历史。

## 本次执行结果

| 产物 | 位置／核验 |
|---|---|
| 结果批次 000001 | [Drive 压缩包](https://drive.google.com/file/d/1z3UNwIeJvLUvgVJIaCvcuEWkJzICyD0J/view)，32,778,487 字节 |
| 批次成员清单 | [上传时的验证清单](https://drive.google.com/file/d/1bf6GN8oOJTrFLNZdkP5RireeOBG4vZr2/view)；删除后状态见本地 `batch_000001.json` 和 `deletion_journal.jsonl` |
| 完整初始盘点 | [压缩清单](https://drive.google.com/file/d/1w9g8T63uyHtYGTZkOZyeLxrknOjREiI7/view)，388,224 项，包含未归档项 |
| 独立项目 Git 备份 | [bundle](https://drive.google.com/file/d/1WEETFhtocZI1MkLh1Dzeml1KSJgQVQCP/view)，截至提交 `270175d`；已下载校验、克隆、`git fsck --full` 通过 |
| 旧源码选取备份 | [132 个源文件和补丁](https://drive.google.com/file/d/1l53DxjVwCVxpob93JpJpHdrCAch3iGso/view)；仅为选取源码，不能代替完整旧 Git/LFS/输入备份 |

结果压缩包 SHA-256：`74fcd25244a65d14ba9fdbdcba91d28bd1da6b3f718e3866fd5ceee04490042f`。
独立项目 bundle SHA-256：`3b4fcf4ce568a21ed32d95460920924ff311da32544921920aa3e86c792094b2`。
初始盘点 gzip SHA-256：`3bcf1b2ff663b7d3a032ab7d999603fb57c1624561cb0cec6f1aa524582fc31e`。

首批源文件清理量：267,508,848 逻辑字节；按文件分配块统计释放 269,414,400 字节（约 256.93 MiB）。打包和下载验证临时副本已删除。保留空目录，未执行 `git clean`、历史改写或旧 Git 删除。

整盘可用空间从执行开始的 182,237,151,232 字节变为检查时的 180,526,747,648 字节；整盘统计受其他写入影响，不能把文件清理量当作整盘净增量。本次归档目录仍保留清单及少量恢复证据。

## 初次执行的阻塞记录与继续入口（认证和容量现已解决）

1. **首次推送遇到认证问题，现已解决并发布。** HTTPS 返回无可用用户名／凭据；SSH 返回 `Permission denied (publickey)`。现有 GitHub 连接器的提交接口要求已有父提交，不能用于保持本次独立根历史。完成本机 GitHub 认证后，在 `cbwce_standalone` 内执行 `git push -u origin training/local-v8.3`，不要 force push。
2. **首次执行等待容量确认；用户已确认，现在持续归档。** Drive 连接器不暴露容量接口；浏览器连接未能打开容量页。需确认该账户剩余容量后继续分批归档。不得把 388,224 项初始清单误认为全部已备份。
3. 旧完整 Git bundle、LFS 对象、其余结果、完整地图输入恢复测试均尚未执行；旧数据保持原位。本次未将旧模型标记为已通过恢复测试。
4. `cbwce_archive/catalogs/migration.json` 是本地机器状态入口，`deletion_journal.jsonl` 逐文件记录删除前核验和删除完成。继续迁移时先读取这些记录，避免重复上传／删除。
5. 此次没有修改训练算法或运行正式训练，v8.3 的本地调度及验证时长同步仍按计划另行实施。

## 防止后台 fetch 重新拉入旧历史

添加远端后检测到新仓库自动出现旧 `origin/main`、`origin/revision` 引用及约 1.74 GiB pack。已仅在独立新仓库删除这些远端跟踪引用并回收不可达对象，旧仓库和远端分支未改动。独立仓库的对象包恢复到约 547 KiB。

新仓库已设置 `remote.origin.fetch=+refs/heads/training/local-v8.3:refs/remotes/origin/training/local-v8.3` 和 `remote.origin.tagOpt=--no-tags`。不要恢复全分支 fetch。归档 bundle 显式选择本地 `main`、`training/local-v8.3`，避免将后台 fetch 的历史一起打包。

一次包含旧远端引用的临时 bundle 超过连接器 512 MiB 上限，上传被拒绝；该临时包已删除并重建为约 541 KiB 的精简包，未删除任何旧项目历史。

## 认证与容量确认后的继续执行

用户已完成 GitHub 登录，并确认 Drive 为无限量校园存储。`training/local-v8.3` 已成功推送；核验时旧远端 `main=1a2e7b2`、`revision=7518db4` 保持不变。此前“认证／容量阻塞”属于历史记录，已经解除。

连接器实际还有 100 MiB 的传输限制，因此每个最多 512 MiB 的归档分成不超过 95 MiB 的传输片段。片段按 manifest 顺序拼接；先验证每片 SHA-256，再验证完整归档和所有成员，最后清理本地源文件。单个片段不能独立解包。每个 manifest 保存各片 Drive ID、大小、哈希及顺序。

新增 `scripts/storage/migrate_storage.py` 和 `scripts/storage/archive_transport.py`，分离打包、传输、验证与清理。清理前重新检查源文件大小、mtime 和 SHA-256；受旧 Git 跟踪的文件先保留，避免旧源码及尚未完成完整历史备份的文件被误删。三项安全测试覆盖损坏下载、源文件变化以及清理日志。

后续批次以 `cbwce_archive/catalogs/batch_*.json` 和 `deletion_journal.jsonl` 为准，不用旧的首批统计推断全部完成。

## 独立代码目录整理

训练所需文件直接作为 `updated_worst_case_MARL` 的 `training/local-v8.3` 分支根目录。Grid 构建辅助文件已移入 `envs/grid_builder.py`；迁移工具集中到 `scripts/storage/`；新增 `requirements.txt`、`.env.example`、`scripts/run_local.sh` 和 `docs/repository_structure_zh.md`。集群模板仅保留为可选文件。

目录调整后，两地图配置检查、15 项单元测试通过；白名单发布包在 `/tmp` 独立解包，并用 Python audit hook 拒绝读取旧工作区。两地图 × 三控制器配置各完成 10 秒 SUMO 环境检查。证据见 `provenance/layout_verification_20261116.json`。这是目录自包含与环境检查，不是正式训练或 v8.3 协议验收。

Google Drive 归档继续，与目录整理并行；未因代码分支工作停止归档。前三批已清理，第四批及后续状态继续记录在项目外 catalogs 中。

## 硬链接与真实释放量

进一步盘点发现 7,720 组硬链接、15,440 个路径，重复路径对应 23,956,581,985 逻辑字节。共演化、图表原始数据和部分 runs 共享这些内容。归档把同一 inode 的路径合并到同一批，tar 使用硬链接条目；超大单文件只上传一份 payload 并列出全部恢复路径。成员验证允许且核验归档内硬链接，不跟随归档外路径。

受旧 Git 跟踪的硬链接也可以随同备份，但本阶段保留本地。清理时仅在最后一个本地硬链接即将移除时计入 `removed_allocated_bytes`；不能用重复路径的逻辑大小声称释放了等量磁盘。恢复时须保留所有 manifest 路径，raw_file 类型没有 tar 层；可按 `hardlink_to` 重建关联。

下载损坏、源文件变化、已记 intent 的中断续接、并行片段重组、raw 单文件恢复、硬链接块数及 tracked 副本保留，均有对应安全测试。

## rclone 授权与切换

用户已完成 Google 授权。对既有私有归档根目录的小文件上传、下载与 SHA-256 核验通过；连接器批次 1–12 已完成核验和清理。后续从第 13 批切换到 `scripts/storage/rclone_migrate.py`，支持完整归档直接传输、逐批恢复验证、持久状态和批次边界停止。

授权配置在项目之外的 `~/.config/rclone/cbwce-archive-20261116.conf`（0600）；便携二进制为 `~/.local/bin/cbwce-rclone`。两者不进入代码发布包。运行状态和日志保存在旧项目 `cbwce_archive/catalogs/rclone_worker.json`、`rclone_worker.log`。归档仍在进行，不表示整体迁移完成；旧仓库 Git/LFS 与未提交源码继续保留。

完成第 12 批后：已校验并清理 191,761 个文件，逻辑大小 45,855,008,917 字节，按删除最后一个硬链接统计释放约 43.1 GiB。此时磁盘可用约 212.8 GiB；最新数值以 `status.py` 输出为准。

## 2026-11-18 输出归档完成核对

原始清单内未跟踪输出已全部处理，141 批，385,341 个文件；每批均完成上传、完整下载 SHA-256 与成员核验，然后清理本地源文件。任务状态为 `inventory_untracked_complete`，不是整个旧项目备份完成。旧 Git/LFS、受 Git 跟踪文件及完整工作区源码仍保留，需单独备份验证后才可清理。

当前发布分支为 `refactor/standalone-local-training`。项目保持白名单结构；旧结果不合并回源码目录。归档清单保存在旧项目的 `cbwce_archive/catalogs`，云端批次 manifest 用于按需恢复；凭据不进入 Git。

## 完整旧仓库备份与清理任务（2026-11-18）

用户已明确授权：完整归档验证后删除旧仓库。`backup_legacy_repository.py` 由独立用户服务 `cbwce-legacy-backup` 执行；状态和日志分别为旧目录 `cbwce_archive/catalogs/legacy_backup_status.json`、`legacy_backup.log`。

范围为剩余旧文件（包含完整 `.git`、LFS、未提交修改和未跟踪文件），排除并保留 `cbwce_standalone`、`cbwce_archive`、`.agents`、`.aws`、`.codex`、`.vscode`。任何符号链接、变化的源文件或校验错误均会中止流程。云端目标为既有私有归档根下 `01_legacy_source/legacy_complete_20261118`。

只有完整下载、逐文件恢复哈希、Git refs、`git fsck --full` 和 LFS 内容哈希通过，且源文件再次核对未变化后，才执行逐文件清理。云端 manifest 和 verification 记录先于删除发布。完整本地备份及恢复副本暂保留至最终交接检查，不立即删除。`legacy_cleaned_backup_retained` 才表示旧文件清理完成；任务启动不等于完成。失败后需检查状态再恢复，不盲目重跑或删除已有备份。
