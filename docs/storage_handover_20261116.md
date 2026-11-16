# 存储迁移交接记录 — 2026-11-16

## 新训练入口

- 本地：`cbwce_standalone/`，独立 Git 仓库。
- 分支：`training/local-v8.3`。
- 远端：https://github.com/TravidP/updated_worst_case_MARL/tree/training/local-v8.3
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
- 真实结果批次：处理中，完成状态以追加记录为准。
- 尚未删除任何旧数据；旧 Git 与 LFS 均保留。
- Drive 总配额尚未核实，不能保证完整历史数据可容纳。

## 归档与恢复规则

清单中的 size/mtime 是初始盘点；打包时才计算内容 SHA-256。每批生成独立压缩包与逐文件 manifest，重新读取包内成员验证哈希。上传后须下载完整包并匹配哈希、检查成员，才允许删除该批精确列出的源文件。任何不匹配均保留原件。

`python scripts/archive_batch.py --help` 显示显式清单打包方式。该工具只打包、不删除，默认每批 256 MiB 源数据；超过单批上限的文件跳过并留在清单，必须另行分片，不能算已归档。完整清单、上传 ID、下载校验和删除记录放在项目外的 `cbwce_archive/catalogs/`。

恢复时先下载归档和 manifest，校验压缩包 SHA-256，再在空目录检查 tar 路径并解包，逐个匹配成员 SHA-256。Git bundle 不包含 Git LFS 的真实对象，旧 `.git` 不得仅凭 bundle 存在就删除。

新训练分支推荐用 `git clone --single-branch --branch training/local-v8.3 --depth 1 https://github.com/TravidP/updated_worst_case_MARL.git` 获取；不要默认拉取全部旧分支历史。
