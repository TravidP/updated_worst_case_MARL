# 旧数据归档与离线恢复

这些脚本不参与训练。`--root` 指向旧工作区（其中包含 `cbwce_archive/catalogs` 和 `cbwce_archive/staging`），不是新训练仓库本身。网络传输由已授权的 Drive 连接器执行；这里不存储账号 token。

| 文件 | 功能 |
|---|---|
| `migrate_storage.py` | `pack` 普通归档；`pack-large` 超大单文件；`verify` 恢复核验；`cleanup` 精确清理 |
| `archive_transport.py` | `split` 生成 95 MiB 片段；`record` 写入 Drive IDs；`download` 有界并行下载与片段校验 |
| `restore_archive.py` | 离线重组已下载片段，恢复原路径与内部硬链接，拒绝覆盖现有目录 |
| `status.py` | 根据已持久化的批次清单统计进度和实际块释放量 |
| `archive_batch.py` | 首批试迁移时使用的只打包工具，保留作来源记录 |

## 恢复一个批次

从 Drive 下载该批 JSON manifest 和它列出的所有片段，保留片段文件名。对于首批没有分片的归档，下载 manifest 中指定的 `.tar.gz`。然后执行：

```bash
python scripts/storage/restore_archive.py \
  --manifest /path/to/downloads/batch_000002.json \
  --parts-dir /path/to/downloads \
  --destination /path/to/new-restored-directory
```

恢复目录必须尚不存在。工具逐片校验大小与 SHA-256，重组后核验完整归档，再核验每个成员；只允许普通文件和指向已验证成员的内部硬链接，拒绝路径穿越与符号链接。验证失败时不发布恢复目录，下载输入保留。`archive_kind=raw_file` 的批次不解 tar，而是恢复 payload 到各成员路径，并重建硬链接。

成功后的目录保留 `runs_eval/...` 等旧相对路径。manifest 内旧机器的绝对 staging 路径仅作来源记录，恢复只使用片段文件名和成员相对路径。

## 续接迁移

先查看 `migration.json`、数字编号的 `batch_*.json` 和 `deletion_journal.jsonl`。`*_transport_progress.json` 记录已经上传的片段 ID，避免超时后盲目重复上传。

- `packed`：原件保留，等待上传。
- `uploaded`：片段 ID 已记录，尚未完成本地恢复核验。
- `download_verified`：片段、完整归档和成员均已核验。
- `local_removed`：该批源文件已清理。
- `local_removed_untracked_only`：未跟踪源文件已清理；受旧 Git 跟踪的成员保留。

仅在 manifest 已上传并读回、完整下载验证通过后调用 `cleanup`。清理前再次检查源文件大小、mtime 与 SHA-256，先持久化删除 intent，再删除精确列出的文件。原件有变化则停止该批。硬链接最后一个本地引用移除时才计入释放块数。

`hardlink_groups.json` 来自旧清单与实际 inode 盘点；它帮助同一份内容只上传一次，同时保留多个原路径。不要把整个目录 `rm -rf` 或 `git clean` 当作本流程的替代。
