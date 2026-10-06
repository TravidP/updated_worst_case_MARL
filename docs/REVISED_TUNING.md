# Revised configuration tuning / Revised 配置筛选

Only `python -m revision.runner` may consume `config/revised`. The legacy
`main.py train` command rejects these files. Publication runs accept only the
tracked eight controller INIs; `--config PATH` is restricted to pilot runs.

只有 `python -m revision.runner` 可以读取 `config/revised`。旧版
`main.py train` 会拒绝这些配置。publication 只能使用八份正式 controller
INI；`--config PATH` 仅允许 pilot 使用。

For each network/controller combination, run the guarded campaign in this
order, using a new timestamped directory:

对每个路网／controller 组合，在新的时间戳目录中按顺序运行：

To execute the same guarded sequence for all eight combinations, use the
matrix driver. It runs one SUMO/TensorFlow job at a time, processes all
combinations phase by phase, skips completed artifacts, and continues with the
other combinations if one has no qualifying candidate:

要自动筛选全部八个组合，可使用总控脚本。它一次只运行一个 SUMO/TensorFlow
任务，按阶段遍历八个组合，自动跳过已完成的产物；某个组合不合格时，其他组合
仍会继续：

```bash
python scripts/training/16_tune_all_revised.py \
  --output output_result/revised/tuning_matrix_YYYYMMDD
```

Preview every command without starting SUMO:

```bash
python scripts/training/16_tune_all_revised.py \
  --output output_result/revised/tuning_matrix_YYYYMMDD \
  --dry-run
```

Use `--through offline` for only the inexpensive first screening, or `--only
grid/ia2c,monaco/ppo` for a subset. Reusing the same output root resumes by
skipping fully completed phases. Interrupted candidate directories are never
deleted or overwritten.

下面是单个组合对应的底层命令：

```bash
python -m experiments.offline_screen collect \
  --network grid --controller ia2c --output CAMPAIGN/trajectory

python -m experiments.tuning create \
  --network grid --controller ia2c \
  --controls CAMPAIGN/trajectory/frozen_episode.controls.jsonl \
  --output CAMPAIGN/search

python -m experiments.tuning launch-offline --campaign CAMPAIGN/search/campaign.json
python -m experiments.tuning launch-pairs --campaign CAMPAIGN/search/campaign.json
python -m experiments.tuning launch-conditionals --campaign CAMPAIGN/search/campaign.json
python -m experiments.tuning rank-pairs --campaign CAMPAIGN/search/campaign.json
python -m experiments.tuning launch-final --campaign CAMPAIGN/search/campaign.json
python -m experiments.tuning promote --campaign CAMPAIGN/search/campaign.json
```

`collect` uses seed 9002 and all eleven demand blocks without learning.
`create` derives the rounded P95 center and isolated `0.5×/1×/2×` candidates.
The offline screen performs 200 identical fixed-batch updates. Pair runs use
2,640 steps and monitors at 0/1,320/2,640. The final run uses 66,000 steps and
monitors every ten episodes. IA2C/MA2C conditional variants are launched only
when the median critic/actor gradient ratio exceeds 10.

`promote` is the only command that can replace a formal controller INI. It
does so only if queue, completed trips, queue slope and clipping-factor gates
all pass. Otherwise it leaves the formal INI unchanged and reports
“当前算法/结构下无合格配置”. Every pilot retains its own TensorBoard events,
candidate table, ranking, gate, and Chinese/English report.

`collect` 使用 seed 9002 和全部 11 个需求 block，且不学习；`create` 根据
P95 生成 1/2/5 舍入中心及隔离候选。离线阶段对同一 batch 更新 200 次；配对
阶段运行 2,640 步，并在 0/1,320/2,640 监测；最终阶段运行 66,000 步，每
10 个 episode 监测。只有所有门槛都通过，`promote` 才会改正式 INI；否则
保留原配置并报告“当前算法/结构下无合格配置”。
