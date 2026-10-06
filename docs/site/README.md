# Bilingual CB-WCE workspace

## Endpoint rewards and native TensorBoard (protocol 5)

Controller rewards use negative queue at the five-second endpoint, without division by 100; MA2C retains 0.9 neighborhood weighting. Monaco IA2C/MA2C batches are now 120. Training episodes remain 6,600 seconds. `train/reward_by_learning_step` records every actual learner reward; `train/episode/mean_total_queue` summarizes a full episode. Before learning, every 50 complete episodes and at final budget, three paired 600-second Uniform tests save per-second CSV, NPZ and TensorBoard curves.

See [settings, commands, metric definitions and output locations](../TRAINING_MONITORING.md). Defaults are `--monitor-every 50 --monitor-rollouts 3`; pilots may shorten the interval. Start fresh parents and generate a new gate for this reward protocol; historical checkpoints remain preserved. The immediate full campaign launches eight parents only. Exact originals and checksums are archived under `docs/history/endpoint_monitoring_20260916T095506Z/`.

```bash
tensorboard --logdir runs/revised --host 127.0.0.1 --port 6007
```

Active protocol: one independent training run per network/controller combination (seed `101`), 8 parents, 8 offline WCE runs, 40 continuations and 9,200 evaluations. 当前方案：每个路网／控制器组合只训练一次，种子 `101`。Ten paired evaluation rollouts per scenario remain. No training-seed confidence interval is reported. `data/protocol.js` mirrors `config/revised/protocol.json`; historical bundled pilot records remain unchanged. Restart the local dashboard after code changes and refresh either language page.

Authored static pages are in `dist/index.html` and `dist/zh.html`. Shared JavaScript and CSS implement the same controls and translated content. Run `python main.py experiment dashboard` from the repository root to enable local training APIs.

The hosted Sites copy contains only these static assets and selected report exports. It cannot launch local training. Publishing uses a separate temporary Git checkout; the main research repository is not committed or pushed.

`data/protocol.js` mirrors `config/revised/protocol.json`. `data/results.js` contains explicitly labeled pilot exports until publication results exist. Refresh it from a validated `dashboard.json` report. Never include local tokens, checkpoint files, absolute paths, or raw historical documents in the hosted bundle.

Both comics were produced with the built-in ImageGen tool. Their exact prompts and English/Chinese transcripts are retained in `dist/assets/`.

## SUMO visualization / SUMO 可视化

The shared Job settings form has a **SUMO visualization** Off/On selector (pilot default On; publication default Off). It is persisted in browser settings across language switches, included as a boolean in local API requests and rendered as `--visualization` or `--no-visualization` in commands. The GUI opens on the training host, not inside the browser. An idle dashboard server must be restarted after the backend update. The private hosted copy is not automatically redeployed by local edits.

任务设置中的 **SUMO 可视化** 在试运行默认开启、论文实验默认关闭。该布尔选择保存在浏览器中，切换语言后保留，并传入本地 API 与生成命令。GUI 显示在训练主机上，不嵌入浏览器。后端更新后须重启空闲的仪表板服务。本地修改不会自动重新发布私有托管副本。

See [visualization usage and checks](../VISUALIZATION.md). Historical verification snapshots in the site remain labeled; publication requires a fresh gate after this source change.

## Traffic generation explanation / 交通生成说明

The Research & protocol view now explains network-specific normalization, all validation/test families and seeds, Poisson counts, departure jitter, positive speed factors, route materialization, paired artifacts and dataset locations. The same shared section is translated into Chinese. See the full [English](../../reviewer_revision_plan.md#traffic-generation) and [Chinese](../../reviewer_revision_plan_zh.md#traffic-generation) guides.

“研究与实验方案”页面增加双语交通生成说明：各路网归一化、验证／测试类别与种子、泊松数量、出发扰动、正速度因子、路径物化、配对文件及位置。本次仅更新说明，不重新生成交通或重新部署私有托管页面。

Choosing a run type applies its display default. An explicitly saved choice is retained when reopening or switching language; users of older saved forms can select On directly. CLI commands still require explicit `--visualization` to show a window; automated verification stays headless.

选择运行类型时应用其显示默认值；重新打开页面或切换语言保留已保存的选择，旧表单可直接选择开启。CLI 仍需显式 `--visualization` 才显示窗口；自动验证保持无窗口。
