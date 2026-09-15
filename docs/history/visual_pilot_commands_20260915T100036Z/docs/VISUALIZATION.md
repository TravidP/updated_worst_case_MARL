# Optional SUMO visualization / 可选 SUMO 可视化

Updated 15 September 2026. The corrected `main.py experiment` interface now supports an optional live SUMO window. It is off by default.

## Commands

From the repository root, activate `deeprlsc` and choose one mode:

```bash
conda activate deeprlsc
python main.py experiment parent --network grid --controller ia2c \
  --seed 9001 --pilot --steps 160 --visualization
```

Replace `--visualization` with `--no-visualization`, or omit it, for headless execution. Do not supply both options or use `--visualization false`. The flags also work for frozen-parent `wce`, all five `continue` methods, and single/suite `evaluate` jobs. Existing checkpoint and gate arguments remain required. Compatibility commands through `python -m revision.runner` support the same flags. Historical `main.py train/evaluate` commands retain their previous options.

```bash
python main.py experiment parent --help
python main.py experiment check
python main.py experiment dashboard
```

In the English or Chinese workspace, select **SUMO visualization / SUMO 可视化 → On / 开启** under Job settings. The generated command, API request and stored browser settings follow that selection. An already running dashboard server needs to be restarted when idle to load the updated backend; refresh the page afterward.

## Behavior and requirements

- The window appears on the **training computer**, not inside the HTML page. Linux requires an accessible X display via `DISPLAY`, and `sumo-gui` must be on `PATH`. The preflight checks for missing prerequisites; it cannot guarantee that a supplied display is accessible.
- GUI episodes use `sumo-gui --start --quit-on-end`. Playback starts automatically. SUMO View Settings controls appearance; Delay controls playback speed. Use the dashboard's Stop job control for a managed interruption.
- Choose the mode before launch. It is not a switch for a running SUMO process. A new resume attempt may choose either mode without altering checkpoint compatibility or its original budget.
- Initialization and route-preparation sessions may remain headless. Dataset preparation/materialization does not enable a GUI. Each training or evaluation episode uses the selected display mode.
- Rendering and manual playback delay add wall-clock overhead. Keep visualization off for timed publication comparisons. Reward definitions, RNG streams, simulation step length, learning cadence, and training budgets are unchanged.
- `<run>/manifest.json` records the requested boolean `visualization`. Each `runtime/startup_*.json` records actual `visualization` and the exact SUMO command. Queue arrays, rewards, checkpoint bundles and result locations retain their existing formats.

## Implementation and evidence

| Location | Change |
|---|---|
| [experiments/runner.py](../experiments/runner.py) | Mutually exclusive CLI switches; pass the selection to the shared environment and record it |
| [envs/experiment_env.py](../envs/experiment_env.py) | Check GUI prerequisites, propagate the choice on episode reset, auto-start GUI playback and record actual startup mode |
| [experiments/cli.py](../experiments/cli.py) | Readiness output includes optional `sumo_gui` and `display`; suite children retain the selected flag |
| [experiments/dashboard.py](../experiments/dashboard.py) | Validate a strict boolean and build explicit on/off subprocess arguments |
| [workspace.js](site/dist/assets/workspace.js) | Matching English/Chinese selector, saved preferences, command builder and API payload |
| [tests/test_visualization.py](../tests/test_visualization.py) | CLI defaults/conflicts, missing prerequisites, seed-preserving reset propagation and local-launch argument checks |
| [experiments/verify.py](../experiments/verify.py) | Include the new visualization tests in future verification runs |

The 22 Python correction/workflow/visualization tests passed. Two real 160-step Grid/IQL pilots, seed 9002, completed with GUI off and on. GUI verification used Xvfb, a temporary virtual X display. Their lane arrays, learner rewards, demand decisions and all saved model/optimizer variables matched exactly. See [the comparison record](../runs_eval/revised/verification/visualization_20260915/comparison.json) and its explicitly named attempt directories. The initial sandbox-restricted attempt failed because local sockets were unavailable; its failure record remains preserved.

English/Chinese command generation, boolean API payloads and saved selection were exercised with a JavaScript fixture. HTTP checks also passed for both language pages, the shared script, GUI-on/off preflight, and rejection of a non-boolean value; no jobs were launched through that temporary test server. These checks are not a visual inspection of the physical desktop. No publication training matrix was launched.

**Publication gate:** this source change invalidates the earlier source-hashed gates. Their historical evidence is preserved. The focused checks above do not replace the full eight-case gate. Run `python main.py experiment verify --workers 4` and select the new passing gate before publication training. Do not edit old gate hashes to bypass verification.

Exact files from before this change are preserved with [checksums](history/visualization_20260915T092606Z/checksums.sha256). The local HTML workspace was updated; the previously hosted private Sites copy was not redeployed by this change.

## 中文使用与验证说明

在仓库根目录激活 `deeprlsc`，给 `python main.py experiment <stage>` 添加 `--visualization` 即可开启；使用 `--no-visualization` 或省略两个选项即可关闭。默认关闭，不能同时传入两个选项，也不能写成 `--visualization false`。支持父模型、冻结父模型 WCE、五种续训方法，以及单文件／套件评估；原有检查点、准入文件和预算要求不变。

本地工作台在“任务设置”中提供“SUMO 可视化”开关。选择会保存到浏览器，切换语言后保留，并传入生成命令和本地 API。已有仪表板后端需要在空闲时重启，再刷新页面。窗口显示在训练电脑上，不嵌入网页；Linux 需要可访问的 `DISPLAY` 和 `sumo-gui`。启动前检查能识别缺少配置，但不能保证已提供的显示地址一定可用。

GUI 自动开始播放。可以通过 View Settings 调整外观，通过 Delay 调整播放速度。请在启动前选择显示模式，不能在运行中直接切换；恢复到新尝试时可重新选择，不改变检查点兼容性和原有预算。初始化与路径准备会话可能保持无窗口运行。论文计时比较应关闭可视化，避免渲染和播放延迟增加耗时。

所选布尔值保存在 `<run>/manifest.json` 的 `visualization` 字段；每个 `runtime/startup_*.json` 保存实际模式与启动命令。奖励、随机流、仿真步长、更新频率、学习预算和其他输出格式不变。

22 项 Python 测试通过；使用相同种子 9002 的两次 Grid/IQL 160 步试运行，分别关闭／开启 GUI，均正常完成。GUI 测试使用 Xvfb 临时虚拟显示。两次运行的车道数据、学习奖励、需求决策、模型及优化器变量完全一致。JavaScript 检查覆盖双语开关、命令、布尔请求和选择保存；HTTP 检查还覆盖双语页面、共用脚本、开启／关闭预检查及非法布尔值拒绝；临时测试服务器没有启动训练任务。这些检查不是对实体桌面的视觉检查。最初因沙箱禁止本地套接字而失败的尝试也予以保留。

本次源代码修改使旧准入文件的哈希失效。以上定向检查不能代替完整八组验证；论文训练前应重新运行 `python main.py experiment verify --workers 4` 并选择新的通过记录。历史证据和修改前文件均已保留，没有启动完整论文矩阵。此次更新的是本地 HTML 工作台，没有重新发布此前的私有 Sites 托管副本。
