# Integrated CB-WCE workspace — 14 September 2026

> Active protocol update (15 September 2026): one training seed (`101`), 8 parents, 8 WCE runs, 40 continuations and 9,200 evaluations. Evidence below describes historical verification, including checks of the earlier statistical design; it does not certify the updated code/protocol.

The corrected implementation now lives in the existing project structure. `main.py experiment` is the public entrypoint; `revision.*` imports the same implementation through compatibility wrappers. Historical training entrypoints retain their previous behavior.

## What changed

| Area | Delivered implementation |
|---|---|
| Controller and WCE | `agents/controller.py`, `agents/wce.py`, `agents/recurrent.py` |
| SUMO measurements | `envs/experiment_env.py` |
| Experiment management | `experiments/`: protocol, paths, immutable records, complete checkpoints, training, evaluation, scenarios, reporting and launcher |
| Inputs | Eight effective controller configurations; explicit eleven-profile normalized training manifests for each network |
| Evaluation preparation | Eleven seen, twelve test and six validation definitions per network; 580 complete paired traffic artifacts |
| User interface | English/Chinese README, reviewer guides, local HTML workspace and private Sites guide |
| Research illustrations | Matching English/Chinese six-panel ImageGen comics, exact prompts and accessible transcripts |

The original source modules, network assets, datasets, checkpoints and historical results remain at their original paths. The tracked-file inventory found no missing files. Sixteen pre-integration snapshots pass [the checksum index](history/checksums.sha256). Only the `main.py` dispatch, documentation and ignore rules replace existing tracked content; the corrected implementation is added alongside the historical code.

## Verification records

The integrated eight-case SUMO verification **passed all C01–C10 checks**. Its [original gate](../runs_eval/revised/verification/integrated_20260914/gate.json) records both networks and all four controller families: eight parents, eight frozen-controller WCE pilots, forty continuation pilots, eight resume checks, sixty-four complete evaluations and eight intentionally interrupted evaluations. Every continuation ended at 2,800 cumulative pilot learning steps (160 parent + 2,640 continuation).

After those pilots, focused changes tightened checkpoint training-seed identity, corrected evaluation-suite method labels, fixed finished-job elapsed time, and excluded evaluation progress from training curves. Learning algorithms, SUMO interaction, rewards and configurations did not change. The final **18 deterministic tests**, seven HTTP checks, one additional full-horizon evaluation and regenerated scientific report passed. The [final source-specific gate](../runs_eval/revised/verification/final_integration_20260914/gate.json) combines that evidence with the original eight-case pilots. It does not claim that the full pilot matrix was rerun after the interface changes. Exact scope and diffs are in [validation_scope.json](../runs_eval/revised/verification/final_integration_20260914/validation_scope.json) and [focused_changes.patch](../runs_eval/revised/verification/final_integration_20260914/focused_changes.patch).

Check the accepted gate without starting training:

```bash
python main.py experiment check --gate runs_eval/revised/verification/final_integration_20260914/gate.json
```

Additional completed checks:

- Ten deterministic correction tests and eight workflow tests passed on the final source.
- Every generated scenario has valid timing and reference rates; all 580 materialized artifacts passed route validation.
- Twenty pilot peak evaluations completed: two networks × five methods × two predefined peaks. These use compatible complete revision checkpoints and are explicitly pilot data.
- Known-answer fixtures check sample SD, five-seed intervals, paired differences, incomplete cells and the common heatmap window.
- Local browser testing completed a 160-step parent job, stopped a longer job after a complete checkpoint, then resumed it in a new attempt to 2,640 steps.
- HTTP checks reject missing session tokens, foreign origins, unknown command fields, invalid stages and publication requests without a gate.
- English/Chinese navigation, results controls, map selection, mobile overflow, script syntax and local links were checked. No blocking browser console errors were observed.

Local launcher records are under `runs_eval/revised/verification/launcher_20260914/` and `runs_eval/revised/jobs/`. The interrupted source attempt is `pilot_20260914T162132_e3839a78`; the completed resume attempt is `pilot_20260914T162434_41ffa2b2`.

## Open the workspace

```bash
conda activate deeprlsc
python main.py experiment dashboard
```

Open [English locally](http://127.0.0.1:8765/) or [中文本地工作台](http://127.0.0.1:8765/zh.html). Choose a pilot, inspect its command and inputs, then launch it. One dashboard job is allowed at a time. Resume requires a complete same-stage checkpoint and creates a new attempt.

[Private bilingual guide](https://cb-wce-training-workspace.loyal-bowl-4834.chatgpt.site) — deployment succeeded through Sites. Access requires the owner's ChatGPT sign-in. The browser reached the authentication screen; authenticated hosted-page interaction could not be checked without that sign-in. The exact static content was tested locally. The hosted copy cannot launch local training.

The main research repository was not committed or pushed. Sites publishing used a separate temporary Git repository containing only the static site and selected pilot exports. No checkpoint binaries, local session tokens or absolute filesystem paths were included.

## Results and limits

The final pilot report is `output_result/revised/pilot_integrated_final_report/`; figure copies are under `figs/revised/pilot_integrated_final_report/`. The earlier report remains preserved. Raw measurements remain the scientific source of truth. Pilot data are not a comparison of trained publication controllers; full-study intervals remain unavailable until all required replicates exist.

The publication matrix has **not** run: 8 parents, 8 offline WCE runs, 40 continuations and 9,200 evaluations remain for explicit launches. Gate checks certify the recorded source and input hashes; later changes require renewed verification. Historical scripts were preserved, not universally certified as bug-free.

Optional WebMCP helpers are registered in the page, but their invocation was not tested because the available browser connection did not expose a supported WebMCP execution tool. The ordinary local interface was exercised directly.

## 中文交付说明

修正代码已合并到熟悉的 `agents/`、`envs/` 和 `experiments/` 目录，统一使用 `python main.py experiment ...`。`revision/` 为兼容入口，不再维护第二套独立实现。历史数据、路网、检查点和结果保留原路径，修改前文件已保存并通过校验。

已提供中英文 README、实验方案、交互网页和六格研究漫画。网页支持单任务启动、进度、停止与恢复；恢复创建新尝试，不覆盖原任务。两路网共 580 个完整交通文件已完成路线验证；20 条峰值试运行评估及其图表仅用于验证流程，不能作为论文性能结论。

八组路网／控制器组合均通过 C01–C10 集成验证，包含四十次续训、八次恢复检查、六十四条完整评估和八条故意中断的评估。随后仅修改了检查点种子校验、评估清单方法标识、结束任务计时和报告曲线分类；学习算法与环境逻辑保持不变。最终十八项测试、七项 HTTP 检查、一条附加完整评估以及重新生成的报告均通过。最终准入文件组合上述证据，明确区分完整试运行与后续定向验证；没有声称最后的界面修改后又重跑了整个八组矩阵。

私有 Sites 部署已成功，查看需使用所有者的 ChatGPT 账户登录。网页托管副本只提供指南与导出结果，训练必须通过本地工作台执行。论文完整训练矩阵尚未启动，主 GitHub 仓库未提交或推送。请参阅 [中文 README](../README_zh.md) 与 [中文实验方案](../reviewer_revision_plan_zh.md)。


## Follow-up: optional SUMO visualization (15 September 2026)

The corrected stage commands and bilingual local workspace now accept an optional SUMO window, off by default. Use `--visualization` / `--no-visualization`; see [the bilingual feature report](VISUALIZATION.md) for commands, requirements, saved fields and scoped verification.

This is a later source change. The 14 September gate above remains historical evidence and no longer matches current source hashes. A new full verification gate is required before publication training. Focused verification passed 22 Python tests and matched GUI/headless 160-step Grid/IQL pilots; it does not claim a rerun of all eight integration cases. Historical data and artifacts remain preserved. The local HTML changed; the previously published private site has not been redeployed.

**中文：** 后续更新增加默认关闭的 SUMO 可视化开关，支持命令行与双语本地工作台。本次代码修改使上方 9 月 14 日准入文件成为历史证据，论文训练前需重新运行完整验证。22 项 Python 检查通过，GUI／无窗口 Grid/IQL 160 步试运行结果一致；这不等于重跑全部八组集成验证。详见[双语功能报告](VISUALIZATION.md)。历史文件保留，私有托管副本未重新发布。
