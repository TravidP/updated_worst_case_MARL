# 新训练分支的目录与必要文件

远端仓库：`TravidP/updated_worst_case_MARL`。新分支：`training/local-v8.3`。
本地工作目录仍叫 `cbwce_standalone`；推送后其内容直接位于分支根目录，不在 GitHub 上再套一层同名目录。无需复制旧项目或修改旧 `main`、`revision`。

```text
repository root/
├── main.py                       # 命令行入口
├── tf_compat.py                  # TensorFlow v1 兼容入口
├── requirements.txt             # 引用唯一的依赖锁
├── .env.example                 # 本地路径与线程设置示例
├── agents/                      # MA2C、DQL、PPO、WCE
├── envs/                        # 两地图环境及 grid_builder.py
├── experiments/                 # 需求、训练、验证、checkpoint、统计
├── config/                      # 当前协议与各地图／算法参数
├── assets/
│   ├── grid/                    # 网络、检测器、11 份原始需求
│   └── monaco/                  # 网络、检测器、11 份原始需求
├── scripts/
│   ├── run_local.sh             # 本地单任务入口；默认每种计算线程为 1
│   ├── verify*.py               # 开发验收与恢复检查
│   ├── evaluate_suite.py        # 最终评估
│   ├── report.py                # 规范结果导出
│   ├── package_code.py          # 白名单发布
│   ├── storage/                 # 旧项目迁移工具，不参与训练
│   └── slurm/                   # 保留的可选集群模板，本阶段不使用
├── tests/                       # 算法、协议、存储及迁移安全检查
├── packaging/                   # 依赖锁、发布白名单、可选容器定义
├── provenance/                  # 来源映射和小型验收证据
├── docs/                        # 双语计划、目录说明、实施与归档交接
└── README.md / README_en.md / LICENSE
```

## 必须保留什么

`main.py`、`tf_compat.py`、`agents`、`envs`、`experiments`、`config`、`assets` 构成运行依赖闭包。每地图的网络 XML、检测器 XML 和 11 份需求 CSV 均保留。原 `large_grid/data/build_file.py` 已移至 `envs/grid_builder.py`，保留原作者信息和实现，修正了所有运行时导入及源码身份收集。

`tests`、依赖锁、许可证和 provenance 用于复现与核验，不应为了几 MiB 的空间删除。原始提取映射是历史记录，不覆盖；新位置另记录在 `provenance/layout_20261116.json`。

## 什么不进入这个分支

旧 `runs_eval`、共演化输出、图片集、checkpoint、replay、旧 `.git`、缓存和完整旧 Monaco 目录不复制进新训练分支。实际训练结果放在项目同级 `cbwce_runs`；共享输入与缓存分别放 `cbwce_data`、`cbwce_cache`。旧实验继续按归档清单上传 Drive，校验后清理。

`slurm` 和容器定义只作为可选配套文件保留，不要求本地运行使用 Hábrók 或 Apptainer。`scripts/storage` 只管理已授权的旧文件归档，训练不会导入或调用它。

## 获取与本地检查

```bash
git clone --single-branch --branch training/local-v8.3 --depth 1 \
  https://github.com/TravidP/updated_worst_case_MARL.git updated_worst_case_MARL-local
cd updated_worst_case_MARL-local
# 使用已有 Python 3.10 环境；若需新环境再安装锁定依赖：
python -m pip install --require-hashes -r requirements.txt
bash scripts/run_local.sh check --network grid
bash scripts/run_local.sh check --network monaco
```

`run_local.sh` 使用当前环境的 `python3`，也可指定 `CBWCE_PYTHON=/path/to/python`。SUMO 及其系统库仍需安装；现有机器若从 `/usr/share/sumo/tools` 使用 TraCI，可设置对应 `PYTHONPATH`，不得加入旧项目目录。检查命令不启动训练。

多个独立实验应使用独立进程、不同输出目录；同一实验恢复仍使用原目录。启动脚本只控制单任务线程，尚不是自动并发调度器。当前代码仍按 v8.1 校验 3600 秒 validation；中文 v8.3 计划的 6600 秒 validation 和完整本地调度尚待实现，不因目录整理就宣称完成。

目录变化会改变源码身份哈希；旧验收证据保留为历史证据，重新运行检查后生成新的记录。不要绕过旧 checkpoint 的源码兼容性检查。

## 当前发布分支

整理后的独立项目发布在现有 GitHub 仓库的 `updated_worst_case_MAR` 分支，项目文件直接位于仓库根目录。本地目录仍名为 `cbwce_standalone`，以保持正在运行的归档脚本路径有效。旧 `main`、`revision` 和 `training/local-v8.3` 分支保留。

克隆精简分支（不下载其他分支历史）：

```bash
git clone --single-branch --branch updated_worst_case_MAR --no-tags https://github.com/TravidP/updated_worst_case_MARL.git updated_worst_case_MAR
```

源码、必要地图与十一种需求保存在 Git；训练输出、模型和旧归档保存在项目外部。Google Drive 归档独立运行，切换此代码分支不停止上传。
