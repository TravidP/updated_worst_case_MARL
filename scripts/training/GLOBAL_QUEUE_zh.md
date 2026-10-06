# 4-worker 全局续训队列

入口仍为 `bash scripts/training/16_remaining_continuations_with_cleanup.sh`。
先用 `--dry-run` 检查任务及gate。默认最多 4 项同时训练，按Random Group、Domain Randomization、Fixed WCE、Online WCE的顺序入队；顺序是优先级，不再是方法之间的等待屏障。任一任务退出就补上下一个，未结束任务继续运行。

完成任务跳过；每次启动重新检查最新checkpoint及parent/WCE绑定。失败任务追加到队尾，从最新有效checkpoint恢复，每任务最多启动3次。结果未达完整预算，即使进程返回0也按未完成重试。失败不取消其他任务。Ctrl+C或低磁盘触发的中断不自动重试，之后重复执行入口恢复。

原有全局campaign锁、gate验证和15分钟清理策略保留。只修改scripts下的调度代码和测试，不改变gate覆盖的训练源码。正在运行的旧程序不会热更新，请勿同时另开同一矩阵。

测试：在deeprlsc环境运行 `python -m unittest discover -s scripts/training -p 'test_global_queue.py'`。
