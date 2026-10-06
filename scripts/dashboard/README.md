# 本地训练进度

累计wall-clock：优先读取每个attempt的result.wall_seconds，未结束则读取progress.wall_seconds最大值；沿恢复链求和，同一祖先只计一次。任务汇总包含各独立分支耗时（进度仍只取代表分支）。包含初始化、训练、monitor和checkpoint开销，不含两次启动之间的停机时间。运行中的未记录尾段或缺失祖先均标“至少”，不使用文件mtime估计。总时长是并行任务小时的和，不是日历跨度。详情可查看每次运行的时长及数据来源。

```bash
bash scripts/dashboard/start_dashboard.sh
# 浏览器打开 http://127.0.0.1:8766
bash scripts/dashboard/start_dashboard.sh status
bash scripts/dashboard/start_dashboard.sh stop
```

需要系统 Python 3.6+，无额外依赖。默认后台运行，立即扫描，此后每1800秒重新读取训练记录；关闭网页不停止采集。页面每10秒读取缓存快照，不扫描训练目录。手动更新触发提前采集。日志、快照和进程锁位于 `runs_eval/revised/dashboard/`。服务失败时检查 `server.log`。暂不配置开机自启。

可选 `start --port 8766 --interval 1800 --selection PATH`；已有服务需先停止才会使用新参数。单次采集：`python3 scripts/dashboard/dashboard.py scan`。前台运行：`python3 scripts/dashboard/dashboard.py serve`。

远程查看：在自己的电脑运行 `ssh -L 8766:127.0.0.1:8766 用户名@训练电脑`，然后访问本机8766。服务只监听 loopback，不公开训练文件；HTTP只返回聚合信息和固定网页资源，不允许控制训练、删除记录或获取任意文件。

默认56个任务只来自 seed101 selection：8个Parent、8个离线WCE、40个续训。Parent/WCE精确取selection；续训校验身份及parent/WCE hash，排除试运行和不匹配记录。结果完整且达到预算才标记完成。恢复链以manifest.resume关联，祖先记录只读到恢复checkpoint；独立分支分别展示，总计取完成优先、进度最高的单个分支，不将重复实验相加。

阶段steps指控制器交互步（通常每步5秒），不是仿真秒数。Learning steps是累计控制器学习步数；WCE训练冻结控制器，因此不能用该数衡量WCE进度。已完成episode取episode_metrics中最后完整回合的编号，Parent最后一个部分回合独立显示。祖先目录被清理时保留最终结果与累计进度，标记早期曲线缺失，不补造历史点。checkpoint仅检查元数据与列出文件的存在性，不加载pickle或模型，不替代训练启动器的哈希校验。

Monitor只读取已有完整测试。每轮默认3次600秒rollout：reward先对智能体平均，再对120个动作平均，最后跨rollout平均；阴影为样本标准差。raw与learner reward分开展示；queue/waiting取已有summary。缺失指标不伪造、不以负queue替代reward。离线WCE没有定期monitor，其训练曲线是冻结controller reward，并非WCE自身优化奖励。

近期日志（1小时以内）显示“近期有进展”；最近30分钟内创建且无完成/失败记录的测试显示“正在测试（日志证据）”；旧日志显示“长时间无更新”。这些都不保证进程存活。只有/proc可见且main.py的--output精确匹配时显示PID；其余显示未知，不误报停止。失败采集保留旧快照并展示时间和原因。

独立测试：`python3 -m unittest discover -s scripts/dashboard -p 'test_*.py' -v`。全部代码/测试位于scripts/dashboard，避免改变gate所覆盖的训练源码。新增页面采用本地资源，不依赖CDN。
