# Phase E local monitor

只读监控 `runs_eval/revised/publication_seed101_v1`，不会启动、停止或修改 evaluation。

```bash
python3 docs/evaluation_workbook/phase_e_monitor/server.py --port 8877
```

浏览器打开 <http://127.0.0.1:8877/>。页面每 10 秒刷新。

获取一次 JSON 快照：

```bash
python3 docs/evaluation_workbook/phase_e_monitor/server.py --snapshot
```
