# Bilingual CB-WCE workspace

Authored static pages are in `dist/index.html` and `dist/zh.html`. Shared JavaScript and CSS implement the same controls and translated content. Run `python main.py experiment dashboard` from the repository root to enable local training APIs.

The hosted Sites copy contains only these static assets and selected report exports. It cannot launch local training. Publishing uses a separate temporary Git checkout; the main research repository is not committed or pushed.

`data/protocol.js` mirrors `config/revised/protocol.json`. `data/results.js` contains explicitly labeled pilot exports until publication results exist. Refresh it from a validated `dashboard.json` report. Never include local tokens, checkpoint files, absolute paths, or raw historical documents in the hosted bundle.

Both comics were produced with the built-in ImageGen tool. Their exact prompts and English/Chinese transcripts are retained in `dist/assets/`.

## SUMO visualization / SUMO 可视化

The shared Job settings form has a **SUMO visualization** Off/On selector (default Off). It is persisted in browser settings across language switches, included as a boolean in local API requests and rendered as `--visualization` or `--no-visualization` in commands. The GUI opens on the training host, not inside the browser. An idle dashboard server must be restarted after the backend update. The private hosted copy is not automatically redeployed by local edits.

任务设置中的 **SUMO 可视化** 默认关闭。该布尔选择保存在浏览器中，切换语言后保留，并传入本地 API 与生成命令。GUI 显示在训练主机上，不嵌入浏览器。后端更新后须重启空闲的仪表板服务。本地修改不会自动重新发布私有托管副本。

See [visualization usage and checks](../VISUALIZATION.md). Historical verification snapshots in the site remain labeled; publication requires a fresh gate after this source change.
