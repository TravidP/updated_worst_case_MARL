# Bilingual CB-WCE workspace

Authored static pages are in `dist/index.html` and `dist/zh.html`. Shared JavaScript and CSS implement the same controls and translated content. Run `python main.py experiment dashboard` from the repository root to enable local training APIs.

The hosted Sites copy contains only these static assets and selected report exports. It cannot launch local training. Publishing uses a separate temporary Git checkout; the main research repository is not committed or pushed.

`data/protocol.js` mirrors `config/revised/protocol.json`. `data/results.js` contains explicitly labeled pilot exports until publication results exist. Refresh it from a validated `dashboard.json` report. Never include local tokens, checkpoint files, absolute paths, or raw historical documents in the hosted bundle.

Both comics were produced with the built-in ImageGen tool. Their exact prompts and English/Chinese transcripts are retained in `dist/assets/`.
