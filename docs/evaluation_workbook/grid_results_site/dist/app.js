"use strict";

const UI = {
  zh: {
    skip: "跳到主要内容", eyebrow: "CBWCE 评估工作簿", filtersTitle: "结果筛选",
    loading: "正在读取数据…", loadingSelection: "正在读取当前场景…", ready: "数据完整 · {count} / {expected}",
    error: "数据读取失败", network: "路网", controller: "Controller", split: "评估数据集", scenario: "场景",
    seen: "Seen（训练分布内）", test: "Test（冻结测试集）", external: "External（补充外部需求）", evaluationSet: "评估集合", waitingImport: "等待导入", provenanceDownload: "来源与路线审计 JSON", pairedDownload: "配对比较 CSV",
    gridSubtitle: "4,600 条已完成 rollout 的只读快照；Monaco 可在完整后导入。",
    monacoSubtitle: "4,600 条已完成 Monaco rollout 的只读快照。",
    overviewTitle: "多算法平均排队总览", overviewDescription: "选择需要比较的方法；每条线是 10 次运行的逐秒平均全网 queue。",
    downloadCurrent: "下载当前筛选 CSV", downloadSummary: "下载汇总 CSV", smoothing: "曲线平滑强度",
    smoothingHelp: "0 为原始曲线；数值越大越平滑。", visibleMethods: "总览显示方法",
    chartHint: "滚轮或按钮缩放，按住曲线左右拖拽；所有图共享时间视窗。",
    bandTitle: "双窗口自定义对比", bandDescription: "左右窗口独立多选算法 × 方法。阴影为 10-run min–max，线为均值；颜色区分方法，线型区分算法。共享路网、场景、坐标和平滑强度。",
    tableTitle: "所选算法的完整指标对比", bestValue: "算法内最佳均值（并列均标注）", tableScope: "每个所选 Controller 展示全部五种方法，独立比较最佳值；不受总览方法勾选影响。", tenRunStats: "单元格：平均值［最小–最大］± 标准差",
    methodTitle: "口径与完整性", queueDefinitionTitle: "Queue 定义", queueDefinition: "每秒对所有受监控 lane 的 queue 求和，单位为车辆数。",
    coverageTitle: "数据覆盖", coverage: "4 controllers × 5 methods × 23 scenarios × 10 runs，共 {count} 条 rollout。",
    provisionalTitle: "结果状态", gridProvisional: "这是完整 Grid 快照，但仍是临时结果；正式 Phase F 需等待全部 9,200 条结果。",
    monacoProvisional: "这是完整 Monaco 快照；与 Grid 合并后可进行双路网正式分析。",
    monacoImportTitle: "Monaco 导入接口", monacoImport: "Monaco 完整后运行只读导入命令，页面会自动启用 Monaco 数据源。",
    metricGlossary: "查看指标说明", footer: "本地只读可视化 · 数据来自 {source}", tableScroll: "可横向滚动的指标比较表",
    method: "方法", mean: "平均值", min: "最小值", max: "最大值", time: "仿真时间（秒）", vehicles: "车辆数", at: "时间", nRuns: "n=10",
    selection: "当前：{network} · {controller} · {split} · {scenario}", family: "场景族",
    csvReady: "当前 CSV 已下载", imageReady: "图像已下载", chartOverview: "所选方法的平均排队折线图", chartBand: "{method} 的 10 次运行排队范围图",
    downloadError: "下载失败：当前场景尚未加载。", generatedMissing: "未找到已生成的数据，请先运行数据导出脚本。",
    countFailed: "数据数量校验失败。", seriesRows: "时序 CSV 行数不是 3,600。", seriesLoad: "当前场景的时序 CSV 无法读取。",
    selectOne: "总览至少保留一种方法。", zoomIn: "放大", zoomOut: "缩小", resetView: "重置视窗", downloadPng: "下载 PNG",
    smoothed: "平滑后", raw: "原始", axisMode: "纵轴比例", axisZero: "自动范围（从 0 开始）", axisDetail: "细节模式（贴合可见数据）", axisHelp: "总览按所选均值线定标；双窗口共用所选阴影范围。缩放后自动更新。细节模式纵轴可能不从 0 开始。", panelLeft: "窗口 A", panelRight: "窗口 B", selectController: "至少选择一个算法。", selectCurve: "每个窗口至少保留一条曲线。", curve: "算法 × 方法"
  },
  en: {
    skip: "Skip to main content", eyebrow: "CBWCE evaluation workbook", filtersTitle: "Result filters",
    loading: "Loading data…", loadingSelection: "Loading selected scenario…", ready: "Complete dataset · {count} / {expected}",
    error: "Unable to load data", network: "Network", controller: "Controller", split: "Evaluate datasets", scenario: "Scenario",
    seen: "Seen (in-distribution)", test: "Test (frozen test set)", external: "External (supplementary demand)", evaluationSet: "Evaluation set", waitingImport: "awaiting import", provenanceDownload: "Source and route audit JSON", pairedDownload: "Paired comparison CSV",
    gridSubtitle: "Read-only snapshot of 4,600 completed rollouts; Monaco can be imported when complete.",
    monacoSubtitle: "Read-only snapshot of 4,600 completed Monaco rollouts.",
    overviewTitle: "Multi-controller mean queue overview", overviewDescription: "Choose the methods to compare; each line is the second-by-second mean network queue across 10 runs.",
    downloadCurrent: "Download filtered CSV", downloadSummary: "Download summary CSV", smoothing: "Curve smoothing strength",
    smoothingHelp: "0 shows raw curves; higher values apply stronger smoothing.", visibleMethods: "Methods shown in overview",
    chartHint: "Use the wheel or buttons to zoom and drag curves horizontally; every chart shares the time window.",
    bandTitle: "Independent side-by-side comparisons", bandDescription: "Choose controller × method curves independently in each panel. Bands show 10-run min–max; lines show means. Color identifies method and dash identifies controller. Network, scenario, axes and smoothing are shared.",
    tableTitle: "All-method metrics by selected controller", bestValue: "Best mean within controller (ties included)", tableScope: "All five methods are shown for each selected controller. Best values are compared within each controller, independently of overview method selections.", tenRunStats: "Cell: mean [minimum–maximum] ± standard deviation",
    methodTitle: "Definitions and completeness", queueDefinitionTitle: "Queue definition", queueDefinition: "At each second, queue is summed across every monitored lane and reported in vehicles.",
    coverageTitle: "Data coverage", coverage: "4 controllers × 5 methods × 23 scenarios × 10 runs: {count} rollouts in total.",
    provisionalTitle: "Result status", gridProvisional: "This is the complete Grid snapshot, but remains preliminary; final Phase F requires all 9,200 results.",
    monacoProvisional: "This is the complete Monaco snapshot; combine it with Grid for the final two-network analysis.",
    monacoImportTitle: "Monaco import interface", monacoImport: "When Monaco is complete, run the read-only import command and the Monaco data source will be enabled automatically.",
    metricGlossary: "View metric glossary", footer: "Local read-only visualization · data from {source}", tableScroll: "Horizontally scrollable metric comparison table",
    method: "Method", mean: "Mean", min: "Minimum", max: "Maximum", time: "simulation time (s)", vehicles: "vehicles", at: "Time", nRuns: "n=10",
    selection: "Current: {network} · {controller} · {split} · {scenario}", family: "Scenario family",
    csvReady: "Filtered CSV downloaded", imageReady: "Chart image downloaded", chartOverview: "Mean queue chart for selected methods", chartBand: "10-run queue range chart for {method}",
    downloadError: "Download failed: the current scenario has not loaded.", generatedMissing: "Generated data was not found; run the data export script first.",
    countFailed: "Dataset count validation failed.", seriesRows: "A time-series CSV does not contain 3,600 rows.", seriesLoad: "The time-series CSV for this scenario could not be loaded.",
    selectOne: "Keep at least one method in the overview.", zoomIn: "Zoom in", zoomOut: "Zoom out", resetView: "Reset view", downloadPng: "Download PNG",
    smoothed: "smoothed", raw: "raw", axisMode: "Y-axis scale", axisZero: "Auto range (zero baseline)", axisDetail: "Detail (fit visible data)", axisHelp: "Overview fits selected means; panels share selected band ranges. Scales update on zoom. Detail mode may use a nonzero baseline.", panelLeft: "Panel A", panelRight: "Panel B", selectController: "Select at least one controller.", selectCurve: "Keep at least one curve per panel.", curve: "Controller × method"
  }
};

const state = {
  lang: "en", registry: null, evaluationSets: null, evaluationSet: "main_v7", network: "grid", networkEntry: null, dataBase: "data",
  catalog: null, summary: [], series: new Map(), controllers: new Set(["ia2c"]), panels: {left: new Set(), right: new Set()}, split: "seen", scenario: null,
  selectedMethods: new Set(), smoothing: 0, yMode: "zero", smoothCache: new Map(), view: {start: 1, end: 3600},
  loadToken: 0, yMax: 1, resizeTimer: null, drag: null, framePending: false
};

const $ = selector => document.querySelector(selector);
const $$ = selector => Array.from(document.querySelectorAll(selector));
function t(key, vars = {}) {
  const value = UI[state.lang][key];
  if (typeof value !== "string") throw new Error(`Missing ${state.lang} translation: ${key}`);
  return value.replace(/\{(\w+)\}/g, (_, name) => String(Object.prototype.hasOwnProperty.call(vars, name) ? vars[name] : ""));
}
function local(obj) { return obj[state.lang]; }
function escapeHtml(value) { return String(value).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"})[c]); }
function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/), headers = lines[0].split(",");
  return lines.slice(1).filter(Boolean).map(line => {
    const values = line.split(","), row = {};
    headers.forEach((header, index) => { row[header] = values[index]; });
    return row;
  });
}
function formatNumber(value, metricKey = "") {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (metricKey === "completion_rate") return `${number.toFixed(2)}%`;
  if (["mean_queue", "peak_queue", "mean_speed", "mean_completed_departure_delay", "mean_completed_time_loss", "mean_completed_travel_time", "mean_completed_waiting_time"].includes(metricKey)) {
    return number.toLocaleString(state.lang === "zh" ? "zh-CN" : "en-US", {maximumFractionDigits: 2});
  }
  return number.toLocaleString(state.lang === "zh" ? "zh-CN" : "en-US", {maximumFractionDigits: 1});
}
function networkLabel(entry) { return local(entry.label); }
function methodDisplay(method) { return `${method.id} · ${method[state.lang]}`; }
function scenarioDisplay(item) { return `${local(item.label)} / ${item.id}`; }
function findScenario() { return state.catalog && state.catalog.scenarios.find(item => item.split === state.split && item.id === state.scenario); }
function familyDisplay(familyId) {
  const family = state.catalog.families.find(item => item.id === familyId);
  return family ? local(family.label) : familyId;
}
function setStatus(key, kind = "", vars = {}) {
  const el = $("#dataset-status");
  el.textContent = t(key, vars); el.className = `status-chip ${kind}`.trim();
}
function showFatal(message) {
  const panel = $("#fatal-error"); panel.textContent = message; panel.hidden = false;
  window.setTimeout(() => { panel.hidden = true; }, 8000);
}

function populateNetworks() {
  if (!state.registry) return;
  if (!state.registry.networks.some(entry => entry.id === state.network)) state.network = state.registry.networks[0].id;
  const select = $("#network-select");
  select.innerHTML = state.evaluationSets.find(entry => entry.id === "main_v7").networks.map(entry => {
    const suffix = entry.available ? "" : ` (${t("waitingImport")})`;
    return `<option value="${escapeHtml(entry.id)}">${escapeHtml(networkLabel(entry) + suffix)}</option>`;
  }).join("");
  select.value = state.network;
}
function datasetLabel() {
  return state.evaluationSet === "main_v7" ? t(state.split) : local(state.registry.label);
}
function selectDataset(id, split = "external") {
  state.registry = state.evaluationSets.find(entry => entry.id === id);
  state.evaluationSet = state.registry.id;
  state.split = split; state.scenario = null;
  if (!state.registry.networks.some(entry => entry.id === state.network && entry.available)) {
    state.network = state.registry.networks.find(entry => entry.available).id;
  }
  populateNetworks(); loadNetwork();
}
function saveUrl() {
  const url = new URL(location.href);
  for (const key of ["evaluationSet", "network", "split", "scenario"]) {
    if (state[key]) url.searchParams.set(key, state[key]); else url.searchParams.delete(key);
  }
  history.replaceState(null, "", url);
}
function populateControllers() {
  $("#controller-select").innerHTML = state.catalog.controllers.map(c => `<label class="controller-option"><input type="checkbox" value="${c.id}" ${state.controllers.has(c.id) ? "checked" : ""}>${escapeHtml(c.label)}</label>`).join("");
  $("#controller-select").querySelectorAll("input").forEach(input => input.addEventListener("change", () => {
    if (input.checked) state.controllers.add(input.value); else state.controllers.delete(input.value);
    if (!state.controllers.size) { state.controllers.add(input.value); input.checked = true; showFatal(t("selectController")); }
    renderSelectionNote(); renderCurveLegend(); renderTable(); drawAll();
  }));
}
function allCurves() {
  return state.catalog.controllers.reduce((all, c, index) => all.concat(state.catalog.methods.map(m => ({id: `${c.id}/${m.id}`, controller: c.id, method: m.id, color: m.color, dash: [[], [9,4], [3,3], [10,3,2,3]][index], zh: `${c.label} · ${m.id} · ${m.zh}`, en: `${c.label} · ${m.id} · ${m.en}`}))), []);
}
function chartCurves(mode, panel) {
  return allCurves().filter(c => mode === "overview" ? state.controllers.has(c.controller) && state.selectedMethods.has(c.method) : state.panels[panel].has(c.id));
}
function curveLabel(c) { return c[state.lang]; }
function renderCurveLegend() {
  $("#curve-legend").innerHTML = chartCurves("overview").map(c => `<span class="legend-item"><svg width="32" height="12" aria-hidden="true"><line x1="0" y1="6" x2="32" y2="6" stroke="${c.color}" stroke-width="2" stroke-dasharray="${c.dash.join(' ')}"/></svg>${escapeHtml(curveLabel(c))}</span>`).join("");
}

function populateSplits() {
  const select = $("#split-select");
  const splits = state.evaluationSet === "main_v7" ? [...new Set(state.catalog.scenarios.map(s => s.split))] : ["seen", "test"];
  const main = splits.map(split => `<option value="${split}">${escapeHtml(t(split))}</option>`);
  const real = state.evaluationSets.filter(entry => entry.id !== "main_v7" && entry.networks.some(network => network.id === state.network && network.available)).map(entry => `<option value="${escapeHtml(entry.id)}">${escapeHtml(local(entry.label))}</option>`);
  select.innerHTML = main.concat(real).join("");
  select.value = state.evaluationSet === "main_v7" ? state.split : state.evaluationSet;
}
function populateScenarios() {
  const select = $("#scenario-select"), items = state.catalog.scenarios.filter(item => item.split === state.split);
  if (!items.some(item => item.id === state.scenario)) state.scenario = items.length ? items[0].id : null;
  select.innerHTML = items.map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(scenarioDisplay(item))}</option>`).join("");
  select.value = state.scenario;
}
function updateDatasetCopy() {
  if (!state.catalog || !state.networkEntry) return;
  const isGrid = state.network === "grid", count = state.catalog.counts.rollouts;
  $("#site-title").textContent = state.evaluationSet === "main_v7" ? local(state.catalog.title) : local(state.registry.label);
  $("#site-subtitle").textContent = t(isGrid ? "gridSubtitle" : "monacoSubtitle");
  $("#coverage-copy").textContent = `4 controllers × 5 methods × ${state.catalog.counts.scenarios} scenarios × 10 runs = ${count.toLocaleString()} rollouts`;
  $("#provisional-copy").textContent = t(isGrid ? "gridProvisional" : "monacoProvisional");
  if (state.evaluationSet === "external_group12") {
    $("#site-subtitle").textContent = state.lang === "zh" ? "独立补充实验；仅一个训练种子的十次评估采样。" : "Separate supplementary campaign; ten evaluation samples from one training seed.";
    $("#provisional-copy").textContent = state.lang === "zh" ? "Grid 上游来源和历史映射尚未验证；当前使用本地稀疏 OD 原生需求。Monaco 等待路线与历史来源核验。" : "Grid upstream provenance and historical mapping are unverified; this tests the native local sparse OD profile. Monaco awaits route and historical provenance validation.";
  }
  if (state.evaluationSet === "monaco_repaired_full14") {
    $("#site-subtitle").textContent = state.lang === "zh" ? "完整14个OD · 六个600秒块 · 20个当前冻结模型 · 200次评估。" : "All 14 OD · six 600s blocks · 20 current frozen models · 200 rollouts.";
    $("#provisional-copy").textContent = state.lang === "zh" ? "模型在原地图训练，通过观测、动作、车道顺序和邻接关系检查后迁移到修复地图；评估期间权重不变。恢复2条遗漏道路，重算几何并调整43个检测器。仅训练种子101，部分历史训练血缘及上游来源尚未核实。" : "Policies trained on the original map transfer to the repaired map after observation, action, lane-order and neighbor checks; weights remain frozen. Two omitted roads restored, geometry recomputed and 43 detectors adjusted. One training seed (101); some historical ancestry and upstream provenance remain unverified.";
  }
  $("#footer-copy").textContent = t("footer", {source: state.catalog.generatedFrom});
  const summary = $("#download-summary");
  summary.href = `${state.dataBase}/metrics_summary.csv`;
  summary.setAttribute("download", state.lang === "zh" ? `${state.evaluationSet}_${state.network}_指标汇总.csv` : `${state.evaluationSet}_${state.network}_metrics_summary.csv`);
  $("#download-provenance").hidden = !state.catalog.provenance;
  $("#download-provenance").href = `${state.dataBase}/catalog.json`;
  $("#download-provenance").setAttribute("download", `${state.network}_provenance.json`);
  $("#download-paired").hidden = !state.catalog.provenance;
  $("#download-paired").href = `${state.dataBase}/queue_comparisons.csv`;
  $("#download-paired").setAttribute("download", `${state.network}_paired_comparisons.csv`);
}
function applyLanguage() {
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en";
  document.title = state.lang === "zh" ? "CBWCE 结果 · Protocol v7" : "CBWCE Results · Protocol v7";
  $$('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
  $$('[data-i18n-aria]').forEach(el => { el.setAttribute("aria-label", t(el.dataset.i18nAria)); });
  populateNetworks();
  if (!state.catalog) {
    if (state.networkEntry && !state.networkEntry.available) showPending();
    return;
  }
  updateDatasetCopy(); populateControllers(); populateSplits(); populateScenarios(); renderSelectionNote(); renderLegend(); renderCurveLegend(); renderMethodCards();
  renderTable(); renderGlossary(); initializeToolbars(); bindChartEvents(); drawAll();
  setReadyStatus();
}
function setReadyStatus() {
  if (!state.catalog) return;
  if (!state.series.size) { setStatus("loadingSelection"); return; }
  setStatus("ready", "ready", {count: state.catalog.counts.rollouts.toLocaleString(), expected: state.networkEntry.expectedRollouts.toLocaleString()});
}
function renderSelectionNote() {
  const scenario = findScenario(); if (!scenario) return;
  saveUrl();
  $("#selection-note").textContent = `${t("selection", {network: networkLabel(state.networkEntry), controller: Array.from(state.controllers).map(c => c.toUpperCase()).join(" + "), split: datasetLabel(), scenario: scenarioDisplay(scenario)})} · ${t("family")}: ${familyDisplay(scenario.family)}`;
}
function renderLegend() {
  $("#legend").innerHTML = state.catalog.methods.map(method => {
    const checked = state.selectedMethods.has(method.id);
    return `<label class="legend-item ${checked ? "" : "is-off"}"><input type="checkbox" value="${method.id}" ${checked ? "checked" : ""}><i class="legend-line" style="border-color:${method.color}"></i><span>${escapeHtml(methodDisplay(method))}</span></label>`;
  }).join("");
  $("#legend").querySelectorAll("input").forEach(input => input.addEventListener("change", event => {
    const id = event.target.value;
    if (event.target.checked) state.selectedMethods.add(id); else state.selectedMethods.delete(id);
    if (!state.selectedMethods.size) {
      state.selectedMethods.add(id); event.target.checked = true; showFatal(t("selectOne"));
    }
    renderLegend(); renderCurveLegend(); renderTable(); drawAll();
  }));
}
function toolbarHtml(chartId) {
  return `<button type="button" class="chart-tool" data-action="zoom-in" data-chart="${chartId}" title="${escapeHtml(t("zoomIn"))}" aria-label="${escapeHtml(t("zoomIn"))}">＋</button><button type="button" class="chart-tool" data-action="zoom-out" data-chart="${chartId}" title="${escapeHtml(t("zoomOut"))}" aria-label="${escapeHtml(t("zoomOut"))}">−</button><button type="button" class="chart-tool" data-action="reset" data-chart="${chartId}" title="${escapeHtml(t("resetView"))}" aria-label="${escapeHtml(t("resetView"))}">↺</button><button type="button" class="chart-tool" data-action="download" data-chart="${chartId}" title="${escapeHtml(t("downloadPng"))}" aria-label="${escapeHtml(t("downloadPng"))}">PNG</button>`;
}
function renderMethodCards() {
  if (!state.catalog) return;
  $("#method-grid").innerHTML = ["left", "right"].map(panel => `<article class="method-card"><header><h3>${t(panel === "left" ? "panelLeft" : "panelRight")}</h3><span class="run-badge">${t("nRuns")}</span></header>
    <div class="panel-picker" role="group" aria-label="${t(panel === 'left' ? 'panelLeft' : 'panelRight')}">${allCurves().map(c => `<label><input type="checkbox" data-panel="${panel}" value="${c.id}" ${state.panels[panel].has(c.id) ? "checked" : ""}><svg width="30" height="12" aria-hidden="true"><line x1="0" y1="6" x2="30" y2="6" stroke="${c.color}" stroke-width="2" stroke-dasharray="${c.dash.join(' ')}"/></svg>${escapeHtml(curveLabel(c))}</label>`).join("")}</div>
    <div class="chart-shell"><div class="chart-toolbar">${toolbarHtml(panel)}</div><canvas id="chart-${panel}" role="img" aria-label="${t(panel === 'left' ? 'panelLeft' : 'panelRight')}"></canvas><div class="tooltip" id="tooltip-${panel}" hidden></div></div></article>`).join("");
  $$('.panel-picker input').forEach(input => input.addEventListener("change", () => {
    const chosen = state.panels[input.dataset.panel];
    if (input.checked) chosen.add(input.value); else chosen.delete(input.value);
    if (!chosen.size) { chosen.add(input.value); input.checked = true; showFatal(t("selectCurve")); }
    hideTooltips(); drawAll();
  }));
}

function initializeToolbars() {
  const overview = $('[data-chart-toolbar="overview"]'); if (overview) overview.innerHTML = toolbarHtml("overview");
  $$('.chart-tool').forEach(button => button.addEventListener("click", () => {
    const action = button.dataset.action;
    if (action === "zoom-in") zoomView(0.7, 0.5);
    else if (action === "zoom-out") zoomView(1.4, 0.5);
    else if (action === "reset") resetView();
    else if (action === "download") downloadChart(button.dataset.chart);
  }));
}

function seriesUrl(curve) { return `${state.dataBase}/series/${encodeURIComponent(curve.controller)}/${encodeURIComponent(curve.method)}/${encodeURIComponent(state.split)}/${encodeURIComponent(state.scenario)}.csv`; }
function showPending() {
  $("#selection-note").textContent = local(state.networkEntry.status);
  $("#dataset-status").textContent = local(state.networkEntry.status);
  $("#site-title").textContent = `${local(state.registry.label)} · ${networkLabel(state.networkEntry)}`;
  $("#site-subtitle").textContent = local(state.networkEntry.status);
  $("#footer-copy").textContent = local(state.networkEntry.status);
}
async function loadNetwork() {
  const token = ++state.loadToken;
  state.series = new Map(); state.catalog = null; state.summary = []; setStatus("loading");
  state.networkEntry = state.registry.networks.find(entry => entry.id === state.network);
  if (!state.networkEntry) { showFatal(t("generatedMissing")); return; }
  saveUrl();
  $("#fatal-error").hidden = true;
  document.querySelectorAll("main > section:not(:first-child)").forEach(section => { section.hidden = !state.networkEntry.available; });
  if (!state.networkEntry.available) {
    state.split = null; state.scenario = null; saveUrl(); showPending();
    $("#download-provenance").hidden = true; $("#download-paired").hidden = true;
    $("#scenario-select").innerHTML = ""; $("#split-select").innerHTML = ""; $("#controller-select").innerHTML = "";
    return;
  }
  state.dataBase = state.networkEntry.base;
  try {
    const responses = await Promise.all([fetch(`${state.dataBase}/catalog.json`, {cache: "no-store"}), fetch(`${state.dataBase}/metrics_summary.csv`, {cache: "no-store"})]);
    if (!responses[0].ok || !responses[1].ok) throw new Error(t("generatedMissing"));
    const catalog = await responses[0].json(), summary = parseCsv(await responses[1].text());
    if (token !== state.loadToken) return;
    if (catalog.counts.rollouts !== state.networkEntry.expectedRollouts || summary.length !== (state.networkEntry.expectedGroups || 460)) throw new Error(t("countFailed"));
    state.catalog = catalog; state.summary = summary; state.controllers = new Set([catalog.controllers[0].id]); state.panels.left = new Set(catalog.methods.map(m => `${catalog.controllers[0].id}/${m.id}`)); state.panels.right = new Set(catalog.methods.map(m => `${catalog.controllers[1].id}/${m.id}`));
    if (!catalog.scenarios.some(s => s.split === state.split)) state.split = catalog.scenarios[0].split;
    if (!catalog.scenarios.some(s => s.id === state.scenario && s.split === state.split)) state.scenario = null;
    state.selectedMethods = new Set(catalog.methods.map(method => method.id));
    populateControllers(); populateSplits(); populateScenarios(); applyLanguage();
    await loadSelection();
  } catch (error) {
    if (token !== state.loadToken) return;
    setStatus("error", "error"); showFatal(`${t("error")}: ${error.message}`);
  }
}
async function loadSelection() {
  const token = ++state.loadToken; state.series = new Map(); state.smoothCache.clear(); resetView(false); setStatus("loadingSelection");
  try {
    const pairs = await Promise.all(allCurves().map(async method => {
      const response = await fetch(seriesUrl(method), {cache: "no-store"});
      if (!response.ok) throw new Error(t("seriesLoad"));
      const rows = parseCsv(await response.text());
      if (rows.length !== 3600) throw new Error(t("seriesRows"));
      return [method.id, rows.map(row => ({time: Number(row.time), min: Number(row.queue_min), mean: Number(row.queue_mean), max: Number(row.queue_max), n: Number(row.n)}))];
    }));
    if (token !== state.loadToken) return;
    state.series = new Map(pairs); state.yMax = pairs.reduce((max, pair) => pair[1].reduce((value, row) => Math.max(value, row.max), max), 1) * 1.04;
    renderSelectionNote(); renderCurveLegend(); renderMethodCards(); initializeToolbars(); bindChartEvents(); renderTable(); drawAll(); setReadyStatus();
  } catch (error) {
    if (token !== state.loadToken) return;
    setStatus("error", "error"); showFatal(`${t("error")}: ${error.message}`);
  }
}
function getDisplayRows(methodId) {
  const raw = state.series.get(methodId); if (!raw || state.smoothing <= 0) return raw;
  const key = `${methodId}:${state.smoothing.toFixed(2)}`; if (state.smoothCache.has(key)) return state.smoothCache.get(key);
  const weightPrevious = state.smoothing, out = [], previous = {min: raw[0].min, mean: raw[0].mean, max: raw[0].max};
  raw.forEach((row, index) => {
    if (index) ["min", "mean", "max"].forEach(field => { previous[field] = weightPrevious * previous[field] + (1 - weightPrevious) * row[field]; });
    out.push({time: row.time, min: previous.min, mean: previous.mean, max: previous.max, n: row.n});
  });
  state.smoothCache.set(key, out); return out;
}

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect(), ratio = Math.max(1, window.devicePixelRatio || 1), width = Math.max(280, rect.width), height = Math.max(220, rect.height);
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {ctx, width, height, pad: {left: width < 430 ? 58 : 70, right: 24, top: 54, bottom: 54}};
}
function tickStep(span, count) {
  const raw = Math.max(span / count, 0.001), power = Math.pow(10, Math.floor(Math.log10(raw)));
  const fraction = raw / power;
  return ([1, 2, 2.5, 5, 10].find(n => n >= fraction) || 10) * power;
}
function chartYRange(mode) {
  const curves = mode === "overview" ? chartCurves("overview") : chartCurves("band", "left").concat(chartCurves("band", "right"));
  let low = Infinity, high = -Infinity;
  curves.forEach(c => visibleRows(getDisplayRows(c.id)).forEach(row => {
    low = Math.min(low, mode === "overview" ? row.mean : row.min);
    high = Math.max(high, mode === "overview" ? row.mean : row.max);
  }));
  if (!Number.isFinite(low)) return {min: 0, max: 1, step: 0.2};
  const padding = Math.max(1, (high - low) * 0.06);
  low = state.yMode === "zero" ? 0 : Math.max(0, low - padding);
  high = Math.max(low + 1, high + padding);
  const step = tickStep(high - low, 6);
  return {min: Math.floor(low / step) * step, max: Math.ceil(high / step) * step, step};
}
function drawAxes(ctx, width, height, pad, range) {
  const plotWidth = width - pad.left - pad.right, plotHeight = height - pad.top - pad.bottom;
  const start = state.view.start, end = state.view.end;
  ctx.clearRect(0, 0, width, height); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#e3e9ed"; ctx.lineWidth = 1; ctx.fillStyle = "#526672"; ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let value = range.min; value <= range.max + range.step * 0.01; value += range.step) {
    const y = pad.top + plotHeight * (range.max - value) / (range.max - range.min);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    ctx.fillText(Number(value.toFixed(3)).toLocaleString(), pad.left - 9, y);
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const xStep = Math.max(1, tickStep(end - start, Math.max(2, Math.floor(plotWidth / 110))));
  for (let value = Math.ceil(start / xStep) * xStep; value <= end; value += xStep) {
    const x = pad.left + plotWidth * (value - start) / (end - start);
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, height - pad.bottom); ctx.stroke();
    ctx.fillText(String(Math.round(value)), x, height - pad.bottom + 9);
  }
  ctx.strokeStyle = "#82939d"; ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, height - pad.bottom); ctx.lineTo(width - pad.right, height - pad.bottom); ctx.stroke();
  ctx.fillStyle = "#354e5d"; ctx.fillText(t("time"), pad.left + plotWidth / 2, height - 21);
  ctx.textAlign = "left"; ctx.fillText(t("vehicles"), pad.left, 19);
  return {plotWidth, plotHeight};
}
function visibleRows(rows) { return rows.filter(row => row.time >= state.view.start - 1 && row.time <= state.view.end + 1); }
function drawSeriesLine(ctx, rows, color, x, y, key, lineWidth) {
  ctx.beginPath(); rows.forEach((row, index) => { const px = x(row.time), py = y(row[key]); if (index) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
  ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.stroke();
}
function drawChart(canvas, mode, method) {
  if (!canvas || !state.series.size) return;
  const built = setupCanvas(canvas), ctx = built.ctx, width = built.width, height = built.height, pad = built.pad, range = chartYRange(mode), yMax = range.max, axes = drawAxes(ctx, width, height, pad, range);
  const x = time => pad.left + axes.plotWidth * (time - state.view.start) / (state.view.end - state.view.start);
  const y = value => pad.top + axes.plotHeight * (1 - (value - range.min) / (range.max - range.min));
  ctx.save(); ctx.beginPath(); ctx.rect(pad.left, pad.top, axes.plotWidth, axes.plotHeight); ctx.clip();
  chartCurves(mode, method).forEach(curve => {
    const rows = visibleRows(getDisplayRows(curve.id));
    if (mode !== "overview") {
      ctx.beginPath(); rows.forEach((row, i) => { if (i) ctx.lineTo(x(row.time), y(row.max)); else ctx.moveTo(x(row.time), y(row.max)); });
      rows.slice().reverse().forEach(row => ctx.lineTo(x(row.time), y(row.min)));
      ctx.closePath(); ctx.fillStyle = `${curve.color}18`; ctx.fill();
    }
    ctx.setLineDash(curve.dash); drawSeriesLine(ctx, rows, curve.color, x, y, "mean", 1.9); ctx.setLineDash([]);
  });
  ctx.restore(); canvas._plot = {pad, plotWidth: axes.plotWidth, plotHeight: axes.plotHeight, width, height, yMax, yMin: range.min, mode, method};
}
function drawOverview() { if (state.series.size) drawChart($("#overview-canvas"), "overview", null); }
function drawAll() {
  if (!state.series.size) return; drawOverview();
  ["left", "right"].forEach(panel => drawChart($(`#chart-${panel}`), "band", panel));
}
function scheduleDraw() { if (state.framePending) return; state.framePending = true; requestAnimationFrame(() => { state.framePending = false; drawAll(); }); }

function normalizeView(start, end) {
  const minSpan = 60, maxStart = 3600;
  let span = Math.max(minSpan, Math.min(3599, end - start));
  let nextStart = start, nextEnd = start + span;
  if (nextStart < 1) { nextStart = 1; nextEnd = 1 + span; }
  if (nextEnd > maxStart) { nextEnd = maxStart; nextStart = maxStart - span; }
  state.view = {start: nextStart, end: nextEnd};
}
function zoomView(factor, anchorFraction) {
  const span = state.view.end - state.view.start, anchor = state.view.start + span * anchorFraction, nextSpan = span * factor;
  normalizeView(anchor - nextSpan * anchorFraction, anchor + nextSpan * (1 - anchorFraction)); scheduleDraw();
}
function resetView(redraw = true) { state.view = {start: 1, end: 3600}; if (redraw) scheduleDraw(); }
function hideTooltips() { $$('.tooltip').forEach(item => { item.hidden = true; }); }
function tooltipHtml(index, mode, panel) {
  const suffix = state.smoothing > 0 ? t("smoothed") : t("raw");
  return `<strong>${t("at")}: ${index + 1} s · ${suffix}</strong>` + chartCurves(mode, panel).map(c => {
    const row = getDisplayRows(c.id)[index];
    return `<div><span style="color:${c.color}">●</span> ${escapeHtml(curveLabel(c))}<br>${t("mean")}: ${formatNumber(row.mean)} ${t("vehicles")}${mode === "overview" ? "" : ` · ${t("min")}: ${formatNumber(row.min)} · ${t("max")}: ${formatNumber(row.max)}`}</div>`;
  }).join("");
}

function positionTooltip(event, canvas, tooltip, mode, method) {
  if (!state.series.size || !canvas._plot || state.drag) return;
  const rect = canvas.getBoundingClientRect(), px = Math.max(canvas._plot.pad.left, Math.min(canvas._plot.width - canvas._plot.pad.right, event.clientX - rect.left));
  const fraction = (px - canvas._plot.pad.left) / canvas._plot.plotWidth, time = state.view.start + fraction * (state.view.end - state.view.start), index = Math.max(0, Math.min(3599, Math.round(time) - 1));
  tooltip.innerHTML = tooltipHtml(index, mode, method); tooltip.hidden = false;
  const shell = canvas.parentElement, tooltipWidth = tooltip.offsetWidth;
  tooltip.style.left = `${Math.max(5, Math.min(shell.clientWidth - tooltipWidth - 5, px + 10))}px`;
  tooltip.style.top = `${Math.max(42, Math.min(shell.clientHeight - tooltip.offsetHeight - 5, event.clientY - rect.top - 15))}px`;
}
function bindCanvas(canvas, tooltip, mode, method) {
  if (!canvas || canvas.dataset.interactionBound === "true") return; canvas.dataset.interactionBound = "true";
  canvas.addEventListener("wheel", event => {
    if (!canvas._plot) return; event.preventDefault();
    const rect = canvas.getBoundingClientRect(), px = event.clientX - rect.left, anchor = Math.max(0, Math.min(1, (px - canvas._plot.pad.left) / canvas._plot.plotWidth));
    zoomView(event.deltaY < 0 ? 0.8 : 1.25, anchor); hideTooltips();
  }, {passive: false});
  canvas.addEventListener("pointerdown", event => {
    if (!canvas._plot) return; canvas.setPointerCapture(event.pointerId); canvas.classList.add("dragging"); hideTooltips();
    state.drag = {pointerId: event.pointerId, x: event.clientX, start: state.view.start, end: state.view.end, plotWidth: canvas._plot.plotWidth};
  });
  canvas.addEventListener("pointermove", event => {
    if (state.drag && state.drag.pointerId === event.pointerId) {
      const span = state.drag.end - state.drag.start, shift = -(event.clientX - state.drag.x) / state.drag.plotWidth * span;
      normalizeView(state.drag.start + shift, state.drag.end + shift); scheduleDraw();
    } else positionTooltip(event, canvas, tooltip, mode, method);
  });
  const stop = event => { if (state.drag && state.drag.pointerId === event.pointerId) { state.drag = null; canvas.classList.remove("dragging"); } };
  canvas.addEventListener("pointerup", stop); canvas.addEventListener("pointercancel", stop); canvas.addEventListener("pointerleave", () => { if (!state.drag) tooltip.hidden = true; });
}
function bindChartEvents() {
  bindCanvas($("#overview-canvas"), $("#overview-tooltip"), "overview", null);
  if (state.catalog) ["left", "right"].forEach(panel => bindCanvas($(`#chart-${panel}`), $(`#tooltip-${panel}`), "band", panel));
}
function chartExportCanvas(canvas, chartId) {
  const curves = chartCurves(chartId === "overview" ? "overview" : "band", chartId);
  const output = document.createElement("canvas"), ctx = output.getContext("2d");
  const font = "14px system-ui, sans-serif", margin = 24, gap = 24, rowHeight = 30;
  ctx.font = font;
  const entries = curves.map(curve => ({curve, label: curveLabel(curve), width: Math.ceil(ctx.measureText(curveLabel(curve)).width) + 48}));
  const width = Math.ceil(Math.max(canvas._plot.width, ...entries.map(entry => entry.width + margin * 2)));
  const rows = []; let row = [], rowWidth = 0;
  entries.forEach(entry => {
    if (row.length && rowWidth + gap + entry.width > width - margin * 2) { rows.push({entries: row, width: rowWidth}); row = []; rowWidth = 0; }
    rowWidth += (row.length ? gap : 0) + entry.width; row.push(entry);
  });
  if (row.length) rows.push({entries: row, width: rowWidth});
  const chartHeight = Math.ceil(canvas._plot.height), height = chartHeight + margin * 2 + rows.length * rowHeight;
  const ratio = Math.max(2, window.devicePixelRatio || 1);
  output.width = Math.ceil(width * ratio); output.height = Math.ceil(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, width, height);
  ctx.drawImage(canvas, (width - canvas._plot.width) / 2, 0, canvas._plot.width, canvas._plot.height);
  ctx.font = font; ctx.textAlign = "left"; ctx.textBaseline = "middle";
  rows.forEach((legendRow, index) => {
    let x = (width - legendRow.width) / 2;
    const y = chartHeight + margin + rowHeight * (index + 0.5);
    legendRow.entries.forEach(entry => {
      ctx.strokeStyle = entry.curve.color; ctx.lineWidth = 2.2; ctx.setLineDash(entry.curve.dash);
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 32, y); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "#354e5d"; ctx.fillText(entry.label, x + 42, y);
      x += entry.width + gap;
    });
  });
  return output;
}
function downloadChart(chartId) {
  const canvas = chartId === "overview" ? $("#overview-canvas") : $(`#chart-${chartId}`);
  if (!canvas || !state.series.size) { showFatal(t("downloadError")); return; }
  chartExportCanvas(canvas, chartId).toBlob(blob => {
    if (!blob) { showFatal(t("downloadError")); return; }
    const url = URL.createObjectURL(blob), anchor = document.createElement("a"), view = `${Math.round(state.view.start)}-${Math.round(state.view.end)}s`;
    anchor.href = url; anchor.download = `${state.network}_${Array.from(state.controllers).join("+")}_${state.split}_${state.scenario}_${chartId}_${view}.png`;
    document.body.appendChild(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(url);
    setStatus("imageReady", "ready"); window.setTimeout(setReadyStatus, 1600);
  }, "image/png");
}

function selectedSummary() {
  return state.summary.filter(row => state.controllers.has(row.controller) && row.split === state.split && row.scenario === state.scenario);
}
function renderTable() {
  if (!state.catalog) return;
  const rows = selectedSummary();
  const header = `<thead><tr><th>${escapeHtml(t("curve"))}</th>${state.catalog.metrics.map(metric => `<th title="${escapeHtml(local(metric.description))}">${escapeHtml(local(metric.label))}<span class="unit">${escapeHtml(local(metric.unit))}</span><span class="direction">${escapeHtml(local(metric.directionLabel))}</span></th>`).join("")}</tr></thead>`;
  const groups = state.catalog.controllers.filter(c => state.controllers.has(c.id)).map(controller => {
    const controllerRows = rows.filter(row => row.controller === controller.id);
    const best = {};
    state.catalog.metrics.forEach(metric => {
      if (metric.direction === "check") return;
      const values = controllerRows.map(row => row[`${metric.key}_mean`] === "" ? NaN : Number(row[`${metric.key}_mean`])).filter(Number.isFinite);
      best[metric.key] = metric.direction === "lower" ? Math.min.apply(null, values) : Math.max.apply(null, values);
    });
    const body = state.catalog.methods.map(method => {
      const row = controllerRows.find(item => item.method === method.id);
      if (!row) return "";
      const cells = state.catalog.metrics.map(metric => {
        const mean = row[`${metric.key}_mean`] === "" ? NaN : Number(row[`${metric.key}_mean`]);
        const isBest = metric.direction !== "check" && Number.isFinite(mean) && Math.abs(mean - best[metric.key]) < 1e-9;
        return `<td data-metric="${metric.key}" class="${isBest ? "best" : ""}"${isBest ? ` title="${escapeHtml(controller.label + ' · ' + t('bestValue'))}"` : ""}>${isBest ? '<span aria-hidden="true">★ </span>' : ""}${formatNumber(mean, metric.key)}<span class="range">[${formatNumber(row[`${metric.key}_min`], metric.key)}–${formatNumber(row[`${metric.key}_max`], metric.key)}] ± ${formatNumber(row[`${metric.key}_sd`], metric.key)}</span></td>`;
      }).join("");
      return `<tr data-controller="${controller.id}" data-method="${method.id}"><td><strong style="color:${method.color}">${escapeHtml(controller.label)} · ${escapeHtml(method.id)}</strong><span class="range">${escapeHtml(method[state.lang])}</span></td>${cells}</tr>`;
    }).join("");
    return `<tbody data-controller-group="${controller.id}"><tr class="controller-group"><th scope="rowgroup" colspan="${state.catalog.metrics.length + 1}">${escapeHtml(controller.label)}</th></tr>${body}</tbody>`;
  }).join("");
  $("#metrics-table").innerHTML = header + groups;
}
function renderGlossary() {
  if (!state.catalog) return;
  $("#metric-glossary").innerHTML = state.catalog.metrics.map(metric => `<article><h3><span>${escapeHtml(local(metric.label))}</span><small>${escapeHtml(local(metric.unit))} · ${escapeHtml(local(metric.directionLabel))}</small></h3><p>${escapeHtml(local(metric.description))}</p></article>`).join("");
}
function downloadCurrentCsv() {
  if (!state.series.size) { showFatal(t("downloadError")); return; }
  const methods = chartCurves("overview").map(item => item.id), header = ["time"].concat(methods.reduce((all, method) => all.concat([`${method}_queue_min`, `${method}_queue_mean`, `${method}_queue_max`]), []));
  const first = state.series.get(methods[0]), lines = [header.join(",")];
  first.forEach((row, index) => { const values = [row.time]; methods.forEach(method => { const item = state.series.get(method)[index]; values.push(item.min.toFixed(6), item.mean.toFixed(6), item.max.toFixed(6)); }); lines.push(values.join(",")); });
  const blob = new Blob([lines.join("\n") + "\n"], {type: "text/csv;charset=utf-8"}), url = URL.createObjectURL(blob), anchor = document.createElement("a");
  anchor.href = url; anchor.download = `${state.lang === "zh" ? "当前筛选" : "filtered"}_${state.network}_${Array.from(state.controllers).join("+")}_${state.split}_${state.scenario}.csv`;
  document.body.appendChild(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(url); setStatus("csvReady", "ready"); window.setTimeout(setReadyStatus, 1600);
}

async function init() {
  applyLanguage();
  try {
    const response = await fetch("data/evaluation_sets.json", {cache: "no-store"}); if (!response.ok) throw new Error(t("generatedMissing"));
    state.evaluationSets = (await response.json()).evaluationSets.filter(entry => entry.id !== "monaco_legacy_replay");
    const params = new URLSearchParams(location.search);
    state.evaluationSet = params.get("evaluationSet") || "main_v7";
    state.registry = state.evaluationSets.find(e => e.id === state.evaluationSet) || state.evaluationSets[0];
    state.evaluationSet = state.registry.id;
    state.network = params.get("network") || state.registry.networks[0].id;
    state.split = params.get("split") || "seen"; state.scenario = params.get("scenario");
    populateNetworks(); await loadNetwork();
  } catch (error) { setStatus("error", "error"); showFatal(`${t("error")}: ${error.message}`); }
}

$("#network-select").addEventListener("change", event => {
  state.network = event.target.value;
  if (!state.registry.networks.some(entry => entry.id === state.network && entry.available)) {
    selectDataset("main_v7", "seen");
  } else { populateNetworks(); loadNetwork(); }
});
$("#split-select").addEventListener("change", event => {
  const value = event.target.value;
  if (state.evaluationSets.some(entry => entry.id === value)) { selectDataset(value); return; }
  if (state.evaluationSet !== "main_v7") { selectDataset("main_v7", value); return; }
  state.split = value; state.scenario = null; populateScenarios(); renderSelectionNote(); loadSelection();
});
$("#scenario-select").addEventListener("change", event => { state.scenario = event.target.value; renderSelectionNote(); loadSelection(); });
$("#smoothing-range").addEventListener("input", event => { state.smoothing = Number(event.target.value); $("#smoothing-value").value = state.smoothing.toFixed(2); state.smoothCache.clear(); scheduleDraw(); });
$("#axis-mode").addEventListener("change", event => { state.yMode = event.target.value; hideTooltips(); drawAll(); });
$("#download-current").addEventListener("click", downloadCurrentCsv);
window.addEventListener("resize", () => { clearTimeout(state.resizeTimer); state.resizeTimer = setTimeout(drawAll, 120); });
init();
