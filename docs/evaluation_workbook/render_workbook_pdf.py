#!/usr/bin/env python3
"""Render the bilingual workbook PDF when the local ctex toolchain is unavailable.

The detailed editable source remains the ctexbook tree. This renderer consumes the same
generated registry and tables and produces the delivery PDF with embedded CJK fonts.
"""
from __future__ import print_function

import html
import json
import os
import textwrap
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, KeepTogether, NextPageTemplate,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SELECTION = ROOT / "runs_eval/revised/selections/final_evaluation_seed101.json"
OUTPUT = ROOT / "output/pdf/cb_wce_evaluation_and_data_validation_workbook_zh_en.pdf"
TEMP = ROOT / "tmp/pdfs/cb_wce_evaluation_workbook/cb_wce_evaluation_and_data_validation_workbook_zh_en.pdf"
CJK_TTF = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
LATIN = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LATIN_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

BLUE = colors.HexColor("#174A7E")
CYAN = colors.HexColor("#0E7490")
INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#52606D")
PALE = colors.HexColor("#EAF2F8")
WARN = colors.HexColor("#B45309")
LIGHT = colors.HexColor("#F6F8FA")


def register_fonts():
    # ReportLab 3.6 cannot embed the CFF outlines in the installed Noto TTC files.
    # Droid Sans Fallback is a Unicode TrueType font and embeds cleanly for both
    # Chinese and English. Headings use the same face at larger sizes.
    pdfmetrics.registerFont(TTFont("NotoSerifSC", CJK_TTF))
    pdfmetrics.registerFont(TTFont("NotoSansSC", CJK_TTF))
    pdfmetrics.registerFont(TTFont("NotoSansSC-Bold", CJK_TTF))
    pdfmetrics.registerFont(TTFont("Mono", MONO))
    pdfmetrics.registerFont(TTFont("Latin", LATIN))
    pdfmetrics.registerFont(TTFont("LatinBold", LATIN_BOLD))


def _is_cjk(character):
    code = ord(character)
    return (0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF
            or 0x3000 <= code <= 0x303F)


def esc(value, mixed=True):
    """Escape markup and switch non-CJK runs to embedded DejaVu Sans."""
    value = str(value).replace("–", "-").replace("—", "-")
    if not mixed:
        return html.escape(value).replace("\n", "<br/>")
    lines = []
    for line in value.split("\n"):
        fragments = []
        runs = []
        for character in line:
            kind = "cjk" if _is_cjk(character) else "latin"
            if not runs or runs[-1][0] != kind:
                runs.append([kind, character])
            else:
                runs[-1][1] += character
        for kind, run in runs:
            encoded = html.escape(run)
            fragments.append('<font name="Latin">{}</font>'.format(encoded) if kind == "latin" else encoded)
        lines.append("".join(fragments))
    return "<br/>".join(lines)


def styles():
    base = getSampleStyleSheet()
    result = {
        "title": ParagraphStyle("TitleCJK", parent=base["Title"], fontName="NotoSansSC-Bold", fontSize=25,
                                leading=32, textColor=BLUE, alignment=TA_CENTER, spaceAfter=8 * mm),
        "subtitle": ParagraphStyle("SubtitleCJK", parent=base["Normal"], fontName="NotoSansSC", fontSize=13,
                                   leading=19, textColor=CYAN, alignment=TA_CENTER),
        "part": ParagraphStyle("PartCJK", parent=base["Heading1"], fontName="NotoSansSC-Bold", fontSize=22,
                               leading=28, textColor=BLUE, spaceAfter=10 * mm, alignment=TA_CENTER),
        "h1": ParagraphStyle("H1CJK", parent=base["Heading1"], fontName="NotoSansSC-Bold", fontSize=17,
                             leading=23, textColor=BLUE, spaceBefore=5 * mm, spaceAfter=3 * mm, keepWithNext=True),
        "h2": ParagraphStyle("H2CJK", parent=base["Heading2"], fontName="NotoSansSC-Bold", fontSize=12.5,
                             leading=17, textColor=CYAN, spaceBefore=4 * mm, spaceAfter=2 * mm, keepWithNext=True),
        "body": ParagraphStyle("BodyCJK", parent=base["BodyText"], fontName="NotoSerifSC", fontSize=9.5,
                               leading=15, textColor=INK, wordWrap="CJK", spaceAfter=2.2 * mm),
        "small": ParagraphStyle("SmallCJK", parent=base["BodyText"], fontName="NotoSerifSC", fontSize=7.7,
                                leading=11, textColor=INK, wordWrap="CJK"),
        "note": ParagraphStyle("NoteCJK", parent=base["BodyText"], fontName="NotoSansSC", fontSize=8.5,
                               leading=13, textColor=MUTED, backColor=PALE, borderColor=CYAN, borderWidth=0.5,
                               borderPadding=7, spaceBefore=2 * mm, spaceAfter=3 * mm, wordWrap="CJK"),
        "warn": ParagraphStyle("WarnCJK", parent=base["BodyText"], fontName="NotoSansSC", fontSize=8.5,
                               leading=13, textColor=WARN, backColor=colors.HexColor("#FFF7ED"), borderColor=WARN,
                               borderWidth=0.5, borderPadding=7, spaceBefore=2 * mm, spaceAfter=3 * mm, wordWrap="CJK"),
        "bullet": ParagraphStyle("BulletCJK", parent=base["BodyText"], fontName="NotoSerifSC", fontSize=9,
                                 leading=14, leftIndent=5 * mm, firstLineIndent=-3 * mm, bulletIndent=0,
                                 textColor=INK, wordWrap="CJK", spaceAfter=1.4 * mm),
        "code": ParagraphStyle("Code", parent=base["Code"], fontName="Mono", fontSize=6.3, leading=8.1,
                               leftIndent=2 * mm, rightIndent=2 * mm, backColor=LIGHT, borderPadding=5,
                               wordWrap="LTR", splitLongWords=True, spaceAfter=3 * mm),
        "caption": ParagraphStyle("Caption", parent=base["BodyText"], fontName="NotoSansSC", fontSize=7.5,
                                  leading=10, textColor=MUTED, alignment=TA_CENTER),
    }
    return result


def p(text, style, **kwargs):
    return Paragraph(text, style, **kwargs)


def bullets(items, st):
    return [Paragraph(esc("• " + item), st["bullet"]) for item in items]


def table(data, widths, header=True, font_size=7.4, landscape_mode=False):
    body = []
    for row_no, row in enumerate(data):
        row_style = ParagraphStyle("table-{}-{}".format(id(data), row_no), fontName="NotoSansSC-Bold" if row_no == 0 and header else "NotoSerifSC",
                                   fontSize=font_size, leading=font_size * 1.45, wordWrap="CJK", textColor=colors.white if row_no == 0 and header else INK)
        body.append([Paragraph(esc(cell), row_style) for cell in row])
    result = Table(body, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), BLUE))
    for row_no in range(1 if header else 0, len(body)):
        if row_no % 2 == 0:
            commands.append(("BACKGROUND", (0, row_no), (-1, row_no), LIGHT))
    result.setStyle(TableStyle(commands))
    return result


def code_block(text, st, width=104):
    wrapped = []
    for line in text.splitlines():
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(line, width=width, subsequent_indent="  ", break_long_words=True,
                                     break_on_hyphens=False, replace_whitespace=False) or [""])
    return Paragraph("<br/>".join(esc(line, mixed=False).replace(" ", "&#160;") for line in wrapped), st["code"])


class WorkbookDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
                         topMargin=21 * mm, bottomMargin=18 * mm,
                         title="CB-WCE Evaluation and Data Validation Workbook",
                         author="CB-WCE Project")
        portrait_frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="portrait")
        land_size = landscape(A4)
        land_frame = Frame(15 * mm, 16 * mm, land_size[0] - 30 * mm, land_size[1] - 32 * mm, id="landscape")
        self.addPageTemplates([
            PageTemplate(id="Portrait", pagesize=A4, frames=portrait_frame, onPage=self._header_footer),
            PageTemplate(id="Landscape", pagesize=land_size, frames=land_frame, onPage=self._header_footer),
        ])

    def _header_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Latin", 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, 10 * mm, "CB-WCE Evaluation Workbook - Protocol v7 specification")
        canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 10 * mm, str(doc.page))
        canvas.setStrokeColor(colors.HexColor("#D5DEE7"))
        canvas.line(doc.leftMargin, doc.pagesize[1] - 13 * mm, doc.pagesize[0] - doc.rightMargin, doc.pagesize[1] - 13 * mm)
        canvas.restoreState()


def add_heading(story, text, st, level=1):
    story.append(Paragraph(esc(text), st["h1" if level == 1 else "h2"]))


def add_status_protocol(story, st, language="zh"):
    zh = language == "zh"
    title = "目标、现状与冻结评估协议" if zh else "Objective, Current State, and Frozen Protocol"
    add_heading(story, title, st, 1)
    story.append(p(esc("本手册是正式评估、数据验证、统计分析、网站导出和论文呈现的唯一操作说明。40/40 个最终组合已有完整 checkpoint，但正式评估仍为 0/9,200。peak_validation 仅是 pilot；旧论文数字属于另一协议。" if zh else
                       "This workbook is the operational source of truth for formal evaluation, validation, statistical analysis, site export, and paper presentation. Complete checkpoints exist for all 40 combinations, but the formal campaign remains at 0/9,200. peak_validation is pilot-only and the paper's current numbers belong to an older protocol."), st["body"]))
    story.append(p(esc("重要限制：保留的 baseline continuation 使用 Python 3.10.12 / TensorFlow 2.15.1，而多数其他方法使用 Python 3.6.13 / TensorFlow 1.12.0，且 experiment_env 源码哈希不同。因此性能结论只针对冻结 checkpoint 与 seed 101；wall-clock 仅作描述性比较。" if zh else
                       "Known limitation: retained baseline continuations use Python 3.10.12 / TensorFlow 2.15.1, while most other methods use Python 3.6.13 / TensorFlow 1.12.0, with a different experiment_env source hash. Performance claims are conditional on frozen checkpoints and seed 101; wall-clock comparisons are descriptive."), st["warn"]))
    add_heading(story, "实验矩阵与默认设置" if zh else "Matrix and simulation defaults", st, 2)
    data = [["项目 / Item", "冻结值 / Frozen value"],
            ["Networks", "Grid, Monaco"], ["Controllers", "IA2C, MA2C, IQLL, PPO"],
            ["Methods", "baseline, random_group, domain_randomization, fixed_wce, online_wce"],
            ["Scenarios", "11 seen + 12 held-out test = 23"], ["Paired seeds", "arrival 51001–51010 ↔ SUMO 61001–61010"],
            ["Horizon", "[0,3600), empty start, no warm-up, no drain"], ["Step", "SUMO 1 s; control 5 s = 2 s yellow + 3 s green"],
            ["Base demand", "Grid 3000 veh/h; Monaco 2383.3333 veh/h"],
            ["Training", "1,000,000 parent + 1,320,000 continuation = 2,320,000 steps"],
            ["Campaign", "2 × 4 × 5 × 23 × 10 = 9,200 rollouts"]]
    story.append(table(data, [45 * mm, 123 * mm], font_size=7.8))
    story.append(Spacer(1, 3 * mm))
    story.append(p(esc("总计 9,200 simulated hours、33.12M simulated seconds 和 6.624M control decisions。" if zh else
                       "The campaign represents 9,200 simulated hours, 33.12M simulated seconds, and 6.624M control decisions."), st["note"]))
    add_heading(story, "12 个新 test 场景" if zh else "Twelve new test scenarios", st, 2)
    scen = [["Family", "Scenarios", "Definition"],
            ["OD redistribution", "σ = 0.25, 0.50, 0.75", "Uniform positive-OD support × LogNormal(0,σ), then renormalize to the base total."],
            ["Demand mixture", "mixture 1, 2, 3", "Fixed Dirichlet convex combinations of the eleven named normalized profiles; constant for one hour."],
            ["Temporal switching", "300, 900, 1200 s", "Seeded random profiles from all 11 seen profiles; no adjacent repeat. The 300 s case is the fast-switching stress test."],
            ["Peak demand", "×1.10, ×1.25, ×1.50", "Base [0,1200), multiplied [1200,2400), base [2400,3600)."]]
    story.append(table(scen, [38 * mm, 40 * mm, 90 * mm], font_size=7.4))
    story.append(p(esc("代码已实现 protocol v7 的冻结 seed 随机切换且禁止相邻重复；版本化 v7 artifacts 尚待物化和审计，旧 v6 artifacts 不得进入 release。" if zh else
                       "The code implements protocol-v7 frozen-seed random switching with no adjacent repeat. Versioned v7 artifacts remain to be materialized and audited; protocol-v6 artifacts must not enter the release."), st["warn"]))
    story.append(p(esc("Mixture 是 held-out interpolation，不是严格 OOD；peak 是强度外推；redistribution 是固定 support 内重分配；switch 是已见 profile 的新时间排列。Validation 六场景仅用于调参。" if zh else
                       "Mixtures are held-out interpolations rather than strict OOD samples; peaks extrapolate intensity, redistributions alter mass within a fixed support, and switches are frozen random temporal rearrangements of all eleven seen profiles. Six validation scenarios are tuning-only."), st["body"]))
    story.append(p(esc("Peak 期望一小时车辆量：Grid 3,100 / 3,250 / 3,500；Monaco 约 2,462.78 / 2,581.94 / 2,780.56。" if zh else
                       "Expected one-hour peak counts: Grid 3,100 / 3,250 / 3,500; Monaco approximately 2,462.78 / 2,581.94 / 2,780.56."), st["note"]))
    add_heading(story, "需求物化" if zh else "Demand materialization", st, 2)
    story.extend(bullets([
        "N ~ Poisson(rate × block_duration / 3600) for every OD.",
        "Departure is uniform inside the block plus Normal(0, 2 s) jitter, clipped inside the block and rounded to 0.01 s.",
        "Speed factor is Normal(1, 0.1); nonpositive samples are redrawn.",
        "SUMO-resolved routes and network/profile/scenario/content hashes are frozen in each artifact.",
    ], st))


def add_metrics(story, st, language="zh"):
    zh = language == "zh"
    add_heading(story, "指标、奖励一致性与统计分析" if zh else "Metrics, Reward Alignment, and Statistics", st, 1)
    story.append(p(esc("定义 Q(t)=Σ_l q_l(t)，其中 l 是全局去重的受控进口车道；Grid 150 条、Monaco 116 条。主指标 J_Q=(1/3600)Σ_t Q(t)，单位 vehicles，是 network-total mean queue，lower is better，不是 per-intersection queue。" if zh else
                       "Define Q(t)=Σ_l q_l(t) over globally deduplicated controlled incoming lanes (150 Grid; 116 Monaco). The primary metric is J_Q=(1/3600)Σ_t Q(t), measured in vehicles. It is a network-total mean queue, lower is better, and is not a per-intersection metric."), st["note"]))
    story.append(p(esc("Controller learner reward = clip(raw negative endpoint queue / manifest reward_norm, -2, 2). CB-WCE maximizes each 600-s mean-queue cost and learns from clip(C_k/d_w, -2, 2), with d_w=3000 Grid and 1000 Monaco. Evaluation always uses raw, unscaled J_Q. Clipping fraction must be reported; strict monotonic alignment is claimed only when it is zero." if not zh else
                       "Controller learner reward = clip(raw negative endpoint queue / manifest reward_norm, -2, 2)。CB-WCE 最大化每个 600 秒 mean-queue cost，learner 使用 clip(C_k/d_w,-2,2)，其中 Grid d_w=3000、Monaco d_w=1000。评价始终使用 raw J_Q。必须报告 clipping fraction；只有其为零时才声称严格单调一致。"), st["body"]))
    add_heading(story, "Rollout 指标字典" if zh else "Rollout metric registry", st, 2)
    metrics = [["Metric", "Unit", "Direction", "Definition / denominator"],
               ["mean_total_queue", "vehicles", "lower", "Primary: mean of 3,600 per-second network-total queues."],
               ["integrated / peak / p95 queue", "veh·s / veh", "lower", "Queue sum, maximum, and 95th percentile."],
               ["three 1,200-s windows", "veh / veh·s", "lower", "Pre-peak, peak, post-peak mean and AUC."],
               ["last_600_s_slope", "veh/s", "lower", "Preregistered linear slope; descriptive stability only."],
               ["mean_speed", "m/s", "higher", "Vehicle-second weighted; denominator stored."],
               ["scheduled/inserted/completed", "vehicles", "context", "Artifact, departed, and arrived counts."],
               ["pending/remaining", "vehicles", "lower", "Not departed and still active at 3,600 s."],
               ["completed-trip means", "seconds", "lower", "Travel/wait/time-loss/depart-delay; denominator and unfinished counts mandatory."],
               ["teleports/collisions", "count", "lower", "Reliability; teleport thresholds differ by network."],
               ["wall_seconds", "seconds", "lower", "Monotonic-clock evaluation duration."]]
    story.append(table(metrics, [45 * mm, 25 * mm, 23 * mm, 75 * mm], font_size=6.8))
    add_heading(story, "配对统计" if zh else "Paired analysis", st, 2)
    story.append(p(esc("四个预注册比较：online WCE − baseline / random group / domain randomization / fixed WCE。对每个场景和 10 个 rollout pair 定义 d_r=J_online,r−J_comp,r；queue 的负值更好。报告 mean、sample SD (ddof=1) 和 10,000 次确定性 paired percentile bootstrap 95% CI。" if zh else
                       "Four preregistered effects subtract baseline, random group, domain randomization, and fixed WCE from online WCE. For each scenario and ten rollout pairs define d_r=J_online,r−J_comp,r; negative queue effects favor online WCE. Report mean, sample SD (ddof=1), and a deterministic 10,000-resample paired percentile-bootstrap 95% CI."), st["body"]))
    story.append(p(esc("Family/suite 先在相同 rollout index 内对固定场景等权平均 D_r=(1/|S|)Σ_s d_s,r，再 bootstrap 十个 D_r block。不得把逐秒、车道或跨场景的 120 条记录当 IID。CI 只表示固定 checkpoint 与固定 suite 下的随机性，不是训练 seed uncertainty；一个训练 seed 不能证明算法总体显著优越。" if zh else
                       "For a family or suite, first compute D_r=(1/|S|)Σ_s d_s,r within each shared rollout index, then bootstrap ten D_r blocks. Per-second, lane, or 120 cross-scenario records are not IID replicates. The CI is conditional on fixed checkpoints and the fixed suite, not training-seed uncertainty; one training seed cannot establish algorithm-wide superiority."), st["warn"]))
    story.extend(bullets([
        "Report absolute effects first; percent improvement is supplemental and omitted near a zero denominator.",
        "Report the equal-weight suite mean, worst scenario mean, worst-three average/CVaR, and across-scenario SD.",
        "Queue-worst and speed-worst scenarios are selected separately; method-specific worst cases are not a common paired curve.",
        "If formal multiplicity-adjusted tests are added, use Holm correction.",
    ], st))
    add_heading(story, "Wall-clock 审计" if zh else "Wall-clock audit", st, 2)
    story.append(p(esc("沿 final manifest 的 resume 链递归汇总 parent、offline WCE、continuation、evaluation 和 abandoned/duplicate time。无 result 的 attempt 使用最后 progress wall_seconds 作为 lower bound。分别报告 standalone method cost 与去重的 campaign cost；component seconds 不是完整分解。由于 baseline 环境不一致，wall-clock 仅作描述性比较，不声称 accelerates training。" if zh else
                       "Recursively trace the final manifest's resume chain and aggregate parent, offline-WCE, continuation, evaluation, and abandoned/duplicate time. For attempts without a result, use the final progress wall_seconds as a lower bound. Report standalone method cost and deduplicated campaign cost separately; component seconds are not a complete decomposition. Because baseline runtime differs, wall-clock comparisons are descriptive and do not support an accelerates-training claim."), st["body"]))


def add_data_site(story, st, language="zh"):
    zh = language == "zh"
    add_heading(story, "数据协议、验证与网站契约" if zh else "Data Contract, Validation, and Site Architecture", st, 1)
    layers = [["Layer", "Path", "Contents"],
              ["Authoritative raw", "runs_eval/revised/publication_seed101_v1/<network>/<controller>/<method>/", "Immutable manifests, environment, result/summary, compressed JSONL/NPZ."],
              ["Validated release", "output_result/revised/publication_seed101_v1/", "release_manifest, validation, CSV, figures, paired bootstrap, dashboard JSON."],
              ["Sanitized site", "docs/site/dist/data/publication/", "Allowlisted aggregates and de-identified rollout JSON only."]]
    story.append(table(layers, [32 * mm, 75 * mm, 61 * mm], font_size=6.8))
    story.append(p(esc("JSON 是权威格式，CSV 仅供阅读；JSON 禁止 NaN/Infinity，缺失值使用 null 并登记 unit、direction、denominator 和 null semantics。" if zh else
                       "JSON is authoritative and CSV is a human-readable export. JSON forbids NaN/Infinity; missing values use null with registered unit, direction, denominator, and null semantics."), st["body"]))
    add_heading(story, "Selection 与完整性" if zh else "Selection and completeness", st, 2)
    story.append(p(esc("final_evaluation_seed101.json 冻结 8 parent、8 offline WCE、40 final continuations 和 40 planned suites，包含相对/绝对路径、checkpoint directory hash、manifest hash、runtime/source compatibility 和计划输出。禁止 mtime/latest 推断。" if zh else
                       "final_evaluation_seed101.json freezes 8 parents, 8 offline-WCE models, 40 final continuations, and 40 planned suites, including relative/absolute paths, checkpoint-directory hashes, manifest hashes, runtime/source compatibility, and planned outputs. mtime/latest inference is forbidden."), st["note"]))
    story.append(code_block("network/controller/method/training_seed/split/scenario/generation_seed/arrival_seed/sumo_seed/policy_seed/attempt", st))
    checks = [
        "Exactly 9,200 accepted; rejected, missing, duplicate, and extra are all zero.",
        "Forty allowlisted 2,320,000-step checkpoints; no pilot, validation, or duplicate attempt.",
        "Every rollout is complete with horizon=sample_count=3,600 and exact seed pairing.",
        "Demand/SUMO pairing is identical across methods; protocol, scenario, profile, network, artifact, checkpoint, and manifest hashes match.",
        "Timestamps are 1–3,600; queues are finite/nonnegative; integrated≈mean×3,600; peak≥mean.",
        "scheduled=inserted+pending; completed≤inserted; completed-trip denominator=completed; NPZ lane sums reconcile with JSONL totals.",
        "Public data contain no absolute paths, credentials, checkpoint payloads, raw logs, pilots, or history.",
    ]
    story.extend(bullets(checks, st))
    story.append(p(esc("磁盘实测约 69 GiB free（2026-09-29）。每次启动前重新检查：至少 60 GiB free，且预计完成后保留 20 GiB。若不满足，先实施 shared provenance、gzip JSONL、精简 manifest 和 compressed NPZ；不得删除未知用户数据。" if zh else
                       "Approximately 69 GiB was free on 2026-09-29. Recheck immediately before launch: require at least 60 GiB free and 20 GiB projected reserve. Otherwise implement shared provenance, gzip JSONL, smaller manifests, and compressed NPZ. Never delete unknown user data."), st["warn"]))
    add_heading(story, "本地实时 + 静态发布" if zh else "Local live + static publication", st, 2)
    story.append(p(esc("现有站点是静态 Sites 项目，本轮不修改或发布。托管浏览器不能扫描实验机目录。本地 dashboard/API 验证 allowlisted release 后导出脱敏快照；托管站点只 fetch 相对 JSON。旧 results.js 是 pilot schema v1，不能作为正式数据源。" if zh else
                       "The existing site is a static Sites project and is not modified or published in this delivery. A hosted browser cannot scan the experiment machine. A local dashboard/API validates allowlisted releases and exports a sanitized snapshot; the hosted site fetches relative JSON only. The old results.js is pilot schema-v1 data and is not a publication source."), st["body"]))
    story.append(code_block("GET /api/releases\nGET /api/releases/<id>/manifest\nGET /api/releases/<id>/overview\nGET /api/releases/<id>/rollouts?network=&controller=&method=&split=&scenario=", st))
    story.extend(bullets([
        "manifest.json and overview.json",
        "8 network-controller rollout shards × 1,150 records",
        "24 network-controller-peak heatmap shards",
        "optional curve shards and checksums.json",
    ], st))


def add_visual_runbook(story, st, language="zh"):
    zh = language == "zh"
    add_heading(story, "论文可视化与执行工作簿" if zh else "Paper Visualizations and Execution Workbook", st, 1)
    visual = [["Priority", "Figure", "Specification"],
              ["1", "Paired-effect forest", "Absolute online-minus-comparator queue effects with conditional 95% CIs; zero line; negative is better."],
              ["2", "Method × scenario heatmap", "Twelve test scenarios; absolute and relative-to-baseline modes; seen/test separate."],
              ["3", "Robustness frontier", "Equal-weight suite mean versus worst-three CVaR."],
              ["4", "Peak response", "Peak 1.25 and 1.50 in main text; shade [1200,2400); peak 1.10 in supplement."],
              ["5", "Network heatmaps", "Five absolute maps plus online-minus-comparator maps; common absolute and symmetric difference scales."],
              ["6", "Training cost", "Parent/offline-WCE/continuation stacks; standalone and deduplicated campaign cost; exact/lower-bound marker."],
              ["7", "Flow/reliability", "Completed, remaining, pending, teleports, collisions; trip denominators visible."],
              ["8", "WCE mechanism", "Mixture weights and entropy in supplement."]]
    story.append(table(visual, [15 * mm, 45 * mm, 108 * mm], font_size=6.8))
    story.append(p(esc("真正的 network-wide heatmap 要求正式评估前聚合所有 non-internal lanes 的峰前/峰中/峰后窗口。当前仅有 controlled incoming lanes；未扩展时必须称为 network-spanning controlled-approach heatmap。" if zh else
                       "A true network-wide heatmap requires pre/peak/post aggregation for all non-internal lanes before formal evaluation. Current data cover controlled incoming lanes only; without the extension, use the title network-spanning controlled-approach heatmap."), st["warn"]))
    add_heading(story, "论文措辞" if zh else "Paper wording", st, 2)
    story.extend(bullets([
        "per intersection → network-total mean queue",
        "unbounded growth → sustained growth within the 3,600-s horizon",
        "algorithm-agnostic confirmed → consistent trend across four architectures for training seed 101",
        "Remove accelerates training without common-runtime time-to-threshold evidence.",
        "Lead with absolute effects; select queue-worst and speed-worst scenarios separately.",
    ], st))
    add_heading(story, "Phase 0 prerequisites" if not zh else "Phase 0：正式运行前", st, 2)
    story.extend(bullets([
        "Implement protocol-v7 seeded random switching across all eleven seen profiles and regenerate/hash all sixty temporal artifacts.",
        "Implement all-non-internal-lane heatmap aggregation or lock the controlled-approach wording.",
        "Rerun the deeprlsc/SUMO/TraCI/TensorFlow/network/profile gate.",
        "Recheck disk, checkpoint hashes, and absent output directories.",
        "Independently review the one-to-one mapping between selection and forty commands.",
    ], st))
    story.append(code_block("cd /home/sdc_joran/Journal/deeprl_signal_control\nexport CBWCE_GATE=/absolute/path/to/protocol_v7/gate.json\n\n./docs/evaluation_workbook/generated/publication_workflow.sh --status\n./docs/evaluation_workbook/generated/publication_workflow.sh --preflight\n./docs/evaluation_workbook/generated/publication_workflow.sh --list\n\nCONFIRM_PUBLICATION=RUN_9200 CBWCE_EVALUATION_WORKERS=4 \\\n  ./docs/evaluation_workbook/generated/publication_workflow.sh --execute", st))
    story.append(p(esc("脚本默认只显示计划。--execute 需要 protocol v7、≥60 GiB、四 worker gate、预生成 artifacts 和不存在的输出根目录；通过后默认并行四个 suites，每个 suite 内 rollout 顺序执行。" if zh else
                       "The script prints a plan by default. --execute requires protocol v7, ≥60 GiB free, a four-worker gate, pre-materialized artifacts, and an absent output root; it then runs four suites concurrently while keeping rollouts sequential within each suite."), st["note"]))
    add_heading(story, "待实现接口 / TO BE IMPLEMENTED", st, 2)
    story.append(code_block("python main.py experiment report --selection <selection.json> --output <release>\npython main.py experiment validate-report --report <release> --require-publication-complete\npython main.py experiment export-site --report <release> --output docs/site/dist/data/publication", st))
    add_heading(story, "审稿意见映射" if zh else "Reviewer-response matrix", st, 2)
    reviewer = [["No.", "Reviewer request", "Workbook evidence"],
                ["1", "Record setup wall-clock and CB-WCE overhead", "Resume-chain audit; standalone and deduplicated campaign cost; runtime matrix."],
                ["2", "Generate unseen groups separate from training", "Frozen 12-test / 6-validation / 11-seen split with explicit interpolation/OOD limits."],
                ["3", "Align reward and primary traffic metric", "Common Q(t), interval cost, and J_Q; disclose scaling, clipping, and adversarial direction."],
                ["4", "Record rollout metrics and uncertainty", "Metric registry, ten paired seeds, sample SD, and paired bootstrap CI."],
                ["5", "Peak-demand network heatmaps", "All-non-internal-lane collection prerequisite plus common-scale absolute/difference maps."]]
    story.append(table(reviewer, [12 * mm, 58 * mm, 98 * mm], font_size=6.8))


def add_appendices(story, st, selection):
    story.append(PageBreak())
    story.append(p(esc("共享自动生成附录 / Shared Generated Appendices"), st["part"]))
    add_heading(story, "A. Named mixture weights / 按名称列出的 mixture 权重", st, 1)
    weights = [np.random.RandomState(seed).dirichlet(np.ones(11)).tolist() for seed in (41004, 41005, 41006)]
    orders = {
        "Grid": ["Center_to_Periphery", "E_to_W", "NE_to_SW", "NW_to_SE", "N_to_S", "Periphery_to_Center", "SE_to_NW", "SW_to_NE", "S_to_N", "Uniform", "W_to_E"],
        "Monaco": ["N_to_S", "S_to_N", "W_to_E", "E_to_W", "NW_to_SE", "SE_to_NW", "SW_to_NE", "NE_to_SW", "Periphery_to_Center", "Center_to_Periphery", "Uniform"],
    }
    for network, names in orders.items():
        add_heading(story, network, st, 2)
        rows = [["Profile", "Mixture 1", "Mixture 2", "Mixture 3"]]
        for i, name in enumerate(names):
            rows.append([name, "{:.9f}".format(weights[0][i]), "{:.9f}".format(weights[1][i]), "{:.9f}".format(weights[2][i])])
        story.append(table(rows, [61 * mm, 35 * mm, 35 * mm, 35 * mm], font_size=7.2))
        story.append(Spacer(1, 3 * mm))

    add_heading(story, "B. Current artifact vehicle counts / 当前 artifact 车辆数", st, 1)
    story.append(p(esc("Rows below describe current v6 artifacts. All three switch rows use legacy two-profile alternation and must be recalculated after the protocol-v7 random-switching amendment."), st["warn"]))
    rows = [["Network", "Scenario", "Mean", "Min", "Max"]]
    scenarios = ("redistribution_0.25", "redistribution_0.5", "redistribution_0.75", "mixture_1", "mixture_2", "mixture_3",
                 "switch_300", "switch_900", "switch_1200", "peak_1.1", "peak_1.25", "peak_1.5")
    for network in ("grid", "monaco"):
        directory = ROOT / ("data_traffic/revised/test/artifacts" if network == "grid" else "real_net_subnet/demand_groups/revised/test/artifacts")
        for scenario in scenarios:
            counts = []
            for seed in range(51001, 51011):
                path = directory / (scenario + "_" + str(seed) + ".json")
                if path.exists():
                    counts.append(len(json.load(path.open())["vehicles"]))
            rows.append([network, scenario, "{:.1f}".format(np.mean(counts)) if counts else "pending",
                         str(min(counts)) if counts else "–", str(max(counts)) if counts else "–"])
    story.append(table(rows, [30 * mm, 70 * mm, 23 * mm, 21 * mm, 21 * mm], font_size=7.0))

    story.append(NextPageTemplate("Landscape"))
    story.append(PageBreak())
    add_heading(story, "C. Forty selected final checkpoints / 40 个最终 checkpoint", st, 1)
    cp_rows = [["No.", "Network", "Controller", "Method", "Runtime", "Checkpoint path", "SHA-256"]]
    for no, item in enumerate(selection["continuations"], 1):
        cp_rows.append([str(no), item["network"], item["controller"], item["method"],
                        "Py {}/TF {}".format(item["runtime"].get("python"), item["runtime"].get("tensorflow")),
                        item["checkpoint_path"], item["checkpoint_hash"][:16] + "…"])
    story.append(table(cp_rows, [11 * mm, 18 * mm, 20 * mm, 35 * mm, 35 * mm, 125 * mm, 35 * mm], font_size=5.6, landscape_mode=True))
    story.append(p(esc("Authoritative absolute paths, full hashes, manifest hashes, runtime fields, and suite outputs are in runs_eval/revised/selections/final_evaluation_seed101.json."), st["caption"]))

    story.append(NextPageTemplate("Portrait"))
    story.append(PageBreak())
    add_heading(story, "D. Forty exact evaluation commands / 40 条精确评估命令", st, 1)
    story.append(p(esc("These commands are listed for reproducibility. They remain blocked until a current v7 verification gate and versioned artifacts pass preflight. No command below was executed while producing this workbook."), st["warn"]))
    gate = "${CBWCE_GATE}"
    for no, item in enumerate(selection["continuations"], 1):
        output = "runs_eval/revised/publication_seed101_v1/{}/{}/{}".format(item["network"], item["controller"], item["method"])
        add_heading(story, "{:02d}. {} / {} / {}".format(no, item["network"], item["controller"], item["method"]), st, 2)
        command = ("conda run -n deeprlsc python main.py experiment evaluate "
                   "--network {network} --controller {controller} --seed 101 "
                   "--parent '{checkpoint}' --gate '{gate}' --suite all --rollouts 10 "
                   "--no-visualization --output '{output}'").format(
                       network=item["network"], controller=item["controller"], checkpoint=item["checkpoint_absolute_path"],
                       gate=gate, output=output)
        story.append(code_block(command, st, width=100))

    add_heading(story, "E. Machine-readable paths and build commands", st, 1)
    paths = [["Artifact", "Path"],
             ["Editable ctexbook", "docs/evaluation_workbook/cb_wce_evaluation_workbook.tex"],
             ["Selection registry", "runs_eval/revised/selections/final_evaluation_seed101.json"],
             ["Schemas", "docs/evaluation_workbook/schemas/*.json"],
             ["Command catalogue", "docs/evaluation_workbook/generated/evaluate_publication_seed101.sh"],
             ["Final PDF", "output/pdf/cb_wce_evaluation_and_data_validation_workbook_zh_en.pdf"]]
    story.append(table(paths, [45 * mm, 123 * mm], font_size=7.4))
    story.append(code_block("sudo apt-get update\nsudo apt-get install -y texlive-luatex texlive-lang-chinese latexmk\n\ncd /home/sdc_joran/Journal/deeprl_signal_control/docs/evaluation_workbook\nlatexmk -lualatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=/home/sdc_joran/Journal/deeprl_signal_control/tmp/pdfs/cb_wce_evaluation_workbook cb_wce_evaluation_workbook.tex", st))
    story.append(p(esc("The standard ctex/LuaLaTeX dependency installation was blocked on this machine by a sudo-password requirement. The delivered PDF was therefore rendered with ReportLab from the same frozen registry and embedded Droid CJK plus DejaVu fonts; the editable ctexbook and standard rebuild command remain authoritative."), st["note"]))


def build_pdf():
    register_fonts()
    st = styles()
    selection = json.load(SELECTION.open())
    TEMP.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    story = []
    story.extend([Spacer(1, 26 * mm), p(esc("CB-WCE 测试与数据验证工作手册"), st["title"]),
                  p(esc("CB-WCE Evaluation and Data Validation Workbook"), st["subtitle"]), Spacer(1, 12 * mm)])
    cover = [["Workbook release", "1.0"], ["Operational protocol", "v7 implemented; versioned artifacts pending"],
             ["Training seed", "101"], ["Formal evaluation status", "0 / 9,200"], ["Snapshot date", "2026-09-29"],
             ["Deliverables", "Bilingual ctexbook, 40-model registry, command catalogue, four JSON Schemas, PDF"]]
    story.append(table(cover, [48 * mm, 112 * mm], header=False, font_size=8.5))
    story.append(Spacer(1, 12 * mm))
    story.append(p(esc("This workbook is independent from paper/main.tex. Pilot and historical results are not publication evidence."), st["note"]))
    story.append(PageBreak())
    story.append(p(esc("内容导航 / Contents"), st["part"]))
    toc = [["Part", "Chapters"], ["I 中文", "协议；指标与统计；数据验证；站点与可视化；执行工作簿"],
           ["II English", "Protocol; metrics and statistics; data validation; site and visualization; runbook"],
           ["Appendices", "Mixture weights; artifact counts; 40 checkpoints; 40 exact commands; paths and build"]]
    story.append(table(toc, [38 * mm, 128 * mm], font_size=8.5))
    story.append(PageBreak())
    story.append(p(esc("第一部分 · 中文测试与数据验证手册"), st["part"]))
    add_status_protocol(story, st, "zh")
    add_metrics(story, st, "zh")
    add_data_site(story, st, "zh")
    add_visual_runbook(story, st, "zh")
    story.append(PageBreak())
    story.append(p(esc("Part II · English Evaluation and Data Validation Workbook"), st["part"]))
    add_status_protocol(story, st, "en")
    add_metrics(story, st, "en")
    add_data_site(story, st, "en")
    add_visual_runbook(story, st, "en")
    add_appendices(story, st, selection)
    story.append(PageBreak())
    story.append(p(esc("Release sign-off / 发布签字页"), st["part"]))
    signoff = [["Check", "Reviewer / date"], ["Protocol v7 implemented and hashed", ""],
               ["Storage gate rechecked", ""], ["40 checkpoints and commands reviewed", ""],
               ["9,200 accepted; zero rejected", ""], ["Public export privacy review", ""],
               ["Paper figures and wording review", ""]]
    story.append(table(signoff, [105 * mm, 60 * mm], font_size=8.5))
    doc = WorkbookDoc(str(TEMP))
    doc.build(story)
    OUTPUT.write_bytes(TEMP.read_bytes())
    print(str(OUTPUT))


if __name__ == "__main__":
    build_pdf()
