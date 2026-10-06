#!/usr/bin/env python3
"""Render the CB-WCE evaluation workbook as a standalone Markdown file."""

from __future__ import annotations

import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "cb_wce_evaluation_workbook.md"


def replace_braced_command(text: str, command: str, left: str, right: str) -> str:
    pattern = re.compile(r"\\" + command + r"\{([^{}]*)\}")
    while pattern.search(text):
        text = pattern.sub(lambda match: left + match.group(1) + right, text)
    return text


def inline_tex(text: str) -> str:
    for command in ("code", "path", "texttt"):
        text = replace_braced_command(text, command, "`", "`")
    text = replace_braced_command(text, "status", "**", "**")
    text = replace_braced_command(text, "textbf", "**", "**")
    text = text.replace(r"\lowerbetter", "*lower is better*")
    text = text.replace(r"\hrulefill", "____________________")
    text = text.replace(r"\%", "%")
    text = text.replace(r"\_", "_")
    text = text.replace(r"\&", "&")
    text = text.replace(r"\#", "#")
    text = text.replace(r"\$", "$")
    text = text.replace("~", " ")
    return text.strip()


def split_table_row(row: str) -> list[str]:
    cells = re.split(r"(?<!\\)&", row)
    return [inline_tex(cell.strip()) for cell in cells]


def markdown_table(block: list[str]) -> list[str]:
    filtered: list[str] = []
    skip_repeat_header = False
    for raw in block:
        line = raw.strip()
        if line == r"\endfirsthead":
            skip_repeat_header = True
            continue
        if line == r"\endhead":
            skip_repeat_header = False
            continue
        if skip_repeat_header:
            continue
        if (not line or line.startswith(r"\begin{") or line.startswith(r"\end{")
                or line in {r"\toprule", r"\midrule", r"\bottomrule", r"\scriptsize"}
                or line.startswith(r"\toprule ")):
            continue
        filtered.append(line)

    joined = " ".join(filtered)
    raw_rows = re.split(r"(?<!\\)\\\\", joined)
    rows = [split_table_row(row) for row in raw_rows if row.strip()]
    if not rows:
        return []

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = rows[0]
    output = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return output


def convert_fragment(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    list_kind: str | None = None
    index = 0

    while index < len(lines):
        raw = lines[index]
        line = raw.strip()

        verbatim = re.match(r"\\begin\{Verbatim\}(?:\[[^]]*\])?", line)
        if verbatim:
            code: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != r"\end{Verbatim}":
                code.append(lines[index])
                index += 1
            output.extend(["```text", *code, "```", ""])
            index += 1
            continue

        table = re.match(r"\\begin\{(longtable|tabularx|tabular)\}", line)
        if table:
            environment = table.group(1)
            block = [line]
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(
                    r"\end{" + environment + "}"):
                block.append(lines[index])
                index += 1
            if index < len(lines):
                block.append(lines[index])
            output.extend(markdown_table(block))
            output.append("")
            index += 1
            continue

        if line == r"\[":
            math: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != r"\]":
                math.append(lines[index].strip())
                index += 1
            output.extend(["$$", *math, "$$", ""])
            index += 1
            continue

        heading = re.match(r"\\(chapter|section|subsection)\*?\{(.+)\}", line)
        if heading:
            levels = {"chapter": "###", "section": "####", "subsection": "#####"}
            output.extend([f"{levels[heading.group(1)]} {inline_tex(heading.group(2))}", ""])
            index += 1
            continue

        begin_list = re.match(r"\\begin\{(itemize|enumerate)\}", line)
        if begin_list:
            list_kind = begin_list.group(1)
            index += 1
            continue
        if line in {r"\end{itemize}", r"\end{enumerate}"}:
            list_kind = None
            output.append("")
            index += 1
            continue
        item = re.match(r"\\item\s*(.*)", line)
        if item:
            marker = "1." if list_kind == "enumerate" else "-"
            output.append(f"{marker} {inline_tex(item.group(1))}")
            index += 1
            continue

        included = re.match(r"\\input\{([^}]+)\}", line)
        if included:
            included_path = HERE / (included.group(1) + ".tex")
            output.append(convert_fragment(included_path).rstrip())
            output.append("")
            index += 1
            continue

        if not line or line in {r"\begin{center}", r"\end{center}", r"\begin{landscape}",
                                r"\end{landscape}", r"\scriptsize"}:
            if output and output[-1] != "":
                output.append("")
            index += 1
            continue

        output.append(inline_tex(line))
        output.append("")
        index += 1

    while output and output[-1] == "":
        output.pop()
    return "\n".join(output) + "\n"


def main() -> None:
    parts = [
        "# CB-WCE 测试与数据验证工作手册",
        "",
        "**CB-WCE Evaluation and Data Validation Workbook**",
        "",
        "Chinese Part I / English Part II / Shared generated appendices",
        "",
        "| Item | Value |",
        "| --- | --- |",
        "| Workbook release | 1.0 |",
        "| Operational protocol | v7 implemented; versioned artifacts pending |",
        "| Training seed | 101 |",
        "| Publication evaluation status | **0 / 9,200 formal rollouts** |",
        "| Snapshot date | 2026-09-29 |",
        "",
        "This workbook is independent from `paper/main.tex`. Pilot and historical results are not publication evidence.",
        "",
        "## 中文测试与数据验证手册",
        "",
    ]

    for path in sorted((HERE / "sections/zh").glob("*.tex")):
        parts.append(convert_fragment(path).rstrip())
        parts.append("")

    parts.extend(["## English Evaluation and Data Validation Workbook", ""])
    for path in sorted((HERE / "sections/en").glob("*.tex")):
        parts.append(convert_fragment(path).rstrip())
        parts.append("")

    parts.extend([
        "## Shared Generated Appendices / 共享自动生成附录",
        "",
        "### Frozen Protocol Snapshot / 冻结协议快照",
        "",
    ])
    for name in ("protocol_snapshot.tex", "mixture_weights.tex", "scenario_counts.tex"):
        parts.append(convert_fragment(HERE / "generated" / name).rstrip())
        parts.append("")

    parts.extend(["### Selected Models / 最终模型清单", ""])
    parts.append(convert_fragment(HERE / "generated/checkpoint_manifest.tex").rstrip())
    parts.extend(["", "### Command Catalogue / 命令目录", ""])
    parts.append(convert_fragment(HERE / "generated/command_catalog.tex").rstrip())
    parts.extend(["", "#### Evaluation campaign script", "", "```bash"])
    parts.append((HERE / "generated/evaluate_publication_seed101.sh").read_text(encoding="utf-8").rstrip())
    parts.extend([
        "```",
        "",
        "### Machine-readable Contracts / 机器可读协议",
        "",
        "The authoritative JSON Schemas are stored in `docs/evaluation_workbook/schemas/`. The selection registry is `runs_eval/revised/selections/final_evaluation_seed101.json`. JSON is authoritative; CSV is a human-readable export only.",
        "",
        "### Release sign-off / 发布签字页",
        "",
        "| Check | Sign-off |",
        "| --- | --- |",
        "| Protocol v7 implemented and hashed | ____________________ |",
        "| Storage gate rechecked immediately before launch | ____________________ |",
        "| 40 checkpoints and commands independently reviewed | ____________________ |",
        "| 9,200 accepted, zero rejected | ____________________ |",
        "| Public export privacy review | ____________________ |",
        "| Paper figures and wording review | ____________________ |",
        "| Reviewer / date | ____________________ |",
        "",
    ])
    OUTPUT.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
