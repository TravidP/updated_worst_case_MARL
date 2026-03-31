#!/usr/bin/env python3
"""Manually compare two existing raw eval CSVs from runs_eval.

Edit the configuration block below, then run:
    python3 plot_manual_eval_comparison.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Tuple

_MPLCONFIGDIR = Path("/tmp/matplotlib")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib as mpl

mpl.use("Agg")


def _tex_package_available(package_name: str) -> bool:
    kpsewhich = shutil.which("kpsewhich")
    if not kpsewhich:
        return False
    result = subprocess.run(
        [kpsewhich, package_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def _configure_plot_style() -> None:
    params = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 16,
        "figure.titlesize": 20,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    missing_bins = [exe for exe in ("latex", "kpsewhich", "dvipng") if shutil.which(exe) is None]
    required_sty = ("newtxtext.sty", "newtxmath.sty", "type1ec.sty")
    missing_sty = [sty for sty in required_sty if not _tex_package_available(sty)]

    if missing_bins or missing_sty:
        params["text.usetex"] = False
        params.pop("text.latex.preamble", None)
        warnings.warn(
            "Falling back to Matplotlib text rendering because TeX dependencies are missing. "
            "Missing executables: %s. Missing TeX packages: %s."
            % (missing_bins or "none", missing_sty or "none"),
            RuntimeWarning,
        )

    mpl.rcParams.update(params)


_configure_plot_style()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ===========================
# Manual configuration block
# ===========================
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_EVAL_DIR = PROJECT_ROOT / "runs_eval"
OUTPUT_DIR = RUNS_EVAL_DIR / "manual_comparisons"

# You can set just a filename (searched recursively inside runs_eval),
# or a relative path under runs_eval, or an absolute path.
Compare = "PPO_worst_queue_comparison.png"
BASELINE_CSV = "signal_controller_benchmark/ppo_marl_group02_demand_w_to_e_queue_raw.csv"
RETRAINED_CSV = "signal_controller_benchmark/ppo_retrained_group05_demand_se_to_nw_queue_raw.csv"

BASELINE_LABEL = "Baseline Worst Group (group02)"
RETRAINED_LABEL = "Retrained Worst Group (group05)"

# Leave empty for automatic values.
PLOT_TITLE = "Worst-case Queue Comparison (PPO)"
Y_LABEL = ""
OUTPUT_FILENAME = Compare


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _resolve_csv(csv_ref: str) -> Path:
    candidate = Path(csv_ref).expanduser()

    if candidate.is_absolute():
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError("CSV path does not exist: %s" % candidate)

    direct_in_runs_eval = RUNS_EVAL_DIR / candidate
    if direct_in_runs_eval.exists():
        return direct_in_runs_eval.resolve()

    direct_in_project = PROJECT_ROOT / candidate
    if direct_in_project.exists():
        return direct_in_project.resolve()

    # If a partial subpath under runs_eval is provided (e.g.
    # "signal_controller_benchmark/file.csv"), search only in that subtree.
    if candidate.parent != Path("."):
        scoped_root = RUNS_EVAL_DIR / candidate.parent
        if scoped_root.exists() and scoped_root.is_dir():
            scoped_matches = sorted(scoped_root.rglob(candidate.name))
            if len(scoped_matches) == 1:
                return scoped_matches[0].resolve()
            if len(scoped_matches) > 1:
                rel_matches = [str(path.relative_to(RUNS_EVAL_DIR)) for path in scoped_matches]
                raise ValueError(
                    "Ambiguous CSV '%s' inside %s. Matches:\n- %s"
                    % (csv_ref, candidate.parent, "\n- ".join(rel_matches))
                )

    matches = sorted(RUNS_EVAL_DIR.rglob(candidate.name))
    if not matches:
        raise FileNotFoundError(
            "Could not find '%s' under %s" % (csv_ref, RUNS_EVAL_DIR)
        )
    if len(matches) > 1:
        rel_matches = [str(path.relative_to(RUNS_EVAL_DIR)) for path in matches]
        raise ValueError(
            "Ambiguous CSV '%s'. Multiple matches under runs_eval:\n- %s\n"
            "Use a relative path like signal_controller_benchmark/..."
            % (csv_ref, "\n- ".join(rel_matches))
        )
    return matches[0].resolve()


def _infer_metric(csv_name: str) -> str:
    name = csv_name.lower()
    if "_queue_raw" in name:
        return "queue"
    if "_speed_raw" in name:
        return "speed"
    return "generic"


def _load_raw_stats(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "t" not in df.columns:
        raise ValueError("CSV missing required 't' column: %s" % csv_path)

    rollout_cols = [col for col in df.columns if col != "t"]
    if not rollout_cols:
        raise ValueError("CSV has no rollout columns: %s" % csv_path)

    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
    values = df[rollout_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if values.ndim != 2:
        raise ValueError("Unexpected CSV shape in %s" % csv_path)

    mean = np.nanmean(values, axis=1)
    vmin = np.nanmin(values, axis=1)
    vmax = np.nanmax(values, axis=1)
    return t, mean, vmin, vmax


def _build_defaults(metric: str) -> Tuple[str, str]:
    if metric == "queue":
        return "Total Queue Comparison", "Total queued vehicles (veh)"
    if metric == "speed":
        return "Average Speed Comparison", "Average speed (m/s)"
    return "Raw Metric Comparison", "Metric value"


def main() -> None:
    baseline_path = _resolve_csv(BASELINE_CSV)
    retrained_path = _resolve_csv(RETRAINED_CSV)

    baseline_metric = _infer_metric(baseline_path.name)
    retrained_metric = _infer_metric(retrained_path.name)
    if (
        baseline_metric in {"queue", "speed"}
        and retrained_metric in {"queue", "speed"}
        and baseline_metric != retrained_metric
    ):
        raise ValueError(
            "CSV types mismatch: '%s' looks like %s, '%s' looks like %s."
            % (baseline_path.name, baseline_metric, retrained_path.name, retrained_metric)
        )

    metric = baseline_metric if baseline_metric != "generic" else retrained_metric
    default_title, default_ylabel = _build_defaults(metric)
    title = PLOT_TITLE.strip() if PLOT_TITLE.strip() else default_title
    ylabel = Y_LABEL.strip() if Y_LABEL.strip() else default_ylabel

    left_t, left_mean, left_min, left_max = _load_raw_stats(baseline_path)
    right_t, right_mean, right_min, right_max = _load_raw_stats(retrained_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILENAME.strip():
        output_name = OUTPUT_FILENAME.strip()
    else:
        metric_slug = metric if metric != "generic" else "metric"
        output_name = "%s_vs_%s_%s.png" % (
            _slugify(BASELINE_LABEL),
            _slugify(RETRAINED_LABEL),
            metric_slug,
        )
    output_path = OUTPUT_DIR / output_name

    plt.figure(figsize=(11, 6))
    left_line, = plt.plot(left_t, left_mean, label=BASELINE_LABEL)
    plt.fill_between(left_t, left_min, left_max, alpha=0.18, color=left_line.get_color())

    right_line, = plt.plot(right_t, right_mean, label=RETRAINED_LABEL)
    plt.fill_between(right_t, right_min, right_max, alpha=0.18, color=right_line.get_color())

    plt.xlabel("Simulation second")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print("Baseline CSV :", baseline_path)
    print("Retrained CSV:", retrained_path)
    print("Saved plot   :", output_path)


if __name__ == "__main__":
    main()
