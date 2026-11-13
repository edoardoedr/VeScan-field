#!/usr/bin/env python3
"""Figures and tables of the VMTK vs thalweg comparison, for analisys/results/<date>_vmtk-vs-thalweg/.

    python statistics_scripts/plots/report_vmtk_vs_thalweg.py --date 2026-11-12 [--force]

Reads only the files in analisys/extractions/<date>_vmtk-vs-thalweg/, written on the server by
statistics_scripts/ (see statistics_scripts/README.md):
  centerline_comparison.xlsx            compare_centerlines.py, reference (a) = 01_preprocessed.vtk
  centerline_comparison_thalwegref.xlsx compare_centerlines.py --reference-surfaces, reference (b) =
                                        thalweg's own boundary
  boundary_vs_reference/*.csv           boundary_vs_reference.py (thalweg boundary - 01_preprocessed.vtk)
  vmtk_seeded.xlsx                      vmtk_seeded.py (VMTK with default seeds vs seeded at thalweg tips)
  crossings_by_radius.xlsx              crossings_by_radius.py
Writes figures/*.png and tables.md (every number of REPORT.md, with its source file) into
analisys/results/<date>_vmtk-vs-thalweg/. REPORT.md is written by hand from them. A non-empty
figures/ folder is not overwritten (--force).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ANALISYS = Path(__file__).resolve().parents[2] / "analisys"
PIPELINES = ("vmtk", "thalweg")
LABEL = {"vmtk": "VMTK", "thalweg": "thalweg"}
COLOR = {"vmtk": "#2a78d6", "thalweg": "#eb6834"}
STRUCTURES = ("artery", "veins", "airways")
SLICE_BINS = ([0, 1, 2, 3], ["<=1", "1-2", "2-3"])
RADIUS_BINS = ["<1", "1-2", "2-3", ">3"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#9cc3ee", "#4f95e0", "#1f5fa8"]                    # light -> dark, for ordered bins

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


# --------------------------------------------------------------------------- helpers
def iqr(s, k=1.0, fmt="{:.3g}"):
    s = pd.to_numeric(s, errors="coerce").dropna() * k
    if not len(s):
        return "–"
    return f"{fmt.format(s.median())} [{fmt.format(s.quantile(.25))}–{fmt.format(s.quantile(.75))}]"


def pvalue(a, b):
    ok = a.notna() & b.notna()
    if ok.sum() <= 10 or not (a[ok] != b[ok]).any():
        return "–"
    return f"{wilcoxon(a[ok], b[ok]).pvalue:.0e}"


def handles(labels=PIPELINES, colors=COLOR, names=LABEL):
    return [plt.Line2D([], [], color=colors[p], lw=6, label=names[p]) for p in labels]


def boxes(ax, data, positions, colors):
    bp = ax.boxplot(data, positions=positions, widths=0.75, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    for patch, c in zip(bp["boxes"], colors):
        patch.set(facecolor=c, edgecolor=SURFACE, linewidth=2)


def finish(fig, path, title, legend=True, top=0.9):
    if legend:
        fig.legend(handles=handles(), loc="upper right", ncol=2)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def paired_boxes(df, panels, path, title):
    """One panel per metric; per structure a VMTK and a thalweg box (no outliers drawn)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.9))
    for ax, (metric, ylabel, scale, log) in zip(np.atleast_1d(axes), panels):
        pos, data, colors = [], [], []
        for i, st in enumerate(STRUCTURES):
            g = df[df.structure == st]
            for j, pl in enumerate(PIPELINES):
                pos.append(i * 3 + j)
                data.append(pd.to_numeric(g[f"{metric}_{pl}"], errors="coerce").dropna().to_numpy() * scale)
                colors.append(COLOR[pl])
        boxes(ax, data, pos, colors)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(STRUCTURES))], STRUCTURES)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.set_xlim(-0.8, len(STRUCTURES) * 3 - 1.2)
    finish(fig, path, title, top=0.92)


# --------------------------------------------------------------------------- figures
def agreement_figure(a, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    for ax, (m, ylabel, k) in zip(axes, [("matched_vmtk_in_thalweg", "VMTK length within thalweg (%)", 100),
                                         ("matched_thalweg_in_vmtk", "thalweg length within VMTK (%)", 100),
                                         ("dist_median_vmtk_to_thalweg", "median distance VMTK → thalweg (mm)", 1)]):
        boxes(ax, [a[a.structure == st][m].dropna() * k for st in STRUCTURES], range(1, 4), ["#a7a6a0"] * 3)
        ax.set_xticks(range(1, 4), STRUCTURES)
        ax.set_ylabel(ylabel)
    finish(fig, path, "Agreement between the two centerlines (one value per patient and structure)",
           legend=False, top=0.92)


def two_references_figure(a, b, path):
    """|radius error| and centering offset against reference (a) and (b)."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    axes[1].sharey(axes[0])
    axes[3].sharey(axes[2])
    panels = [(a, "abs_radius_err_median", "|radius error| (%)", "(a) 01_preprocessed.vtk"),
              (b, "abs_radius_err_median", "|radius error| (%)", "(b) thalweg boundary"),
              (a, "offset_rel_median", "centering offset / radius (%)", "(a) 01_preprocessed.vtk"),
              (b, "offset_rel_median", "centering offset / radius (%)", "(b) thalweg boundary")]
    for ax, (df, m, ylabel, title) in zip(axes, panels):
        pos, data, colors = [], [], []
        for i, st in enumerate(STRUCTURES):
            g = df[df.structure == st]
            for j, pl in enumerate(PIPELINES):
                pos.append(i * 3 + j)
                data.append(g[f"{m}_{pl}"].dropna() * 100)
                colors.append(COLOR[pl])
        boxes(ax, data, pos, colors)
        ax.set_xticks([i * 3 + 0.5 for i in range(3)], STRUCTURES)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10)
    finish(fig, path, "Local precision depends on the reference surface (lower is better)", top=0.88)


def boundary_figure(fine, path):
    """thalweg boundary - 01_preprocessed.vtk by local radius: mm per structure; % by slice spacing."""
    fine = fine.copy()
    fine["r_mid"] = (fine.r_lo + fine.r_hi) / 2
    fine = fine[fine.r_lo >= 0.5]                              # below 0.5 mm the radius is unreliable and % explodes
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    st_colors = {"artery": "#d6452a", "veins": "#2a78d6", "airways": "#3a9b5c"}
    st_style = {"artery": dict(lw=3.5), "veins": dict(lw=1.8, ls="--"), "airways": dict(lw=2)}   # veins overlap the arteries
    for st in STRUCTURES:
        g = fine[(fine.structure == st) & (fine.r_mid <= 6)].groupby("r_mid").dist_median
        med, lo, hi = g.median(), g.quantile(.25), g.quantile(.75)
        axes[0].plot(med.index, med, color=st_colors[st], label=st, **st_style[st])
        axes[0].fill_between(med.index, lo, hi, color=st_colors[st], alpha=0.15, lw=0)
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].set_xlabel("local vessel radius (mm)")
    axes[0].set_ylabel("thalweg boundary − reference (mm)")
    axes[0].set_title("Offset (median, IQR band)", loc="left", fontsize=10)
    axes[0].legend()
    for ax, st in zip(axes[1:], STRUCTURES):
        for color, sb in zip(SEQ, SLICE_BINS[1]):
            g = fine[(fine.structure == st) & (fine.sbin == sb) & (fine.r_mid <= 4)].groupby("r_mid").rel_median
            ax.plot(g.median().index, g.median() * 100, color=color, lw=2, label=f"slice {sb} mm")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("local vessel radius (mm)")
        ax.set_ylabel("offset (% of the radius)")
        ax.set_title(f"{st}: by slice spacing", loc="left", fontsize=10)
        ax.legend()
    finish(fig, path, "Where thalweg's boundary lies against 01_preprocessed.vtk (positive = outside)",
           legend=False, top=0.9)


def by_slice_figure(a, b, path):
    """Per slice-spacing bin: VMTK's unreached surface and |radius error| (a) and (b), arteries."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    x = np.arange(len(SLICE_BINS[1]))
    for st, mk in zip(STRUCTURES, ("o", "s", "^")):
        g = a[a.structure == st].groupby("sbin", observed=False).missing_area_pct_vmtk.median()
        axes[0].plot(x, g.to_numpy(), marker=mk, color=MUTED, lw=1.5, label=st)
    axes[0].set_ylabel("surface not reached by VMTK (%)")
    axes[0].set_title("VMTK unreached surface", loc="left", fontsize=10)
    axes[0].legend()
    for ax, df, title in ((axes[1], a, "artery, |radius error| vs (a)"), (axes[2], b, "artery, |radius error| vs (b)")):
        g = df[df.structure == "artery"]
        for pl in PIPELINES:
            m = g.groupby("sbin", observed=False)[f"abs_radius_err_median_{pl}"].median() * 100
            ax.plot(x, m.to_numpy(), marker="o", color=COLOR[pl], lw=2)
        ax.set_ylabel("|radius error| (%)")
        ax.set_title(title, loc="left", fontsize=10)
    for ax in axes:
        ax.set_xticks(x, [f"{s} mm" for s in SLICE_BINS[1]])
        ax.set_xlabel("slice spacing")
    finish(fig, path, "Effect of the slice spacing (medians per bin)", top=0.88)


def seeded_figure(s, path):
    runs = {"default": "#2a78d6", "seeded": "#8fb8ea"}
    names = {"default": "VMTK, default seeds", "seeded": "VMTK, seeded at the thalweg tips"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, (cols, ylabel) in zip(axes, [((("reached_frac_default", "default"), ("reached_frac_seeded", "seeded")),
                                          "targets reached (%)"),
                                         ((("missing_area_pct_default", "default"), ("missing_area_pct_seeded", "seeded"),
                                           ("missing_area_pct_both", "both")), "surface reached by thalweg only (%)")]):
        pos, data, colors = [], [], []
        for i, st in enumerate(STRUCTURES):
            g = s[s.structure == st]
            for j, (c, run) in enumerate(cols):
                pos.append(i * (len(cols) + 1) + j)
                k = 100 if c.startswith("reached") else 1
                data.append(pd.to_numeric(g[c], errors="coerce").dropna() * k)
                colors.append(runs.get(run, "#a7a6a0"))
        boxes(ax, data, pos, colors)
        ax.set_xticks([i * (len(cols) + 1) + (len(cols) - 1) / 2 for i in range(3)], STRUCTURES)
        ax.set_ylabel(ylabel)
    fig.legend(handles=[plt.Line2D([], [], color=c, lw=6, label=names[r]) for r, c in runs.items()]
               + [plt.Line2D([], [], color="#a7a6a0", lw=6, label="neither VMTK run")], loc="upper right", ncol=3)
    finish(fig, path, "VMTK seeded at the thalweg tips (NEXT 7.4)", legend=False, top=0.86)


def crossings_figure(sm, ag, path):
    """Prevalence by radius bin at min depth 0 and 9 mm, both pipelines, with Cohen's kappa."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.6), sharey="row")
    radii = ["<1", "1-2", ">2", "any"]
    for row, depth in enumerate((0, 9)):
        for ax, st in zip(axes[row], STRUCTURES):
            x = np.arange(len(radii))
            for j, pl in enumerate(PIPELINES):
                v = [sm[(sm.structure == st) & (sm.contact == "any") & (sm.radius == r) & (sm.min_depth_mm == depth)
                        & (sm.pipeline == pl)].prevalence_pct.iloc[0] for r in radii]
                ax.bar(x + (j - 0.5) * 0.38, v, 0.36, color=COLOR[pl], edgecolor=SURFACE, linewidth=2)
            for xi, r in zip(x, radii):
                k = ag[(ag.structure == st) & (ag.contact == "any") & (ag.radius == r) & (ag.min_depth_mm == depth)]
                kv = k.kappa_presence.iloc[0] if len(k) else np.nan
                top = sm[(sm.structure == st) & (sm.contact == "any") & (sm.radius == r) & (sm.min_depth_mm == depth)].prevalence_pct.max()
                ax.text(xi, top + 2, f"κ {kv:.2f}" if pd.notna(kv) else "κ –", ha="center", fontsize=8, color=MUTED)
            ax.set_xticks(x, [f"r {r}" if r != "any" else "any r" for r in radii])
            ax.set_title(f"{st}, depth ≥ {depth} mm", loc="left", fontsize=10)
            ax.set_ylim(0, 105)
        axes[row][0].set_ylabel("patients with ≥ 1 crossing (%)")
    finish(fig, path, "Crossings by vessel radius and depth past the fissure (all datasets; κ = Cohen's kappa on presence)",
           top=0.92)


# --------------------------------------------------------------------------- tables
METRICS = {
    "agreement": [("matched_vmtk_in_thalweg", "VMTK length within thalweg (%)", 100),
                  ("matched_thalweg_in_vmtk", "thalweg length within VMTK (%)", 100),
                  ("matched_f1_vmtk_thalweg", "F1", 1),
                  ("dist_median_vmtk_to_thalweg", "distance VMTK → thalweg, median (mm)", 1),
                  ("dist_p95_vmtk_to_thalweg", "distance VMTK → thalweg, p95 (mm)", 1),
                  ("dist_median_thalweg_to_vmtk", "distance thalweg → VMTK, median (mm)", 1),
                  ("dist_p95_thalweg_to_vmtk", "distance thalweg → VMTK, p95 (mm)", 1)],
    "paired": [("missing_area_pct", "surface not reached (%)", 1, True),
               ("length_mm", "total length (m)", 0.001, None),
               ("leaves", "terminal branches", 1, None),
               ("max_generation", "maximum generation", 1, None),
               ("path_max_mm", "path root → farthest tip (mm)", 1, None),
               ("path_p90_mm", "path root → tip, p90 (mm)", 1, None),
               ("radius_min_mm", "smallest radius (mm)", 1, None),
               ("radius_p05_mm", "radius, 5th percentile (mm)", 1, None),
               ("spur_frac", "terminal spurs, new definition (%)", 100, True),
               ("short_leaf_frac", "terminal spurs, old definition (%)", 100, True),
               ("curvature_median", "curvature (rad/mm)", 1, True),
               ("time_s", "run time per structure (s)", 1, True)],
    "precision": [("offset_rel_median", "centering offset / radius (%)", 100, True),
                  ("offset_mm_median", "centering offset (mm)", 1, True),
                  ("abs_radius_err_median", "radius error, absolute (%)", 100, True),
                  ("radius_err_median", "radius error, signed (%)", 100, None),
                  ("outside_frac", "length outside the surface (%)", 100, True)],
}


def paired_table(df, metrics, out):
    out.append("| metric | VMTK, median [IQR] | thalweg, median [IQR] | thalweg better | Wilcoxon p |")
    out.append("|---|---|---|---|---|")
    for m, name, k, lower_better in metrics:
        v, t = pd.to_numeric(df[f"{m}_vmtk"], errors="coerce"), pd.to_numeric(df[f"{m}_thalweg"], errors="coerce")
        ok = v.notna() & t.notna()
        better = "" if lower_better is None else f"{100 * (t < v)[ok].mean():.0f} %"
        out.append(f"| {name} | {iqr(v, k)} | {iqr(t, k)} | {better} | {pvalue(v, t)} |")


def tables(a, b, bnd, fine, s, sm, ag, dropped, ext):
    src = lambda f: f"*Source: `analisys/extractions/{ext}/{f}`*"     # noqa: E731
    out = [f"# Tables: VMTK vs thalweg ({ext})", "",
           "Generated by `statistics_scripts/plots/report_vmtk_vs_thalweg.py`. Medians over patient × structure, "
           "[interquartile range]. *thalweg better* = share of cases where thalweg's value is lower. "
           "Wilcoxon signed-rank test on the same cases.", ""]

    out += ["## T1. Cohort", "", src("centerline_comparison.xlsx") + ", sheet `per_structure` "
            "(`statistics_scripts/compare_centerlines.py`)", "",
            "| dataset | artery | veins | airways | slice spacing, median [range] (mm) |", "|---|---|---|---|---|"]
    for ds, g in a.groupby("dataset"):
        n = g.groupby("structure").size()
        out.append(f"| {ds} | {n.get('artery', 0)} | {n.get('veins', 0)} | {n.get('airways', 0)} | "
                   f"{g.slice_mm.median():.2f} [{g.slice_mm.min():.2f}–{g.slice_mm.max():.2f}] |")
    n = a.groupby("structure").size()
    out.append(f"| **all** | {n['artery']} | {n['veins']} | {n['airways']} | {a.slice_mm.median():.2f} |")

    out += ["", "## T2. Agreement between the centerlines (7.1)", "",
            src("centerline_comparison.xlsx") + " (`compare_centerlines.py`)", "",
            "| metric | " + " | ".join(STRUCTURES) + " |", "|---|---|---|---|"]
    for m, name, k in METRICS["agreement"]:
        out.append(f"| {name} | " + " | ".join(iqr(a[a.structure == st][m], k) for st in STRUCTURES) + " |")

    out += ["", "## T3. Completeness, depth, spurs, smoothness, time (7.1, 7.3)", "",
            src("centerline_comparison.xlsx") + " (`compare_centerlines.py`)"]
    for st in STRUCTURES:
        g = a[a.structure == st]
        out += ["", f"### {st} (n = {len(g)})", ""]
        paired_table(g, METRICS["paired"], out)

    out += ["", "## T4. By dataset (medians)", "", src("centerline_comparison.xlsx"), "",
            "| structure | dataset | n | VMTK unreached (%) | thalweg within VMTK (%) | dist. VMTK→thalweg (mm) | "
            "VMTK time (s) | thalweg time (s) | VMTK max gen. | thalweg max gen. |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (st, ds), g in a.groupby(["structure", "dataset"]):
        out.append(f"| {st} | {ds} | {len(g)} | {g.missing_area_pct_vmtk.median():.1f} | "
                   f"{100 * g.matched_thalweg_in_vmtk.median():.1f} | {g.dist_median_vmtk_to_thalweg.median():.3f} | "
                   f"{g.time_s_vmtk.median():.0f} | {g.time_s_thalweg.median():.0f} | "
                   f"{g.max_generation_vmtk.median():.0f} | {g.max_generation_thalweg.median():.0f} |")

    for ref, df, f in (("(a) `01_preprocessed.vtk`", a, "centerline_comparison.xlsx"),
                       ("(b) thalweg's own boundary", b, "centerline_comparison_thalwegref.xlsx")):
        out += ["", f"## T5{'a' if df is a else 'b'}. Local precision against reference {ref} (7.2)", "",
                src(f) + (" (`compare_centerlines.py --reference-surfaces`)" if df is b else " (`compare_centerlines.py`)")]
        for st in STRUCTURES:
            out += ["", f"### {st}", ""]
            paired_table(df[df.structure == st], METRICS["precision"], out)

    out += ["", "## T6. thalweg boundary − `01_preprocessed.vtk`, by local radius (7.2)", "",
            "*Source: `analisys/extractions/" + ext + "/boundary_vs_reference/boundary_vs_reference_<dataset>.csv` "
            "(`statistics_scripts/boundary_vs_reference.py`)*. Per case, the median signed distance (positive = "
            "thalweg's boundary outside the reference) of the boundary points in each radius bin; the table gives "
            "the median [IQR] over cases. Relative = distance / local radius.", "",
            "| structure | radius bin | cases | local radius (mm) | distance (mm) | relative (%) |", "|---|---|---|---|---|---|"]
    for st in STRUCTURES:
        for rb in RADIUS_BINS + ["all"]:
            g = bnd[(bnd.structure == st) & (bnd.bin == rb)]
            out.append(f"| {st} | {rb} | {len(g)} | {g.radius_median.median():.2f} | {iqr(g.dist_median)} | "
                       f"{iqr(g.rel_median, 100)} |")
    out += ["", "### By slice spacing: median distance (mm) and relative (%)", "",
            "| structure | radius bin | " + " | ".join(f"slice {x} mm" for x in SLICE_BINS[1]) + " |", "|---|---|---|---|---|"]
    for st in STRUCTURES:
        for rb in RADIUS_BINS:
            cells = []
            for sb in SLICE_BINS[1]:
                g = bnd[(bnd.structure == st) & (bnd.bin == rb) & (bnd.sbin == sb)]
                cells.append(f"{g.dist_median.median():.2f} mm ({100 * g.rel_median.median():.0f} %, n={len(g)})")
            out.append(f"| {st} | {rb} | " + " | ".join(cells) + " |")

    out += ["", "## T7. By slice spacing (7.3)", "",
            src("centerline_comparison.xlsx") + " and `centerline_comparison_thalwegref.xlsx`. Slice spacing = "
            "z spacing of the masks (`slice_mm`).", "",
            "| structure | slice | n | VMTK unreached (%) | thalweg within VMTK (%) | abs. radius err. VMTK (a) | "
            "abs. radius err. thalweg (a) | abs. radius err. VMTK (b) | abs. radius err. thalweg (b) | thalweg leaves |", "|---|---|---|---|---|---|---|---|---|---|"]
    for st in STRUCTURES:
        for sb in SLICE_BINS[1]:
            g, h = a[(a.structure == st) & (a.sbin == sb)], b[(b.structure == st) & (b.sbin == sb)]
            out.append(f"| {st} | {sb} mm | {len(g)} | {g.missing_area_pct_vmtk.median():.1f} | "
                       f"{100 * g.matched_thalweg_in_vmtk.median():.1f} | "
                       f"{100 * g.abs_radius_err_median_vmtk.median():.1f} % | {100 * g.abs_radius_err_median_thalweg.median():.1f} % | "
                       f"{100 * h.abs_radius_err_median_vmtk.median():.1f} % | {100 * h.abs_radius_err_median_thalweg.median():.1f} % | "
                       f"{g.leaves_thalweg.median():.0f} |")
    from scipy.stats import spearmanr
    out.append("")
    out.append("Spearman correlation with the slice spacing: " + "; ".join(
        f"{st}: VMTK unreached {spearmanr(g.slice_mm, g.missing_area_pct_vmtk).statistic:.2f}, "
        f"VMTK leaves {spearmanr(g.slice_mm, g.leaves_vmtk).statistic:.2f}, "
        f"thalweg leaves {spearmanr(g.slice_mm, g.leaves_thalweg).statistic:.2f}"
        for st, g in a.groupby("structure")) + ".")

    out += ["", "## T8. VMTK seeded at the thalweg tips (7.4)", "",
            src("vmtk_seeded.xlsx") + ", sheet `per_case` (`statistics_scripts/vmtk_seeded.py`). "
            "default = VMTK's automatic endpoints; seeded = thalweg tips. Both from VMTK's own source point, "
            "on `01_preprocessed.vtk`, raw centerlines (no branch split).", "",
            "| metric | " + " | ".join(f"{st} default | {st} seeded" for st in STRUCTURES) + " |",
            "|---|" + "---|" * 6]
    rows = [("targets_{}", "targets", 1), ("reached_frac_{}", "targets reached (%)", 100),
            ("excluded_{}", "targets excluded by the bisection", 1),
            ("missing_area_pct_{}", "surface reached by thalweg only (%)", 1),
            ("matched_thalweg_in_{}", "thalweg length within VMTK (%)", 100),
            ("matched_{}_in_thalweg", "VMTK length within thalweg (%)", 100),
            ("dist_median_{}_to_thalweg", "distance VMTK → thalweg, median (mm)", 1),
            ("length_{}_mm", "length, overlaps once (m)", 0.001), ("time_{}_s", "extraction time (s)", 1)]
    for c, name, k in rows:
        out.append(f"| {name} | " + " | ".join(iqr(s[s.structure == st][c.format(r)], k)
                                             for st in STRUCTURES for r in ("default", "seeded")) + " |")
    out.append("| cases with ≥ 1 excluded target (%) | " + " | ".join(
        f"{100 * (pd.to_numeric(s[s.structure == st][f'excluded_{r}'], errors='coerce') > 0).mean():.1f}"
        for st in STRUCTURES for r in ("default", "seeded")) + " |")
    out.append("")
    out.append("Surface reached by thalweg and by neither VMTK run (`missing_area_pct_both`): " + "; ".join(
        f"{st} {iqr(s[s.structure == st].missing_area_pct_both)} %" for st in STRUCTURES) + ".")
    out.append(f"Thalweg tips per case: " + "; ".join(f"{st} {iqr(s[s.structure == st].thalweg_tips)}" for st in STRUCTURES)
               + "; tips dropped as farther than 1.5 × radius from the surface: " + "; ".join(
                   f"{st} {iqr(s[s.structure == st].thalweg_tips_far_from_surface)}" for st in STRUCTURES) + ".")

    out += ["", "## T9. Crossings by vessel radius and depth past the fissure (7.5)", "",
            src("crossings_by_radius.xlsx") + ", sheets `summary` and `agreement`, rows `dataset = all` "
            "(`statistics_scripts/crossings_by_radius.py`). Radius = the pipeline's own radius at the piercing "
            "point; depth = length after the piercing. Prevalence = patients with ≥ 1 crossing; κ = Cohen's kappa "
            "on presence per patient; exact = patients with the same count in both pipelines.", "",
            "| structure | radius | min depth (mm) | VMTK prevalence | thalweg prevalence | VMTK / patient | "
            "thalweg / patient | exact count (%) | κ |", "|---|---|---|---|---|---|---|---|---|"]
    for st in STRUCTURES + ("any",):
        for r in ("any", "<1", "1-2", ">2"):
            for d in (0, 2, 5, 9, 20):
                q = (sm.structure == st) & (sm.contact == "any") & (sm.radius == r) & (sm.min_depth_mm == d)
                v, t = sm[q & (sm.pipeline == "vmtk")].iloc[0], sm[q & (sm.pipeline == "thalweg")].iloc[0]
                k = ag[(ag.structure == st) & (ag.contact == "any") & (ag.radius == r) & (ag.min_depth_mm == d)].iloc[0]
                kv = "–" if pd.isna(k.kappa_presence) else f"{k.kappa_presence:.2f}"
                out.append(f"| {st} | {r} | {d} | {v.prevalence_pct:.1f} % | {t.prevalence_pct:.1f} % | "
                           f"{v.crossings_per_patient:.2f} | {t.crossings_per_patient:.2f} | "
                           f"{k.exact_count_match_pct:.1f} | {kv} |")
    out += ["", "### Per contact, any radius", "",
            "| structure | contact | min depth (mm) | VMTK prevalence | thalweg prevalence | κ |", "|---|---|---|---|---|---|"]
    for st in STRUCTURES + ("any",):
        for c in ("R-HIF", "R-UOIF", "R-LOIF", "L-major"):
            for d in (0, 9, 20):
                q = (sm.structure == st) & (sm.contact == c) & (sm.radius == "any") & (sm.min_depth_mm == d)
                k = ag[(ag.structure == st) & (ag.contact == c) & (ag.radius == "any") & (ag.min_depth_mm == d)].iloc[0]
                out.append(f"| {st} | {c} | {d} | {sm[q & (sm.pipeline == 'vmtk')].prevalence_pct.iloc[0]:.1f} % | "
                           f"{sm[q & (sm.pipeline == 'thalweg')].prevalence_pct.iloc[0]:.1f} % | {k.kappa_presence:.2f} |")
    out += ["", "### Crossings not counted (sheet `dropped`)", "", "| pipeline | reason | crossings |", "|---|---|---|"]
    for (pl, reason), n in dropped.groupby(["pipeline", "reason"]).crossings.sum().items():
        out.append(f"| {pl} | {reason} | {n} |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ext = f"{args.date}_vmtk-vs-thalweg"
    src, out = ANALISYS / "extractions" / ext, ANALISYS / "results" / ext
    figures = out / "figures"
    if figures.exists() and any(figures.iterdir()) and not args.force:
        raise SystemExit(f"{figures} is not empty (--force)")
    figures.mkdir(parents=True, exist_ok=True)

    a = pd.read_excel(src / "centerline_comparison.xlsx", "per_structure")
    b = pd.read_excel(src / "centerline_comparison_thalwegref.xlsx", "per_structure")
    for df in (a, b):
        df["sbin"] = pd.cut(df.slice_mm, SLICE_BINS[0], labels=SLICE_BINS[1])
    slices = a[["dataset", "patient", "structure", "sbin"]]
    bdir = src / "boundary_vs_reference"
    bnd = pd.concat([pd.read_csv(f) for f in sorted(bdir.glob("boundary_vs_reference_*.csv")) if not f.stem.endswith("_fine")])
    fine = pd.concat([pd.read_csv(f) for f in sorted(bdir.glob("boundary_vs_reference_*_fine.csv"))])
    bnd, fine = (d.merge(slices, on=["dataset", "patient", "structure"], how="inner") for d in (bnd, fine))
    s = pd.read_excel(src / "vmtk_seeded.xlsx", "per_case")
    cx = src / "crossings_by_radius.xlsx"
    sm, ag, dropped = (pd.read_excel(cx, sh) for sh in ("summary", "agreement", "dropped"))
    sm, ag = sm[sm.dataset == "all"], ag[ag.dataset == "all"]

    agreement_figure(a, figures / "agreement.png")
    paired_boxes(a, [("missing_area_pct", "surface not reached (%)", 1, False),
                     ("length_mm", "total centerline length (m)", 0.001, False),
                     ("max_generation", "maximum generation", 1, False)],
                 figures / "completeness.png", "Completeness")
    two_references_figure(a, b, figures / "precision_two_references.png")
    boundary_figure(fine, figures / "boundary_vs_reference.png")
    paired_boxes(a, [("path_max_mm", "path root → farthest tip (mm)", 1, False),
                     ("radius_min_mm", "smallest radius (mm)", 1, False),
                     ("spur_frac", "terminal spurs, new definition (%)", 100, False)],
                 figures / "depth_and_spurs.png", "Tree depth and terminal spurs")
    paired_boxes(a, [("curvature_median", "curvature (rad/mm)", 1, False),
                     ("time_s", "run time per structure (s)", 1, True)],
                 figures / "smoothness_and_time.png", "Smoothness and run time (lower is better)")
    by_slice_figure(a, b, figures / "by_slice.png")
    seeded_figure(s, figures / "vmtk_seeded.png")
    crossings_figure(sm, ag, figures / "crossings_by_radius.png")
    (out / "tables.md").write_text(tables(a, b, bnd, fine, s, sm, ag, dropped, ext))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
