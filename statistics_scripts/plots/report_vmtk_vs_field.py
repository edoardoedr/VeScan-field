#!/usr/bin/env python3
"""Figures and tables of the VMTK vs field comparison, for analisys/results/<date>_vmtk-vs-field/.

    python statistics_scripts/plots/report_vmtk_vs_field.py --date 2026-11-03 [--force]

Reads analisys/extractions/<date>_vmtk-vs-field/centerline_comparison.csv, the two crossing
extractions <date>_vmtk/ and <date>_field/, and the fissure_length_curves / fissure_vessel_composition
results of each pipeline (run those first). Writes figures/*.png and tables.md (the numbers of
the report) into analisys/results/<date>_vmtk-vs-field/; the report text (REPORT.md) is written
by hand from them. A non-empty figures/ folder is not overwritten (--force).
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
PIPELINES = ("vmtk", "field")
LABEL = {"vmtk": "VMTK", "field": "field"}
COLOR = {"vmtk": "#2a78d6", "field": "#eb6834"}          # categorical slots 1-2
STRUCTURES = ("artery", "veins", "airways")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FISSURES = {"R-HIF": "right horizontal", "R-UOIF": "right oblique, upper",
            "R-LOIF": "right oblique, lower", "L-major": "left oblique"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def legend_handles():
    return [plt.Line2D([], [], color=COLOR[p], lw=6, label=LABEL[p]) for p in PIPELINES]


def paired_boxes(df, panels, path, title):
    """One panel per metric; per structure a VMTK and a field box (no outliers drawn)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.9))
    for ax, (metric, ylabel, scale, log) in zip(np.atleast_1d(axes), panels):
        pos, data, colors = [], [], []
        for i, st in enumerate(STRUCTURES):
            g = df[df.structure == st]
            for j, pl in enumerate(PIPELINES):
                pos.append(i * 3 + j)
                data.append(g[f"{metric}_{pl}"].dropna().to_numpy() * scale)
                colors.append(COLOR[pl])
        bp = ax.boxplot(data, positions=pos, widths=0.75, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=MUTED),
                        capprops=dict(color=MUTED))
        for patch, c in zip(bp["boxes"], colors):
            patch.set(facecolor=c, edgecolor=SURFACE, linewidth=2)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(STRUCTURES))], STRUCTURES)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.set_xlim(-0.8, len(STRUCTURES) * 3 - 1.2)
    fig.legend(handles=legend_handles(), loc="upper right", ncol=2)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def crossings_figure(cross, path):
    """Mean crossings per patient by type, per structure."""
    kinds = ["n_interlobar", "n_translobar", "n_unclassified"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, st in zip(axes, STRUCTURES):
        x = np.arange(len(kinds))
        for j, pl in enumerate(PIPELINES):
            s = cross[pl]
            s = s[s.structure == st]
            means = [s[k].mean() for k in kinds]
            bars = ax.bar(x + (j - 0.5) * 0.38, means, 0.36, color=COLOR[pl], edgecolor=SURFACE, linewidth=2)
            for b, m in zip(bars, means):
                ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8, color=MUTED)
        ax.set_xticks(x, ["interlobar", "translobar", "unclassified"])
        ax.set_title(st, loc="left")
    axes[0].set_ylabel("crossing vessels per patient (mean)")
    fig.legend(handles=legend_handles(), loc="upper right", ncol=2)
    fig.suptitle("Crossing vessels per patient, by type", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fissure_curves_figure(curves, path):
    """Share of patients with a crossing running >= x mm past each fissure, VMTK vs field."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    for ax, (fissure, name) in zip(axes, FISSURES.items()):
        for pl in PIPELINES:
            c = curves[pl]
            ax.plot(c.index, c[fissure], color=COLOR[pl], lw=2)
        ax.set_title(f"{fissure} ({name})", loc="left", fontsize=10)
        ax.set_xlabel("length past the fissure (mm)")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("patients (%)")
    fig.legend(handles=legend_handles(), loc="upper right", ncol=2)
    fig.suptitle("Patients with a vessel crossing each fissure, by length past the fissure",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def iqr(s):
    return f"{s.median():.3g} [{s.quantile(.25):.3g}–{s.quantile(.75):.3g}]"


def tables(df, cross, curves, comp):
    out = []
    metrics = [("offset_rel_median", "centering offset / section radius", 1, False),
               ("offset_mm_median", "centering offset (mm)", 1, False),
               ("abs_radius_err_median", "radius error, absolute (%)", 100, False),
               ("radius_err_median", "radius error, signed (%)", 100, None),
               ("outside_frac", "length outside the surface (%)", 100, False),
               ("curvature_median", "curvature (rad/mm)", 1, False),
               ("short_leaf_frac", "terminal spurs (%)", 100, False),
               ("missing_area_pct", "surface not reached (%)", 1, False),
               ("length_mm", "total length (m)", 0.001, None),
               ("groups", "branches (groups)", 1, None),
               ("max_generation", "maximum generation", 1, None),
               ("time_s", "run time per structure (s)", 1, False)]
    for st in STRUCTURES:
        g = df[df.structure == st]
        out.append(f"\n### {st} (n = {len(g)})\n")
        out.append("| metric | VMTK, median [IQR] | field, median [IQR] | field better | Wilcoxon p |")
        out.append("|---|---|---|---|---|")
        for m, name, k, lower_better in metrics:
            v, f = g[f"{m}_vmtk"] * k, g[f"{m}_field"] * k
            ok = v.notna() & f.notna()
            p = wilcoxon(v[ok], f[ok]).pvalue if ok.sum() > 10 and (v[ok] != f[ok]).any() else np.nan
            better = "" if lower_better is None else f"{100 * (f < v)[ok].mean():.0f} %"
            out.append(f"| {name} | {iqr(v)} | {iqr(f)} | {better} | {p:.0e} |")
        out.append(f"\nAgreement: VMTK length matched in field {iqr(g.matched_vmtk)}, field length matched in "
                   f"VMTK {iqr(g.matched_field)}, F1 {iqr(g.matched_f1)}; median distance VMTK→field "
                   f"{iqr(g.dist_median_vmtk_to_field)} mm, p95 {iqr(g.dist_p95_vmtk_to_field)} mm.")
    out.append("\n### Crossings per patient\n")
    out.append("| structure | pipeline | patients | with ≥1 crossing | crossings / patient | interlobar | translobar | unclassified |")
    out.append("|---|---|---|---|---|---|---|---|")
    for st in STRUCTURES:
        for pl in PIPELINES:
            s = cross[pl][cross[pl].structure == st]
            out.append(f"| {st} | {LABEL[pl]} | {s.patient.nunique()} | {100 * (s.n_crossings > 0).mean():.0f} % | "
                       f"{s.n_crossings.mean():.2f} | {s.n_interlobar.mean():.2f} | {s.n_translobar.mean():.2f} | "
                       f"{s.n_unclassified.mean():.2f} |")
    paired = df[["structure", "n_crossings_vmtk", "n_crossings_field"]].dropna()
    out.append("\nSame patient, both pipelines: " + "; ".join(
        f"{st}: presence agrees in {100 * ((g.n_crossings_vmtk > 0) == (g.n_crossings_field > 0)).mean():.0f} %, "
        f"same count in {100 * (g.n_crossings_vmtk == g.n_crossings_field).mean():.0f} %"
        for st, g in paired.groupby("structure")) + ".")
    out.append("\n### Fissure crossing prevalence (% of patients)\n")
    out.append("| fissure | VMTK 0 mm | field 0 mm | VMTK 10 mm | field 10 mm | VMTK 20 mm | field 20 mm |")
    out.append("|---|---|---|---|---|---|---|")
    for fissure in FISSURES:
        row = [f"{curves[pl].loc[x, fissure]:.1f}" for x in (0.0, 10.0, 20.0) for pl in PIPELINES]
        out.append(f"| {fissure} | " + " | ".join(row) + " |")
    out.append("\n### Vessel composition of the crossings (share of each fissure's crossings, at 0 mm)\n")
    out.append("| fissure | vessel | VMTK n | VMTK % | field n | field % |")
    out.append("|---|---|---|---|---|---|")
    for (fissure, vt), r in comp["vmtk"].set_index(["fissure", "vessel_type"]).iterrows():
        f = comp["field"].set_index(["fissure", "vessel_type"]).loc[(fissure, vt)]
        out.append(f"| {fissure} | {vt} | {int(r.n_crossings)} | {r.share_at_0mm_pct:.1f} | {int(f.n_crossings)} | {f.share_at_0mm_pct:.1f} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ext, res = ANALISYS / "extractions", ANALISYS / "results"
    out = res / f"{args.date}_vmtk-vs-field"
    figures = out / "figures"
    if figures.exists() and any(figures.iterdir()) and not args.force:
        raise SystemExit(f"{figures} is not empty (--force)")
    figures.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ext / f"{args.date}_vmtk-vs-field" / "centerline_comparison.csv")
    cross = {pl: pd.read_excel(ext / f"{args.date}_{pl}" / "crossing_statistics.xlsx", "structures") for pl in PIPELINES}
    for pl in PIPELINES:                                  # vessel_type -> the structure names of the comparison
        cross[pl]["structure"] = cross[pl]["structure"].str.replace("final_export_", "")
    curves = {pl: pd.read_csv(res / f"{args.date}_{pl}" / "fissure_length_curves" / "fissure_length_curves.csv",
                              index_col=0) for pl in PIPELINES}
    comp = {pl: pd.read_csv(res / f"{args.date}_{pl}" / "fissure_vessel_composition" / "fissure_vessel_composition.csv")
            for pl in PIPELINES}

    paired_boxes(df, [("offset_rel_median", "offset / section radius", 1, False),
                      ("abs_radius_err_median", "|radius error| (%)", 100, False),
                      ("outside_frac", "length outside the surface (%)", 100, False)],
                 figures / "precision.png", "Local precision (lower is better)")
    paired_boxes(df, [("missing_area_pct", "surface not reached (%)", 1, False),
                      ("length_mm", "total centerline length (m)", 0.001, False),
                      ("max_generation", "maximum generation", 1, False)],
                 figures / "completeness.png", "Completeness")
    paired_boxes(df, [("curvature_median", "curvature (rad/mm)", 1, False),
                      ("short_leaf_frac", "terminal spurs (%)", 100, False),
                      ("time_s", "run time per structure (s)", 1, True)],
                 figures / "quality_and_time.png", "Smoothness, spurs and run time (lower is better)")
    agreement_figure(df, figures / "agreement.png")
    crossings_figure(cross, figures / "crossings_per_patient.png")
    fissure_curves_figure(curves, figures / "fissure_length_curves.png")
    (out / "tables.md").write_text(tables(df, cross, curves, comp))
    print(f"-> {out}")


def agreement_figure(df, path):
    """One box per structure: these metrics belong to the pair, not to one pipeline."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    for ax, (m, ylabel, k) in zip(axes, [("matched_vmtk", "VMTK length within field (%)", 100),
                                         ("matched_field", "field length within VMTK (%)", 100),
                                         ("dist_median_vmtk_to_field", "median distance VMTK → field (mm)", 1)]):
        bp = ax.boxplot([df[df.structure == st][m].dropna() * k for st in STRUCTURES], widths=0.55,
                        patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.5),
                        whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        for patch in bp["boxes"]:
            patch.set(facecolor="#a7a6a0", edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(range(1, len(STRUCTURES) + 1), STRUCTURES)
        ax.set_ylabel(ylabel)
    fig.suptitle("Agreement between the two centerlines (one value per patient and structure)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
