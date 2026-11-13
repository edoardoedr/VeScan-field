#!/usr/bin/env python3
"""For each of the four fissures: of every crossing that exists on it, what
share is vein, what share is artery, what share is airway?

    python3 statistics_scripts/plots/fissure_vessel_composition.py -i analisys/extractions/<extraction>/crossing_statistics.xlsx
            [--datasets NAME ...] [-o outdir] [--force]

By default the results go to analisys/results/<extraction>/fissure_vessel_composition/, and an existing non-empty
folder is never overwritten (--force). --datasets keeps only some datasets of the workbook: an
extraction holding both pipelines (luna16_vmtk, luna16_field, ...) must be analysed one
pipeline at a time, or every patient counts twice.

UNIT = THE CROSSING, not the patient. A patient with 5 artery crossings on a
fissure contributes 5 to the artery count there, not 1 - this is deliberately
different from fissure_length_curves.py / fissure_length_curves_by_vessel.py,
which count each patient once.

For fissure B and tree T:

    numerator(B, T)   = number of `crossings` rows mapped to boundary B with
                        vessel_type == T (every length, including rows with no
                        length at all - this bar does not care how far the
                        vessel continues, only that it crosses).
    denominator(B)    = number of `crossings` rows mapped to boundary B, of
                        ANY tree (vein + artery + airway) - the total traffic
                        through that fissure. Fixed per fissure; the same
                        number for all three bars in its group.
    bar height        = 100 * numerator(B, T) / denominator(B)

No patient-cohort restriction here (unlike the other two scripts): the
question is about crossings summed over every patient in the sheet, so nobody
is excluded for missing a structure.

THE ERROR BAR is the range that share can take as the length requirement used
elsewhere in this analysis is swept from 0 to 20 mm. Concretely:

    bar top   = share with NO length requirement (threshold 0 mm) - every
                mapped crossing of that tree counts, which is the maximum
                the share can be.
    bar bottom (whisker) = share counting only crossings that reach at least
                20 mm past the fissure (threshold 20 mm) - the strictest cut
                used anywhere in this analysis, which is the minimum the share
                can be.

The denominator does NOT change between the two - only the numerator is
length-filtered - so the top is always >= the bottom by construction, and
every intermediate threshold's share (0 to 20 mm, not otherwise plotted) has
to fall inside the whisker. That is what "our max and min possible percentage
bars" means: the bar height is the 0 mm case, the whisker foot is the 20 mm
case, together bounding every threshold in between.

FOUR FISSURES, FROM UNORDERED LOBE PAIRS - identical mapping to the other two
scripts in this folder:

    {RUL, RML} -> R-HIF     right horizontal (minor) fissure
    {RUL, RLL} -> R-UOIF    right oblique fissure, upper portion
    {RML, RLL} -> R-LOIF    right oblique fissure, lower portion
    {LUL, LLL} -> L-major   the single left fissure

WHAT GETS DROPPED before a row can be counted at all (denominator and every
numerator alike) - only what is needed to assign one fissure unambiguously.
`crossing_type` plays no role here, deliberately: this script counts every
crossing regardless of whether it is interlobar, translobar or unclassified,
since the question is "does a crossing exist on this fissure", not "is it
resolved to a specific pattern". In practice an unclassified row almost never
carries a lobe pair, so it drops out on its own at the next rule anyway.

  * a missing `from_lobe` or `to_lobe`.
  * multi-lobe endpoints (`RLL+RUL`, `LLL+LUL`, ...): the vessel touches more
    than one lobe at that end, so the fissure is ambiguous.
  * same-lobe pairs (`RML -> RML`) - no fissure is crossed.
  * cross-lung pairs (`RUL -> LUL`) - not one of the four surfaces.

`length_after_piercing_mm` is never a reason to drop a row - a crossing with
no recorded length, or a non-positive one, still counts toward the 0 mm bar
(it exists); it simply cannot satisfy the >= 20 mm cut for the whisker foot.

Each filtering step reports how many rows it removed, so the counts behind the
chart stay auditable.

Writes two files into the output directory:

    fissure_vessel_composition.png   the plot
    fissure_vessel_composition.csv   one row per fissure x tree, with the
                                     numerator, the fixed denominator, and the
                                     0 mm / 20 mm shares (%)

Requires pandas, openpyxl and matplotlib.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

RESULTS = HERE.parents[1] / "analisys" / "results"       # analisys/results/<extraction>/<script>/
DATASETS = None                        # --datasets: the datasets kept (None = all)


def select_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """Only the rows of the --datasets chosen."""
    return df if DATASETS is None else df[df["dataset"].isin(DATASETS)]

# Unordered lobe pair -> fissure label. frozenset keys so that a crossing
# recorded as RUL -> RML and one recorded as RML -> RUL hit the same entry.
FISSURES = {
    frozenset({"RUL", "RML"}): "R-HIF",
    frozenset({"RUL", "RLL"}): "R-UOIF",
    frozenset({"RML", "RLL"}): "R-LOIF",
    frozenset({"LUL", "LLL"}): "L-major",
}
FISSURE_ORDER = ["R-HIF", "R-UOIF", "R-LOIF", "L-major"]

# The three trees, in fixed plotting order - color follows the tree, never its
# rank, so a series keeps its hue even if one tree is thin at some fissure.
# Slots 1/2/3 of the reference categorical palette - the three that validate
# all-pairs (not just adjacent), the right check for a grouped bar chart.
VESSEL_ORDER = ["artery", "vein", "airway"]
VESSEL_COLORS = {
    "artery": "#2a78d6",
    "vein": "#eb6834",
    "airway": "#1baf7a",
}

MAX_THRESHOLD_MM = 20.0

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


def load_crossings(path: Path) -> pd.DataFrame:
    """Reads the `crossings` sheet - one row per crossing vessel."""
    df = pd.read_excel(path, sheet_name="crossings")
    df = select_datasets(df)
    print(f"read {len(df)} crossings")
    return df


def derive_boundary(df: pd.DataFrame) -> pd.DataFrame:
    """Assigns each row to exactly one of the four fissures, or drops it.

    `length_after_piercing_mm` and `crossing_type` are untouched here - see
    the module docstring for why neither gates inclusion.
    """
    n = len(df)

    for col in ("from_lobe", "to_lobe"):
        df[col] = df[col].astype("string").str.strip().str.upper()
    df["vessel_type"] = df["vessel_type"].astype("string").str.strip().str.lower()

    df = df.dropna(subset=["from_lobe", "to_lobe"])
    print(f"  complete lobes: dropped {n - len(df)}")
    n = len(df)

    ambiguous = df["from_lobe"].str.contains(r"\+") | df["to_lobe"].str.contains(r"\+")
    df = df[~ambiguous]
    print(f"  single-lobe endpoints: dropped {n - len(df)}")
    n = len(df)

    pair = df.apply(lambda r: frozenset({r["from_lobe"], r["to_lobe"]}), axis=1)
    df = df.assign(fissure=pair.map(FISSURES))
    unmapped = df[df["fissure"].isna()]
    if not unmapped.empty:
        print(
            "  unmapped lobe pairs: dropped "
            f"{len(unmapped)} ({sorted(unmapped['lobe_pair'].dropna().unique())})"
        )
    df = df.dropna(subset=["fissure"])

    print(f"  kept {len(df)} crossings mapped to a fissure")
    return df


def composition(df: pd.DataFrame) -> pd.DataFrame:
    """One row per fissure x tree: numerator, fixed denominator, 0/20 mm shares.

    denom(B) is the same for every tree within a fissure - the total crossing
    traffic through it, vein + artery + airway, no length filter.
    """
    denom = df.groupby("fissure").size()

    all_counts = df.groupby(["fissure", "vessel_type"]).size()
    long_counts = (
        df[df["length_after_piercing_mm"] >= MAX_THRESHOLD_MM]
        .groupby(["fissure", "vessel_type"])
        .size()
    )

    rows = []
    for fissure in FISSURE_ORDER:
        d = int(denom.get(fissure, 0))
        for vessel in VESSEL_ORDER:
            n_all = int(all_counts.get((fissure, vessel), 0))
            n_long = int(long_counts.get((fissure, vessel), 0))
            rows.append(
                {
                    "fissure": fissure,
                    "vessel_type": vessel,
                    "n_crossings": n_all,
                    "n_crossings_ge_20mm": n_long,
                    "denominator": d,
                    "share_at_0mm_pct": 100.0 * n_all / d if d else np.nan,
                    "share_at_20mm_pct": 100.0 * n_long / d if d else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot(comp: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    n_groups = len(FISSURE_ORDER)
    n_series = len(VESSEL_ORDER)
    group_width = 0.72
    bar_width = group_width / n_series
    group_centers = np.arange(n_groups)

    for i, vessel in enumerate(VESSEL_ORDER):
        sub = comp[comp["vessel_type"] == vessel].set_index("fissure").loc[FISSURE_ORDER]
        offset = (i - (n_series - 1) / 2) * bar_width
        x = group_centers + offset

        top = sub["share_at_0mm_pct"].to_numpy()
        bottom = sub["share_at_20mm_pct"].to_numpy()
        # Whisker foot at the 20 mm share, nothing drawn above the bar top -
        # the bar itself already marks the 0 mm maximum.
        yerr = np.vstack([top - bottom, np.zeros_like(top)])

        ax.bar(
            x,
            top,
            width=bar_width * 0.92,
            color=VESSEL_COLORS[vessel],
            label=vessel,
            yerr=yerr,
            ecolor=INK_SECONDARY,
            capsize=3,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(FISSURE_ORDER)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Share of that fissure's crossings (%)", color=INK_SECONDARY)
    ax.set_title(
        "Vessel-type composition of crossings, per fissure",
        color=INK_PRIMARY,
        fontsize=13,
        loc="left",
        pad=36,
    )
    ax.annotate(
        "Bar = share with no length requirement (max); whisker foot = share among\n"
        "crossings reaching ≥20 mm past the fissure (min) - one point per tree per fissure",
        xy=(0, 1.015),
        xycoords="axes fraction",
        va="bottom",
        fontsize=9,
        color=INK_SECONDARY,
    )

    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
    ax.tick_params(colors=INK_SECONDARY, length=0)

    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", type=Path, required=True,
                    help="aggregated workbook, analisys/extractions/<extraction>/crossing_statistics.xlsx")
    ap.add_argument("--datasets", nargs="+", help="only these datasets (the 'dataset' column)")
    ap.add_argument("--force", action="store_true", help="write into a non-empty output folder")
    ap.add_argument("-o", "--outdir", type=Path, default=None,
                    help="where the .png and .csv go (default: analisys/results/<extraction>/fissure_vessel_composition/, "
                         "with a -<datasets> suffix when --datasets is given)")
    args = ap.parse_args()

    global DATASETS
    DATASETS = args.datasets
    if args.outdir is None:
        suffix = "-" + "+".join(args.datasets) if args.datasets else ""
        args.outdir = RESULTS / args.input.resolve().parent.name / f"fissure_vessel_composition{suffix}"
    if args.outdir.exists() and any(args.outdir.iterdir()) and not args.force:
        raise SystemExit(f"{args.outdir} is not empty: choose another -o or use --force")
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"results -> {args.outdir}")

    df = derive_boundary(load_crossings(args.input))
    comp = composition(df)

    print(comp.round(1).to_string(index=False))

    out_csv = args.outdir / "fissure_vessel_composition.csv"
    comp.to_csv(out_csv, index=False, float_format="%.2f")
    print(f"wrote {out_csv}")

    plot(comp, args.outdir / "fissure_vessel_composition.png")


if __name__ == "__main__":
    main()
