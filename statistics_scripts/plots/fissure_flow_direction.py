#!/usr/bin/env python3
"""For each fissure and each tree, which direction does the crossing traffic
mostly run in, and how lopsided is it?

    python3 statistics_scripts/plots/fissure_flow_direction.py -i analisys/extractions/<extraction>/crossing_statistics.xlsx
            [--datasets NAME ...] [-o outdir] [--force]

By default the results go to analisys/results/<extraction>/fissure_flow_direction/, and an existing non-empty
folder is never overwritten (--force). --datasets keeps only some datasets of the workbook: an
extraction holding both pipelines (luna16_vmtk, luna16_field, ...) must be analysed one
pipeline at a time, or every patient counts twice.

THE DISTANCE IS A PARAMETER, same `length_after_piercing_mm` cut used
throughout this folder, and the script runs the whole analysis once per value
in MIN_LENGTHS_MM = [0, 10, 20] - three independent snapshots, not a sweep. At
threshold 0 mm every crossing counts, no matter what its length is (or
whether it has one at all) - the direction-only analysis this script started
as. At 10 and 20 mm, only crossings that still reach at least that far past
the fissure count, so both n_dominant and n_opposite shrink together and a
tree with mostly short crossings can end up with too few left at 20 mm to say
much about its direction at all - the panel says so when that happens. Each
threshold writes its own numbered PNG and CSV; nothing is overwritten.

UNIT = THE CROSSING, same as fissure_vessel_composition.py, not the patient: a
patient with 5 crossings running the same way contributes 5, not 1.

HEMODYNAMIC DIRECTION, NOT THE RAW GRAPH DIRECTION. `from_lobe -> to_lobe` as
stored is the vessel TREE's own traversal direction (root to leaf along that
tree), and for arteries and airways that already runs the way blood/air
actually moves - heart/hilum outward to the periphery. For VEINS it is the
reverse: blood in a vein flows periphery -> hilum -> heart, opposite to how
the tree was traversed to record the row. So the true flow direction,
`flow_src -> flow_dst`, is:

    artery, airway:  flow_src, flow_dst = from_lobe, to_lobe   (unchanged)
    vein:            flow_src, flow_dst = to_lobe, from_lobe   (swapped)

Everything below counts this corrected direction, never the raw one.

NUMERATOR, PER (fissure, tree) PAIR. A fissure joins exactly two lobes, so
`flow_src -> flow_dst` can only take one of two values - lobe A -> lobe B, or
the reverse. Within a (fissure, tree) pair:

    n_dominant  = count of whichever of the two directions is more common
    n_opposite  = count of the other one

A tie is broken toward the alphabetically-first lobe as source, only so the
choice is reproducible - it does not change n_dominant or n_opposite.

DENOMINATOR = n_dominant + n_opposite, i.e. every crossing recorded for that
(fissure, tree) pair - not every patient, only the crossings that exist
there, same accounting as fissure_vessel_composition.py.

FOUR FISSURES, FROM UNORDERED LOBE PAIRS - identical mapping to the other
scripts in this folder:

    {RUL, RML} -> R-HIF     right horizontal (minor) fissure
    {RUL, RLL} -> R-UOIF    right oblique fissure, upper portion
    {RML, RLL} -> R-LOIF    right oblique fissure, lower portion
    {LUL, LLL} -> L-major   the single left fissure

WHAT GETS DROPPED, same rule as fissure_vessel_composition.py and for the same
reason - only what is needed to assign one fissure unambiguously. Direction by
itself does not depend on `crossing_type` or `length_after_piercing_mm`, so
neither gates inclusion at this stage - the length cut is applied afterwards,
once per threshold, described above:

  * a missing `from_lobe` or `to_lobe`.
  * multi-lobe endpoints (`RLL+RUL`, `LLL+LUL`, ...) - the fissure is
    ambiguous.
  * same-lobe pairs (`RML -> RML`) - no fissure is crossed.
  * cross-lung pairs (`RUL -> LUL`) - not one of the four surfaces.

Each filtering step reports how many rows it removed, so the counts behind the
chart stay auditable.

MAIN CHART: a butterfly (diverging) bar chart, one small-multiple panel per
fissure. Inside a panel, one horizontal bar per tree, zero at the centre,
n_opposite drawn to the left and n_dominant to the right - the standard way to
show a binary asymmetry, so both the direction and the imbalance read at a
glance. Each panel keeps its own x scale (fissures differ hugely in traffic -
R-HIF alone carries more crossings than the other three combined), so a bar's
width is only compared against its own panel, never across panels; the
dominant destination lobe is labelled at the right tip and the opposite one at
the left tip, since which lobe is "dominant" can differ tree by tree on the
same fissure.

Writes, per threshold t in MIN_LENGTHS_MM, into the output directory:

    fissure_flow_direction_{t}mm.png   the plot at that threshold
    fissure_flow_direction_{t}mm.csv   one row per fissure x tree: n_dominant,
                                       n_opposite, denominator, the two
                                       directions as text, and the dominant
                                       share (%), all computed at that
                                       threshold

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

# Unordered lobe pair -> fissure label, and the reverse - both keyed the same
# way as the other scripts in this folder.
FISSURES = {
    frozenset({"RUL", "RML"}): "R-HIF",
    frozenset({"RUL", "RLL"}): "R-UOIF",
    frozenset({"RML", "RLL"}): "R-LOIF",
    frozenset({"LUL", "LLL"}): "L-major",
}
FISSURE_ORDER = ["R-HIF", "R-UOIF", "R-LOIF", "L-major"]
# The two lobes of each fissure, alphabetically - only to make tie-breaking
# and axis labelling reproducible, not anatomically meaningful.
FISSURE_LOBES = {label: tuple(sorted(pair)) for pair, label in FISSURES.items()}

# The three trees, in fixed plotting order - color follows the tree, never its
# rank. Same slots (and roles) as fissure_vessel_composition.py, so a tree
# keeps its identity color across both charts.
VESSEL_ORDER = ["artery", "vein", "airway"]
VESSEL_COLORS = {
    "artery": "#2a78d6",
    "vein": "#eb6834",
    "airway": "#1baf7a",
}
VEIN_INVERTED = {"vein"}

# Three fixed snapshots, not a sweep - see the module docstring.
MIN_LENGTHS_MM = [0.0, 10.0, 20.0]

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

    Same rules as fissure_vessel_composition.py - see the module docstring.
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


def add_flow_direction(df: pd.DataFrame) -> pd.DataFrame:
    """flow_src -> flow_dst: the hemodynamic direction, veins swapped."""
    is_vein = df["vessel_type"].isin(VEIN_INVERTED)
    df = df.assign(
        flow_src=np.where(is_vein, df["to_lobe"], df["from_lobe"]),
        flow_dst=np.where(is_vein, df["from_lobe"], df["to_lobe"]),
    )
    df["pair_flow"] = df["flow_src"] + " -> " + df["flow_dst"]
    return df


def apply_min_length(df: pd.DataFrame, min_length_mm: float) -> pd.DataFrame:
    """Keeps only crossings reaching at least `min_length_mm` past the fissure.

    At 0 mm this is a no-op - every row counts, including one with no recorded
    length at all - matching the "no requirement" reading used throughout this
    folder. Above 0 mm it is a plain `>=` filter, which naturally excludes a
    missing length too (NaN >= x is False), on the same footing as a row that
    is simply too short.
    """
    if min_length_mm <= 0:
        return df
    n = len(df)
    df = df[df["length_after_piercing_mm"] >= min_length_mm]
    print(f"  length_after_piercing_mm >= {min_length_mm:g}mm: dropped {n - len(df)}")
    return df


def direction_counts(df: pd.DataFrame) -> pd.DataFrame:
    """One row per fissure x tree: n_dominant, n_opposite and their labels."""
    counts = df.groupby(["fissure", "vessel_type", "pair_flow"]).size()

    rows = []
    for fissure in FISSURE_ORDER:
        lobe_a, lobe_b = FISSURE_LOBES[fissure]
        flow_ab, flow_ba = f"{lobe_a} -> {lobe_b}", f"{lobe_b} -> {lobe_a}"
        for vessel in VESSEL_ORDER:
            n_ab = int(counts.get((fissure, vessel, flow_ab), 0))
            n_ba = int(counts.get((fissure, vessel, flow_ba), 0))
            if n_ab >= n_ba:
                n_dominant, n_opposite = n_ab, n_ba
                dominant_flow, opposite_flow = flow_ab, flow_ba
            else:
                n_dominant, n_opposite = n_ba, n_ab
                dominant_flow, opposite_flow = flow_ba, flow_ab
            denom = n_dominant + n_opposite
            rows.append(
                {
                    "fissure": fissure,
                    "vessel_type": vessel,
                    "dominant_flow": dominant_flow,
                    "opposite_flow": opposite_flow,
                    "n_dominant": n_dominant,
                    "n_opposite": n_opposite,
                    "denominator": denom,
                    "dominant_share_pct": 100.0 * n_dominant / denom if denom else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot(dc: pd.DataFrame, min_length_mm: float, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), facecolor=SURFACE)
    length_clause = (
        "every crossing, no minimum length"
        if min_length_mm <= 0
        else f"crossings reaching ≥{min_length_mm:g} mm past the fissure only"
    )
    fig.suptitle(
        f"Which way crossing traffic runs, per fissure and tree ({length_clause})",
        x=0.02,
        ha="left",
        fontsize=13.5,
        color=INK_PRIMARY,
    )
    fig.text(
        0.02,
        0.935,
        "Dominant direction (hemodynamic - veins reversed) to the right, opposite to the left;\n"
        "each panel has its own scale - fissures differ hugely in total traffic",
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK_SECONDARY,
    )

    y_pos = np.arange(len(VESSEL_ORDER))[::-1]  # artery on top, airway on bottom

    for ax, fissure in zip(axes.flat, FISSURE_ORDER):
        sub = dc[dc["fissure"] == fissure].set_index("vessel_type").loc[VESSEL_ORDER]

        rights = sub["n_dominant"].to_numpy(dtype=float)
        lefts = -sub["n_opposite"].to_numpy(dtype=float)
        colors = [VESSEL_COLORS[v] for v in VESSEL_ORDER]

        ax.barh(y_pos, rights, color=colors, height=0.55, zorder=3)
        ax.barh(y_pos, lefts, color=colors, height=0.55, alpha=0.45, zorder=3)
        ax.axvline(0, color=INK_SECONDARY, linewidth=1.0, zorder=2)

        span = max(1.0, float(max(rights.max(), -lefts.min())))
        pad = span * 0.42
        ax.set_xlim(-span - pad, span + pad)

        for y, vessel in zip(y_pos, VESSEL_ORDER):
            row = sub.loc[vessel]
            n_dom, n_opp = int(row["n_dominant"]), int(row["n_opposite"])
            dom_dst = row["dominant_flow"].split(" -> ")[1]
            opp_dst = row["opposite_flow"].split(" -> ")[1]

            if n_dom > 0:
                ax.annotate(
                    f"→ {dom_dst} ({n_dom})",
                    xy=(n_dom, y),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    color=INK_PRIMARY,
                )
            if n_dom == 0 and n_opp == 0:
                # Nothing survived this threshold for this tree - say so
                # rather than leave a silently empty row.
                ax.annotate(
                    "no crossings ≥ this length",
                    xy=(0, y),
                    xytext=(0, 0),
                    textcoords="offset points",
                    va="center",
                    ha="center",
                    fontsize=8,
                    style="italic",
                    color=INK_SECONDARY,
                )
            if n_opp > 0:
                ax.annotate(
                    f"({n_opp}) {opp_dst} ←",
                    xy=(-n_opp, y),
                    xytext=(-4, 0),
                    textcoords="offset points",
                    va="center",
                    ha="right",
                    fontsize=8.5,
                    color=INK_SECONDARY,
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"{v} (n={int(sub.loc[v, 'denominator'])})" for v in VESSEL_ORDER],
            color=INK_SECONDARY,
        )
        ax.set_xticks([])
        lobe_a, lobe_b = FISSURE_LOBES[fissure]
        ax.set_title(f"{fissure}  ({lobe_a} ↔ {lobe_b})", loc="left", fontsize=11, color=INK_PRIMARY)

        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#d8d7d2")
        ax.tick_params(length=0)
        ax.set_ylim(y_pos.min() - 0.6, y_pos.max() + 0.6)

    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
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
                    help="where the .png and .csv go (default: analisys/results/<extraction>/fissure_flow_direction/, "
                         "with a -<datasets> suffix when --datasets is given)")
    args = ap.parse_args()

    global DATASETS
    DATASETS = args.datasets
    if args.outdir is None:
        suffix = "-" + "+".join(args.datasets) if args.datasets else ""
        args.outdir = RESULTS / args.input.resolve().parent.name / f"fissure_flow_direction{suffix}"
    if args.outdir.exists() and any(args.outdir.iterdir()) and not args.force:
        raise SystemExit(f"{args.outdir} is not empty: choose another -o or use --force")
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"results -> {args.outdir}")

    base = add_flow_direction(derive_boundary(load_crossings(args.input)))

    for min_length_mm in MIN_LENGTHS_MM:
        tag = f"{min_length_mm:g}mm"
        print(f"--- threshold {tag} ---")

        df = apply_min_length(base, min_length_mm)
        dc = direction_counts(df)
        print(dc.round(1).to_string(index=False))

        out_csv = args.outdir / f"fissure_flow_direction_{tag}.csv"
        dc.to_csv(out_csv, index=False, float_format="%.2f")
        print(f"wrote {out_csv}")

        plot(dc, min_length_mm, args.outdir / f"fissure_flow_direction_{tag}.png")


if __name__ == "__main__":
    main()
