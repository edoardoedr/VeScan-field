#!/usr/bin/env python3
"""Same question as fissure_length_curves.py, split one chart per vessel tree:
for arteries, for veins and for airways separately, how often does each of the
four fissures carry a vessel OF THAT TREE running at least x mm past it, over
the first 20 mm.

    python3 statistics_scripts/plots/fissure_length_curves_by_vessel.py -i analisys/extractions/<extraction>/crossing_statistics.xlsx
            [--datasets NAME ...] [-o outdir] [--force]

By default the results go to analisys/results/<extraction>/fissure_length_curves_by_vessel/, and an existing non-empty
folder is never overwritten (--force). --datasets keeps only some datasets of the workbook: an
extraction holding both pipelines (luna16_vmtk, luna16_field, ...) must be analysed one
pipeline at a time, or every patient counts twice.

The unit of the y axis is still the PATIENT: at a given x, a patient counts
once if THAT TREE has a crossing of that fissure whose `length_after_piercing_mm`
is >= x, zero otherwise. Unlike fissure_length_curves.py, trees are NOT pooled
here - a vein crossing does not qualify a patient on the artery chart. That is
the whole point of splitting: it isolates whether a given tree tends to reach
past a fissure, tree by tree.

DENOMINATOR IS TREE-SPECIFIC, NOT THE FULL COHORT. Each of the three trees was
extracted for a different number of patients (airway 68, artery 67, vein 67 -
one segmentation failed on the artery tree, one on the vein tree, both
succeeded for airway - see `structures`). Using the same denominator for all
three trees would either overcount patients who never had that tree extracted
in the vein/artery chart, or undercount airway's true denominator. So each
chart's denominator is `structures[structures.vessel_type == T].patient`
counted distinct - the patients that tree was actually extracted for, whether
or not it produced any crossing. That sheet is the only place a
zero-crossing patient still leaves a row; `crossings` would silently drop them
and inflate the prevalence.

Patient ids are unique only within a dataset, so every patient count is keyed
on (dataset, patient) - keying on the id alone would merge two different
patients who happen to share one.

FOUR FISSURES, FROM UNORDERED LOBE PAIRS - identical mapping to
fissure_length_curves.py:

    {RUL, RML} -> R-HIF     right horizontal (minor) fissure
    {RUL, RLL} -> R-UOIF    right oblique fissure, upper portion
    {RML, RLL} -> R-LOIF    right oblique fissure, lower portion
    {LUL, LLL} -> L-major   the single left fissure

WHAT GETS DROPPED from the numerator (same rules as fissure_length_curves.py,
applied per vessel_type instead of pooled):

  * rows of a different vessel_type than the chart being built.
  * `crossing_type` is kept whatever it is (interlobar, translobar or
    unclassified) - it plays no filtering role by itself. An `unclassified`
    row is dropped anyway by the next rule: it almost never carries a lobe
    pair or a piercing length, since it is a crossing the pipeline could not
    resolve rather than a crossing of length zero.
  * a missing `from_lobe`, `to_lobe` or `length_after_piercing_mm`.
  * multi-lobe endpoints (`RLL+RUL`, `LLL+LUL`, ...) - the fissure is
    ambiguous, so there is no honest way to assign one curve.
  * same-lobe pairs (`RML -> RML`) - no fissure is crossed.
  * cross-lung pairs (`RUL -> LUL`) - not one of the four surfaces.
  * crossings belonging to a patient that tree was not extracted for, so
    numerator and denominator are drawn from exactly the same patient set.

Each filtering step reports how many rows it removed, so the numbers behind
the curves stay auditable.

Writes, per vessel type T in {artery, vein, airway}, into the output
directory:

    fissure_length_curves_T.png   the plot
    fissure_length_curves_T.csv   the same curves as numbers (x, one column
                                  per fissure)

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

# Patient identity is the pair, never the id alone - see the module docstring.
PATIENT_KEY = ["dataset", "patient"]

# The three trees to chart, in the order they are drawn.
VESSEL_TYPES = ["artery", "vein", "airway"]

# Unordered lobe pair -> fissure label. frozenset keys so that a crossing
# recorded as RUL -> RML and one recorded as RML -> RUL hit the same entry.
FISSURES = {
    frozenset({"RUL", "RML"}): "R-HIF",
    frozenset({"RUL", "RLL"}): "R-UOIF",
    frozenset({"RML", "RLL"}): "R-LOIF",
    frozenset({"LUL", "LLL"}): "L-major",
}

# Fixed plotting order and fixed colors: the color follows the fissure, never
# its rank, so a curve keeps its hue even if one of the four is ever missing.
# Slots 1/2/3/7 of the reference categorical palette - validated all-pairs.
FISSURE_ORDER = ["R-HIF", "R-UOIF", "R-LOIF", "L-major"]
FISSURE_COLORS = {
    "R-HIF": "#2a78d6",
    "R-UOIF": "#eb6834",
    "R-LOIF": "#1baf7a",
    "L-major": "#4a3aa7",
}

X_MAX_MM = 20.0
X_STEP_MM = 0.1

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


def tree_denominators(path: Path) -> dict[str, pd.MultiIndex]:
    """Per vessel_type, the distinct patients that tree was extracted for.

    Read from `structures`, which carries a row per patient x structure even
    when that structure produced zero crossings - the only honest denominator.
    """
    df = pd.read_excel(path, sheet_name="structures")
    df = select_datasets(df)
    df["vessel_type"] = df["vessel_type"].astype("string").str.strip().str.lower()

    out: dict[str, pd.MultiIndex] = {}
    print("per-tree cohort, from `structures`:")
    for vt in VESSEL_TYPES:
        patients = df.loc[df["vessel_type"] == vt, PATIENT_KEY].drop_duplicates()
        idx = pd.MultiIndex.from_frame(patients)
        out[vt] = idx
        print(f"  {vt}: {len(idx)} patients")
    return out


def load_crossings(path: Path) -> pd.DataFrame:
    """Reads the `crossings` sheet - one row per crossing vessel."""
    df = pd.read_excel(path, sheet_name="crossings")
    df = select_datasets(df)
    print(f"read {len(df)} crossings")
    df["vessel_type"] = df["vessel_type"].astype("string").str.strip().str.lower()
    for col in ("from_lobe", "to_lobe"):
        df[col] = df[col].astype("string").str.strip().str.upper()
    return df


def clean_for_tree(df: pd.DataFrame, vessel_type: str, cohort: pd.MultiIndex) -> pd.DataFrame:
    """Keeps the rows of one tree attributable to exactly one fissure, in a
    patient that tree was extracted for. Every step prints what it removed.
    """
    print(f"filtering for vessel_type={vessel_type!r}:")

    df = df[df["vessel_type"] == vessel_type]
    n = len(df)
    print(f"  vessel_type == {vessel_type!r}: {n} rows")

    # 1. the patient must be one this tree was actually extracted for.
    df = df[pd.MultiIndex.from_frame(df[PATIENT_KEY]).isin(cohort)]
    print(f"  patients with this tree extracted: dropped {n - len(df)}")
    n = len(df)

    # 2. the three fields the plot actually reads must all be present. Empty
    #    cells arrive as NaN or as an empty string; both are dropped.
    df = df.replace({"from_lobe": {"": pd.NA}, "to_lobe": {"": pd.NA}})
    df = df.dropna(subset=["from_lobe", "to_lobe", "length_after_piercing_mm"])
    print(f"  complete lobes + length: dropped {n - len(df)}")
    n = len(df)

    # 3. an endpoint naming several lobes ("RLL+RUL") leaves the fissure
    #    ambiguous - drop rather than guess.
    ambiguous = df["from_lobe"].str.contains(r"\+") | df["to_lobe"].str.contains(r"\+")
    df = df[~ambiguous]
    print(f"  single-lobe endpoints: dropped {n - len(df)}")
    n = len(df)

    # 4. map the unordered pair onto a fissure. Same-lobe pairs collapse to a
    #    one-element frozenset and cross-lung pairs are simply not in the
    #    table, so both fall out here as unmapped.
    pair = df.apply(lambda r: frozenset({r["from_lobe"], r["to_lobe"]}), axis=1)
    df = df.assign(fissure=pair.map(FISSURES))
    unmapped = df[df["fissure"].isna()]
    if not unmapped.empty:
        print(
            "  unmapped lobe pairs: dropped "
            f"{len(unmapped)} ({sorted(unmapped['lobe_pair'].dropna().unique())})"
        )
    df = df.dropna(subset=["fissure"])

    # 5. a non-positive distal length would be a zero-length remnant, not a
    #    vessel running past the fissure.
    n = len(df)
    df = df[df["length_after_piercing_mm"] > 0]
    print(f"  positive length: dropped {n - len(df)}")

    print(f"  kept {len(df)} crossings in {df.groupby(PATIENT_KEY).ngroups} patients")
    return df


def patient_reach(df: pd.DataFrame) -> pd.DataFrame:
    """Longest crossing per patient x fissure - the whole numerator in one table.

    A patient counts at threshold x exactly when their LONGEST crossing of
    that fissure (in this tree) reaches x, so collapsing to the maximum loses
    nothing and makes the count a single comparison per patient.
    """
    return (
        df.groupby(PATIENT_KEY + ["fissure"])["length_after_piercing_mm"]
        .max()
        .unstack("fissure")
    )


def curves(reach: pd.DataFrame, n_patients: int) -> pd.DataFrame:
    """Percentage of the tree's cohort with a qualifying crossing, per fissure
    per x. Returned frame is indexed by x (mm) with one column per fissure, in
    FISSURE_ORDER - exactly what gets plotted and written to CSV.
    """
    x = np.round(np.arange(0.0, X_MAX_MM + X_STEP_MM / 2, X_STEP_MM), 3)
    out = pd.DataFrame(index=pd.Index(x, name="length_after_piercing_mm"))

    for fissure in FISSURE_ORDER:
        if fissure not in reach.columns:
            continue
        # Patients with no crossing of this fissure are NaN here; dropping
        # them removes them from the numerator while the denominator stays
        # the full tree cohort, which is the point of the prevalence.
        longest = reach[fissure].dropna().to_numpy()
        at_least = longest.size - np.searchsorted(np.sort(longest), x, side="left")
        out[fissure] = 100.0 * at_least / n_patients if n_patients else np.nan

    return out


def plot(curve_df: pd.DataFrame, vessel_type: str, n_patients: int, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for fissure in curve_df.columns:
        ax.plot(
            curve_df.index,
            curve_df[fissure],
            color=FISSURE_COLORS[fissure],
            linewidth=2.0,
            solid_capstyle="round",
            label=fissure,
        )
        # Direct label at the right edge as well as the legend: identity is
        # never carried by color alone, and it is the relief the palette's
        # low-contrast slot requires.
        ax.annotate(
            fissure,
            xy=(curve_df.index[-1], curve_df[fissure].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=FISSURE_COLORS[fissure],
        )

    ax.set_xlim(0, X_MAX_MM)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, X_MAX_MM + 1, 2))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("Minimum length past the fissure contact point (mm)", color=INK_SECONDARY)
    ax.set_ylabel("Patients with such a crossing (%)", color=INK_SECONDARY)
    ax.set_title(
        f"How often a fissure carries a {vessel_type} running this far beyond it",
        color=INK_PRIMARY,
        fontsize=13,
        loc="left",
        # Room above the axes for the two-line subtitle the title sits on top of.
        pad=36,
    )
    ax.annotate(
        f"Share of the {n_patients} patients with a {vessel_type} tree extracted having at "
        f"least one {vessel_type}\ncrossing that fissure and extending at least x mm past it",
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

    # Room on the right for the direct labels.
    fig.tight_layout()
    fig.subplots_adjust(right=0.88)
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
                    help="where the .png and .csv go (default: analisys/results/<extraction>/fissure_length_curves_by_vessel/, "
                         "with a -<datasets> suffix when --datasets is given)")
    args = ap.parse_args()

    global DATASETS
    DATASETS = args.datasets
    if args.outdir is None:
        suffix = "-" + "+".join(args.datasets) if args.datasets else ""
        args.outdir = RESULTS / args.input.resolve().parent.name / f"fissure_length_curves_by_vessel{suffix}"
    if args.outdir.exists() and any(args.outdir.iterdir()) and not args.force:
        raise SystemExit(f"{args.outdir} is not empty: choose another -o or use --force")
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"results -> {args.outdir}")

    cohorts = tree_denominators(args.input)
    crossings = load_crossings(args.input)

    for vessel_type in VESSEL_TYPES:
        df = clean_for_tree(crossings, vessel_type, cohorts[vessel_type])
        reach = patient_reach(df)
        curve_df = curves(reach, len(cohorts[vessel_type]))

        print(f"{vessel_type}: prevalence at 0 / 5 / 10 / 20 mm (% of tree cohort):")
        print(curve_df.loc[[0.0, 5.0, 10.0, 20.0]].round(1).to_string())

        out_csv = args.outdir / f"fissure_length_curves_{vessel_type}.csv"
        curve_df.to_csv(out_csv, float_format="%.2f")
        print(f"wrote {out_csv}")

        plot(
            curve_df,
            vessel_type,
            len(cohorts[vessel_type]),
            args.outdir / f"fissure_length_curves_{vessel_type}.png",
        )


if __name__ == "__main__":
    main()
