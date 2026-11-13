#!/usr/bin/env python3
"""Crossing vessels by vessel radius and depth past the fissure, per pipeline, with the agreement
between pipelines (NEXT.md 7.5).

    python statistics_scripts/crossings_by_radius.py --datasets configs/analysis/server.json -o out.xlsx

Reads statistics.json of every pipeline in the datasets file (patients present in all of them,
with all three structures, minus the excluded ones). The question: which crossings reflect the
anatomy, and which the segmentation model (unreliable below ~1 mm radius, thin vessels lost on
thick slices) or the lobe mask (its fissure is placed with an error of a few mm). A crossing that
is thick and runs far past the fissure is anatomy; a thin one that barely crosses may be neither.

Each crossing vessel gets, as in statistics_scripts/plots/fissure_length_curves.py:
  contact   from the unordered lobe pair {from_lobe, to_lobe}: R-HIF {RUL,RML}, R-UOIF {RUL,RLL},
            R-LOIF {RML,RLL}, L-major {LUL,LLL}; ambiguous (RLL+RUL), same-lobe or cross-lung
            pairs are dropped;
  depth     length_after_piercing_mm, the longest path past the fissure (must be > 0; a crossing
            that never pierces a sheet has none and is dropped);
  radius    radius_mm, the median vessel radius at its piercings.

Groups: radius <1, 1-2, >2 mm (and any); minimum depth 0, 1, 2, 5, 9, 20 mm (the lobe-mask
boundary-shift sensitivity of the 2026-11-03 analysis, plus 20 mm); structure (and any: the
three summed per patient); contact (and any).

Writes one .xlsx (never overwritten without --force) and a .png next to it:
  crossings   one row per crossing kept, with its pipeline, contact, radius and depth;
  summary     per dataset (and all), structure, contact, radius bin, minimum depth and pipeline:
              patients, crossings per patient, prevalence (% of patients with at least one);
  agreement   the same groups, per pair of pipelines: exact count match (% of patients with the
              same count) and Cohen's kappa on presence/absence per patient;
  dropped     how many crossings each rule dropped, per pipeline.
  .png        prevalence vs minimum depth, one panel per radius bin, one line per pipeline
              (all datasets, any structure, any contact).
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets_file                                                   # noqa: E402

STRUCTURES = ("artery", "veins", "airways")
CONTACTS = {frozenset({"RUL", "RML"}): "R-HIF", frozenset({"RUL", "RLL"}): "R-UOIF",
            frozenset({"RML", "RLL"}): "R-LOIF", frozenset({"LUL", "LLL"}): "L-major"}
CONTACT_ORDER = ["R-HIF", "R-UOIF", "R-LOIF", "L-major"]
RADIUS_BINS = [(0.0, 1.0, "<1"), (1.0, 2.0, "1-2"), (2.0, np.inf, ">2"), (0.0, np.inf, "any")]
MIN_DEPTHS = [0.0, 1.0, 2.0, 5.0, 9.0, 20.0]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#c4398b"]


# --------------------------------------------------------------------------- reading
def read_crossings(datasets, labels, exclude):
    """(crossings DataFrame, cohort {dataset: [patients]}, dropped counts)."""
    rows, cohort, dropped = [], {}, []
    for name, d in datasets.items():
        found = {}                                 # (patient, structure) -> {label: statistics.json}
        for label, root in d["runs"].items():
            for path in Path(root).glob("*/final_export_*/statistics.json"):
                structure = path.parent.name[len("final_export_"):]
                found.setdefault((path.parent.parent.name, structure), {})[label] = path
        complete = {}
        for (patient, structure), files in found.items():
            if len(files) == len(labels) and patient not in exclude:
                complete.setdefault(patient, {})[structure] = files
        patients = sorted(p for p, s in complete.items() if set(STRUCTURES) <= set(s))
        cohort[name] = patients
        print(f"[{name}] {len(patients)} patients with the three structures in every pipeline "
              f"({len(complete) - len(patients)} incomplete)")
        for patient in patients:
            for structure in STRUCTURES:
                for label in labels:
                    stats = json.loads(complete[patient][structure][label].read_text())
                    for c in stats.get("crossings", []):
                        reason = None
                        ends = [str(c.get("from_lobe") or ""), str(c.get("to_lobe") or "")]
                        pair = frozenset(ends)
                        depth = c.get("length_after_piercing_mm")
                        if not all(ends):
                            reason = "no lobe pair"
                        elif any("+" in e for e in ends):
                            reason = "ambiguous lobe pair"
                        elif pair not in CONTACTS:
                            reason = "same-lobe or cross-lung pair"
                        elif depth is None or depth <= 0:
                            reason = "no piercing / no depth"
                        elif c.get("radius_mm") is None:
                            reason = "no radius"
                        if reason:
                            dropped.append(dict(dataset=name, pipeline=label, reason=reason))
                            continue
                        rows.append(dict(dataset=name, patient=patient, structure=structure, pipeline=label,
                                         contact=CONTACTS[pair], radius_mm=float(c["radius_mm"]),
                                         depth_mm=float(depth), crossing_type=c.get("crossing_type")))
    df = pd.DataFrame(rows, columns=["dataset", "patient", "structure", "pipeline", "contact",
                                     "radius_mm", "depth_mm", "crossing_type"])
    drop = (pd.DataFrame(dropped).groupby(["dataset", "pipeline", "reason"]).size().rename("crossings")
            .reset_index()) if dropped else pd.DataFrame(columns=["dataset", "pipeline", "reason", "crossings"])
    return df, cohort, drop


# --------------------------------------------------------------------------- statistics
def kappa(a, b):
    """Cohen's kappa of two boolean arrays (NaN when undefined: both constant and equal)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def tables(df, cohort, labels):
    """The summary and agreement tables."""
    names = list(cohort) + (["all"] if len(cohort) > 1 else [])
    summary, agreement = [], []
    for dataset in names:
        keys = [(d, p) for d in cohort for p in cohort[d]] if dataset == "all" else [(dataset, p) for p in cohort[dataset]]
        index = pd.MultiIndex.from_tuples(keys, names=["dataset", "patient"])
        part = df if dataset == "all" else df[df.dataset == dataset]
        for structure in STRUCTURES + ("any",):
            s = part if structure == "any" else part[part.structure == structure]
            for contact in CONTACT_ORDER + ["any"]:
                c = s if contact == "any" else s[s.contact == contact]
                for lo, hi, rbin in RADIUS_BINS:
                    r = c[(c.radius_mm >= lo) & (c.radius_mm < hi)]
                    for depth in MIN_DEPTHS:
                        x = r[r.depth_mm >= depth]
                        counts = {label: x[x.pipeline == label].groupby(["dataset", "patient"]).size()
                                  .reindex(index, fill_value=0).to_numpy() for label in labels}
                        key = dict(dataset=dataset, structure=structure, contact=contact,
                                   radius=rbin, min_depth_mm=depth)
                        for label in labels:
                            n = counts[label]
                            summary.append(dict(key, pipeline=label, patients=len(n),
                                                crossings_per_patient=float(n.mean()) if len(n) else np.nan,
                                                prevalence_pct=float(100 * (n > 0).mean()) if len(n) else np.nan))
                        for a, b in itertools.combinations(labels, 2):
                            agreement.append(dict(key, pair=f"{a}-{b}", patients=len(index),
                                                  exact_count_match_pct=float(100 * (counts[a] == counts[b]).mean()),
                                                  kappa_presence=kappa(counts[a] > 0, counts[b] > 0)))
    return pd.DataFrame(summary), pd.DataFrame(agreement)


def figure(summary, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dataset = "all" if "all" in set(summary.dataset) else summary.dataset.iloc[0]
    s = summary[(summary.dataset == dataset) & (summary.structure == "any") & (summary.contact == "any")]
    bins = [name for _, _, name in RADIUS_BINS]
    fig, axes = plt.subplots(1, len(bins), figsize=(3.8 * len(bins), 3.6), sharey=True)
    for ax, rbin in zip(axes, bins):
        for label, color in zip(labels, COLORS):
            g = s[(s.radius == rbin) & (s.pipeline == label)].sort_values("min_depth_mm")
            ax.plot(g.min_depth_mm, g.prevalence_pct, marker="o", color=color, lw=2, label=label)
        ax.set_title(f"radius {rbin} mm")
        ax.set_xlabel("minimum depth past the fissure (mm)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("patients with a crossing (%)")
    axes[0].legend(frameon=False)
    n = int(s.patients.iloc[0]) if len(s) else 0
    fig.suptitle(f"{dataset}: {n} patients, any structure, any contact", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def write_run_info(path, datasets_path, failed=()):
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    info = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), command=sys.argv,
                git_commit=commit, datasets=json.loads(Path(datasets_path).read_text()))
    path.write_text(json.dumps(info, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", required=True, help="datasets file (configs/analysis/*.json)")
    ap.add_argument("-o", "--output", required=True, help="output .xlsx")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists() and not args.force:
        sys.exit(f"{out} exists: choose another -o or use --force")

    datasets, labels, exclude = datasets_file.load(args.datasets)
    df, cohort, dropped = read_crossings(datasets, labels, exclude)
    if not any(cohort.values()):
        sys.exit("no patient with the three structures in every pipeline")
    summary, agreement = tables(df, cohort, labels)

    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as xl:
        df.to_excel(xl, sheet_name="crossings", index=False)
        summary.to_excel(xl, sheet_name="summary", index=False)
        agreement.to_excel(xl, sheet_name="agreement", index=False)
        dropped.to_excel(xl, sheet_name="dropped", index=False)
    figure(summary, labels, out.with_suffix(".png"))
    write_run_info(out.with_suffix(".run.json"), args.datasets)

    # on screen: all datasets, any structure, any contact
    top = "all" if len(cohort) > 1 else next(iter(cohort))
    s = summary[(summary.dataset == top) & (summary.structure == "any") & (summary.contact == "any")]
    print(f"\n{top}: patients with a crossing (%), any structure and contact")
    print(s.pivot_table(index=["radius", "min_depth_mm"], columns="pipeline", values="prevalence_pct")
          [list(labels)].round(1).to_string())
    a = agreement[(agreement.dataset == top) & (agreement.structure == "any") & (agreement.contact == "any")]
    print("\nagreement: exact count match (%) / kappa")
    print(a.pivot_table(index=["radius", "min_depth_mm"], columns="pair",
                        values=["exact_count_match_pct", "kappa_presence"]).round(2).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
