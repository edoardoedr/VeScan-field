#!/usr/bin/env python3
"""Turns the raw files of a VMTK + field extraction into the curated ones.

    python statistics_scripts/curate_extraction.py RAW_DIR --date 2026-11-03 \
        [--rename SHORT=NAME ...] [--exclude DATASET:PATIENT ...] [--force]

RAW_DIR holds what the server wrote, untouched: crossing_statistics.xlsx (both pipelines, datasets
named <dataset>_vmtk / <dataset>_field) and centerline_comparison.xlsx/.csv. Writes, under
analisys/extractions/:

    <date>_vmtk/crossing_statistics.xlsx         the VMTK datasets, renamed to <dataset>
    <date>_field/crossing_statistics.xlsx        the field datasets, renamed to <dataset>
    <date>_vmtk-vs-field/centerline_comparison.xlsx/.csv   both pipelines, as before

--rename gives the crossing workbooks the dataset names of the comparison (e.g. spie=
spie_aapm_lung_ct_challenge), so that every file of the extraction names a dataset the same way.
--exclude (with those names) removes a patient from every file (both pipelines, every structure), so that the two
pipelines and the comparison keep the same cohort. Each output gets a .curation.json with the
raw source and what was removed. An existing output is never overwritten (--force).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTRACTIONS = ROOT / "analisys" / "extractions"
PIPELINES = ("vmtk", "field")
sys.path.insert(0, str(ROOT / "statistics_scripts"))

from aggregate_statistics import write_xlsx          # noqa: E402  same formatting as the extraction


def split_dataset(name):
    """'luna16_vmtk' -> ('luna16', 'vmtk')."""
    base, _, pipeline = str(name).rpartition("_")
    if pipeline not in PIPELINES or not base:
        raise ValueError(f"dataset {name!r} does not end in _vmtk or _field")
    return base, pipeline


def drop_excluded(df, excluded, dataset_col="dataset"):
    keep = ~pd.Series([(d, str(p)) in excluded for d, p in zip(df[dataset_col], df["patient"])], index=df.index)
    return df[keep], int((~keep).sum())


def check_free(paths, force):
    taken = [p for p in paths if p.exists()]
    if taken and not force:
        sys.exit("already there (use --force): " + ", ".join(str(p) for p in taken))


def record(path, raw, excluded, removed):
    info = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), command=sys.argv[1:],
                source=str(raw), excluded=[f"{d}:{p}" for d, p in sorted(excluded)], removed_rows=removed)
    path.with_name(path.stem + ".curation.json").write_text(json.dumps(info, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path)
    ap.add_argument("--date", required=True, help="the date of the extraction (folder names)")
    ap.add_argument("--rename", nargs="*", default=[], metavar="SHORT=NAME")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="DATASET:PATIENT")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    excluded = {tuple(e.split(":", 1)) for e in args.exclude}
    rename = dict(r.split("=", 1) for r in args.rename)

    crossing_out = {pl: EXTRACTIONS / f"{args.date}_{pl}" / "crossing_statistics.xlsx" for pl in PIPELINES}
    comparison_out = EXTRACTIONS / f"{args.date}_vmtk-vs-field" / "centerline_comparison.xlsx"
    check_free([*crossing_out.values(), comparison_out, comparison_out.with_suffix(".csv")], args.force)

    # crossing statistics: one workbook per pipeline
    sheets = pd.read_excel(args.raw / "crossing_statistics.xlsx", sheet_name=None)
    for pl in PIPELINES:
        out, removed = [], {}
        for name, df in sheets.items():
            base_pl = df["dataset"].map(split_dataset)
            df = df[base_pl.map(lambda x: x[1]) == pl].copy()
            df["dataset"] = base_pl[df.index].map(lambda x: rename.get(x[0], x[0]))
            df, removed[name] = drop_excluded(df, excluded)
            out.append((name, df))
        crossing_out[pl].parent.mkdir(parents=True, exist_ok=True)
        write_xlsx(crossing_out[pl], out)
        record(crossing_out[pl], args.raw / "crossing_statistics.xlsx", excluded, removed)
        n = out[1][1].groupby("dataset")["patient"].nunique().to_dict()
        print(f"{pl}: {n} patients, removed rows {removed} -> {crossing_out[pl]}")

    # centerline comparison: both pipelines together, the summary sheets recomputed
    df = pd.read_csv(args.raw / "centerline_comparison.csv")
    df, n_removed = drop_excluded(df, excluded)
    comparison_out.parent.mkdir(parents=True, exist_ok=True)
    from compare_centerlines import write_report        # imports vtk: only needed here
    write_report(df.to_dict("records"), comparison_out)
    record(comparison_out, args.raw / "centerline_comparison.csv", excluded, {"per_structure": n_removed})


if __name__ == "__main__":
    main()
