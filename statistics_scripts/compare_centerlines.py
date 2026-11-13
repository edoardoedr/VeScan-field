#!/usr/bin/env python3
"""Compares the centerlines of two or more pipelines (vmtk, field, thalweg, ...), run on the same patients.

    python statistics_scripts/compare_centerlines.py --datasets configs/analysis/server.json -o out.xlsx

--datasets reads the roots of every dataset and pipeline, and the excluded patients, from one
file (statistics_scripts/datasets_file.py). The run labels are the column suffixes (vmtk,
thalweg, field, ...); the first is the baseline of the wins sheet and the default --reference.

Each root holds one folder per patient, with final_export_<structure>/ inside. Only the patients and
structures present in every root are compared; --exclude drops patients (e.g. a fragmented
segmentation).

There is no ground-truth centerline. All are measured against the same vessel surface,
supporting_files/01_preprocessed.vtk of the --reference run, which is the segmentation that
every pipeline starts from. --reference-surfaces DIR takes it from DIR/<dataset>/ instead, e.g.
the field's own boundary written by boundary_vs_reference.py --write-surfaces DIR. Per patient
and structure:

  Geometry (each pipeline), on the centerline resampled every 0.5 mm, one cell per group:
    length_mm, groups, max_generation, leaves, outside_frac (length outside the surface);
    short_leaf_frac: leaves shorter than twice their radius (the 2026-11-03 spur definition;
    the field prunes most of these by construction);
    spur_frac / spurs: leaves whose tip protrudes less than their own radius beyond the wall of
    the vessel they leave (distance from the tip to the parent's and siblings' centerlines minus
    the parent radius at the junction), a definition that depends on neither pipeline's pruning;
    path_max_mm / path_p50_mm / path_p90_mm: path length from the root to each tip, along the
    tree (a depth that does not depend on how bifurcations are counted);
    radius_min_mm / radius_p05_mm: the smallest branch radius reached (minimum, 5th percentile).
  Centering (each pipeline), on points at least 2 radii from a group's ends (no bifurcations):
    the cross-section of the surface in the plane normal to the centerline, from 24 rays.
    offset_rel = distance from the point to the section's centroid / section radius (0 = centred);
    radius_err = (Radius array - section radius) / section radius (signed);
    curvature: median turning angle per mm (staircase and jitter raise it).
  Agreement (every pair of pipelines a, b):
    dist_<stat>_<a>_to_<b> = distance from a's centerline to b's (mean, median, p95, max);
    matched_<a>_in_<b> = fraction of a's length within max(1 mm, local lumen radius) of b, and
    matched_f1_<a>_<b>;
    missing_area_pct_<a>_vs_<b> = surface area (% of the total) that a does not reach but b does:
    distance > 2 x b's + 1 mm. These are branches only b has.
    missing_area_pct_<a> = the same against the best of all the others (with two pipelines,
    the same as the pairwise one).
  JSON (each pipeline): crossing counts and length (statistics.json), run time and, for the
    field pipeline, the Dice of the margin's argmax against the masks (run_summary.json).
  Case: slice_mm, the slice spacing of the patient's masks (the CT's), from the patient_dir of
    patient_summary.json; NaN if that folder is not readable from here.

An existing output is never overwritten (--force to do it); <out>.run.json records the command,
the date, the git commit and the roots. Writes an .xlsx: one row per patient and structure
(per_structure), the medians per dataset and structure (summary), the same per slice-spacing
bin (by_slice: <=1, 1-2, 2-3, >3 mm), and, for each precision metric, the share of cases where
each pipeline is better than the baseline (wins).
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import vtk
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # the repo root: vescan_field

import datasets_file                                                   # noqa: E402  (this folder)
from line_distance import Lines, distance                              # noqa: E402  (this folder)
from vescan_field.surface import (cell_areas, cell_array, cell_point_ids, close_if_open, load_surface,
                                  point_array, points, points_inside, representative_cells, triangles)

STEP_MM = 0.5           # resampling of the centerlines
CENTER_EVERY = 2         # centering on one sample in CENTER_EVERY (1 mm)
MAX_CENTER_SAMPLES = 20000
N_RAYS = 24
RADIUS_ARRAYS = ("Radius", "MaximumInscribedSphereRadius")
# metric -> True if larger is better (for the wins sheet)
PRECISION = {"offset_rel_median": False, "abs_radius_err_median": False, "outside_frac": False,
             "short_leaf_frac": False, "spur_frac": False, "curvature_median": False, "missing_area_pct": False}
SLICE_BINS = [(0.0, 1.0, "<=1"), (1.0, 2.0, "1-2"), (2.0, 3.0, "2-3"), (3.0, np.inf, ">3")]

logger = logging.getLogger("compare_centerlines")


# --------------------------------------------------------------------------- loading
def structure_dirs(root):
    """{(patient, structure): final_export folder} under one results root."""
    found = {}
    for patient in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for d in sorted(patient.glob("final_export_*")):
            found[patient.name, d.name[len("final_export_"):]] = d
    return found


def load_tree(folder):
    """The branch tree of one run: one cell per group (the one with most points: VMTK traces
    a branch once per source centerline), with its radius."""
    poly = load_surface(folder / "supporting_files" / "05_branch_tree.vtk")
    xyz, cells = points(poly), cell_point_ids(poly)
    group = cell_array(poly, "GroupIds")
    group = cell_array(poly, "GroupId") if group is None else group
    radius = next((r for r in (point_array(poly, n) for n in RADIUS_ARRAYS) if r is not None), None)
    if radius is None:
        radius = np.zeros(len(xyz))
    bif = cell_array(poly, "IsBifurcation")
    rep = representative_cells(group, cells)
    cell_ids = list(rep.values())
    return dict(xyz=xyz, cells=[cells[c] for c in cell_ids], radius=np.asarray(radius, float),
                groups=[int(g) for g in rep], bifurcation=[bool(bif[c]) if bif is not None else False for c in cell_ids])


def resample(tree):
    """Samples every STEP_MM along each cell: position, tangent, radius, distance to the nearer
    end of the cell, curvature (turning angle per mm over +-1 mm), and whether the cell is a
    bifurcation group."""
    out = {k: [] for k in ("x", "t", "r", "end", "kappa", "bif")}
    for ids, bif in zip(tree["cells"], tree["bifurcation"]):
        p, r = tree["xyz"][ids], tree["radius"][ids]
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        length = s[-1]
        if length < STEP_MM:
            continue
        q = np.arange(STEP_MM / 2, length, STEP_MM)
        at = lambda v: np.stack([np.interp(np.clip(v, 0, length), s, p[:, k]) for k in range(3)], axis=1)
        x, before, after = at(q), at(q - 1.0), at(q + 1.0)
        a, b = x - before, after - x
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        tangent = after - before
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
        cos = np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-12)
        kappa = np.arccos(np.clip(cos, -1, 1)) / np.maximum((na + nb) / 2, 1e-6)
        kappa[(q < 1.0) | (q > length - 1.0)] = np.nan
        out["x"].append(x)
        out["t"].append(tangent)
        out["r"].append(np.interp(q, s, r))
        out["end"].append(np.minimum(q, length - q))
        out["kappa"].append(kappa)
        out["bif"].append(np.full(len(q), bif))
    return {k: np.concatenate(v) if v else np.zeros((0, 3) if k in ("x", "t") else 0) for k, v in out.items()}


def tree_metrics(folder, tree):
    """Topology, spurs, root-to-tip path lengths and smallest radius of one tree."""
    topo = json.loads((folder / "supporting_files" / "05_branch_tree_topology.json").read_text())
    nodes = {int(k): v for k, v in topo["nodes"].items()}
    children, parent = {}, {}
    for e in topo["edges"]:
        children.setdefault(e["parent"], []).append(e["child"])
        parent[e["child"]] = e["parent"]
    branches = {g: n for g, n in nodes.items() if not n.get("blanked") and not n.get("virtual")}
    leaves = [g for g in branches if not children.get(g)]
    radius = lambda g: branches[g].get("avg_radius") or 0.0
    short = sum(1 for g in leaves if branches[g]["length"] < 2 * radius(g))

    # path length from the root to every tip, along the tree (bifurcation groups included)
    depth, stack = {}, [(r, 0.0) for r in topo["roots"]]
    while stack:
        g, d = stack.pop()
        if g in depth:
            continue
        depth[g] = d + (nodes.get(g, {}).get("length") or 0.0)
        stack.extend((c, depth[g]) for c in children.get(g, []))
    paths = np.array([depth[g] for g in leaves if g in depth])
    radii = np.array([radius(g) for g in branches if radius(g) > 0])

    # spurs: the tip protrudes less than the leaf's own radius beyond the wall of the vessel it
    # leaves (the parent branch and the siblings at the same bifurcation)
    cell_of = dict(zip(tree["groups"], tree["cells"]))
    xyz, rad = tree["xyz"], tree["radius"]
    tested = spurs = 0
    for g in leaves:
        b = parent.get(g)
        p = parent.get(b) if b is not None and nodes.get(b, {}).get("blanked") else None
        if p is None or p not in cell_of or g not in cell_of:
            continue                                  # a leaf hanging from the root: no parent wall
        vessel = [cell_of[p]] + [cell_of[s] for s in children.get(b, []) if s != g and s in cell_of]
        leaf = cell_of[g]
        junction = xyz[cell_of[p][-1]]
        ends = xyz[[leaf[0], leaf[-1]]]
        tip = ends[np.argmax(np.linalg.norm(ends - junction, axis=1))]      # the end away from the junction
        d = float(distance(tip[None], xyz, vessel)[0])
        tested += 1
        spurs += (d - rad[cell_of[p][-1]]) < radius(g)
    return dict(groups=len(branches), max_generation=max((n["generation"] for n in branches.values()), default=0),
                leaves=len(leaves), short_leaf_frac=short / len(leaves) if leaves else np.nan,
                spurs=spurs, spur_frac=spurs / tested if tested else np.nan,
                path_max_mm=float(paths.max()) if len(paths) else np.nan,
                path_p50_mm=float(np.median(paths)) if len(paths) else np.nan,
                path_p90_mm=float(np.percentile(paths, 90)) if len(paths) else np.nan,
                radius_min_mm=float(radii.min()) if len(radii) else np.nan,
                radius_p05_mm=float(np.percentile(radii, 5)) if len(radii) else np.nan)


def slice_spacing(folders, patient_dir=None):
    """The slice spacing (mm) of the patient's masks: in `patient_dir` if given (a local copy),
    otherwise in the first readable patient_dir of the runs' patient_summary.json."""
    import SimpleITK as sitk
    candidates = [patient_dir] if patient_dir is not None else []
    for folder in folders:
        summary = folder.parent / "patient_summary.json"
        if summary.exists():
            candidates.append(Path(json.loads(summary.read_text()).get("patient_dir", "")))
    for patient_dir in candidates:
        masks = sorted(patient_dir.glob("lung_*.nii.gz")) if patient_dir.is_dir() else []
        if masks:
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(masks[0]))
            reader.ReadImageInformation()
            return float(reader.GetSpacing()[2])
    return np.nan


def json_metrics(folder):
    out = {}
    stats = folder / "statistics.json"
    if stats.exists():
        s = json.loads(stats.read_text())
        for k in ("n_crossings", "n_interlobar", "n_translobar", "n_unclassified"):
            out[k] = s.get(k)
        out["crossing_length_mm"] = s.get("total_length_mm")
    summary = folder / "run_summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        timings = s.get("timings_s") or {}
        # the whole structure: "total" when there is one (before 2026-11-12 it was summed with the
        # steps it already contains, which doubled every time)
        total = timings.get("total")
        if not isinstance(total, (int, float)):
            total = sum(v for v in timings.values() if isinstance(v, (int, float)))
        out["time_s"] = round(total, 1) or None
        out["dice"] = (s.get("qa_mask") or {}).get("dice")
    return out


# --------------------------------------------------------------------------- centering
def cross_sections(locator, x, t, r):
    """For each point (x, tangent t, radius r): (centroid offset, section radius) of the
    surface's cross-section in the normal plane, from N_RAYS rays (NaN if a ray finds no wall)."""
    angles = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    hit, pcoords, sub = [0.0] * 3, [0.0] * 3, vtk.reference(0)
    tt = vtk.reference(0.0)
    offset, radius = np.full(len(x), np.nan), np.full(len(x), np.nan)
    for i in range(len(x)):
        u = np.cross(t[i], [1.0, 0, 0] if abs(t[i][0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(t[i], u)
        reach = 4 * max(r[i], 1.0) + 5.0
        d = np.empty(N_RAYS)
        for k in range(N_RAYS):
            end = x[i] + reach * (cos[k] * u + sin[k] * v)
            if not locator.IntersectWithLine(x[i], end, 1e-6, tt, hit, pcoords, sub):
                break
            d[k] = tt.get() * reach
        else:
            a, b = d * cos, d * sin                      # the section polygon, in plane coordinates
            a2, b2 = np.roll(a, -1), np.roll(b, -1)
            cross = a * b2 - a2 * b
            area = cross.sum() / 2
            if area <= 0:
                continue
            ca, cb = ((a + a2) * cross).sum() / (6 * area), ((b + b2) * cross).sum() / (6 * area)
            radius[i] = np.sqrt(area / np.pi)
            offset[i] = np.hypot(ca, cb)
    return offset, radius


# --------------------------------------------------------------------------- one structure
def compare(job):
    """All the metrics of one (dataset, patient, structure). `folders`: {label: folder}, in order."""
    dataset, patient, structure, folders, reference, patients_root = job
    pipelines = tuple(folders)
    row = dict(dataset=dataset, patient=patient, structure=structure)
    ref_dir = folders[reference] if isinstance(reference, str) else reference     # a label, or a folder
    surface = close_if_open(load_surface(ref_dir / "supporting_files" / "01_preprocessed.vtk"),
                            f"{patient}/{structure} {ref_dir} 01_preprocessed.vtk")
    row["reference"] = reference if isinstance(reference, str) else str(reference)
    row["slice_mm"] = slice_spacing(folders.values(), patients_root / patient if patients_root else None)
    row["surface_points"] = surface.GetNumberOfPoints()
    spts = points(surface)
    tri = triangles(surface)
    area_per_point = np.bincount(tri.ravel(), np.repeat(cell_areas(surface) / 3, 3), minlength=len(spts))
    total_area = area_per_point.sum()
    surface_tree = cKDTree(spts)
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    rng = np.random.default_rng(0)

    trees, samples = {}, {}
    for pl in pipelines:
        trees[pl] = load_tree(folders[pl])
        s = samples[pl] = resample(trees[pl])
        inside = points_inside(surface, s["x"], 1e-3)
        s["inside"] = inside
        row[f"length_mm_{pl}"] = round(len(s["x"]) * STEP_MM, 1)
        row[f"outside_frac_{pl}"] = float(1 - inside.mean()) if len(inside) else np.nan
        for k, v in tree_metrics(folders[pl], trees[pl]).items():
            row[f"{k}_{pl}"] = v
        row[f"curvature_median_{pl}"] = float(np.nanmedian(s["kappa"])) if np.isfinite(s["kappa"]).any() else np.nan

        # centering, away from bifurcations and cell ends, inside the surface
        ok = inside & ~s["bif"] & (s["end"] >= np.maximum(2 * s["r"], 2.0))
        idx = np.flatnonzero(ok)[::CENTER_EVERY]
        if len(idx) > MAX_CENTER_SAMPLES:
            idx = np.sort(rng.choice(idx, MAX_CENTER_SAMPLES, replace=False))
        offset, radius = cross_sections(locator, s["x"][idx], s["t"][idx], s["r"][idx])
        good = np.isfinite(offset)
        rel = offset[good] / radius[good]
        err = (s["r"][idx][good] - radius[good]) / radius[good]
        row[f"center_samples_{pl}"] = int(good.sum())
        row[f"offset_rel_median_{pl}"] = float(np.median(rel)) if len(rel) else np.nan
        row[f"offset_rel_p90_{pl}"] = float(np.percentile(rel, 90)) if len(rel) else np.nan
        row[f"offset_mm_median_{pl}"] = float(np.median(offset[good])) if good.any() else np.nan
        row[f"radius_err_median_{pl}"] = float(np.median(err)) if len(err) else np.nan
        row[f"abs_radius_err_median_{pl}"] = float(np.median(np.abs(err))) if len(err) else np.nan
        for k, v in json_metrics(folders[pl]).items():
            row[f"{k}_{pl}"] = v

    # agreement, every ordered pair (plain distance to the polylines: line_distance.py)
    lines = {pl: Lines(trees[pl]["xyz"], trees[pl]["cells"]) for pl in pipelines}
    for a, b in itertools.permutations(pipelines, 2):
        xa = samples[a]["x"]
        d = lines[b].distance(xa)
        lumen = surface_tree.query(xa)[0]
        row[f"dist_mean_{a}_to_{b}"] = float(d.mean())
        row[f"dist_median_{a}_to_{b}"] = float(np.median(d))
        row[f"dist_p95_{a}_to_{b}"] = float(np.percentile(d, 95))
        row[f"dist_max_{a}_to_{b}"] = float(d.max())
        row[f"matched_{a}_in_{b}"] = float((d <= np.maximum(lumen, 1.0)).mean())
    for a, b in itertools.combinations(pipelines, 2):
        m1, m2 = row[f"matched_{a}_in_{b}"], row[f"matched_{b}_in_{a}"]
        row[f"matched_f1_{a}_{b}"] = 2 * m1 * m2 / (m1 + m2) if m1 + m2 else 0.0

    d_surf = {pl: lines[pl].distance(spts) for pl in pipelines}
    for a, b in itertools.permutations(pipelines, 2):
        missing = d_surf[a] > 2 * d_surf[b] + 1.0
        row[f"missing_area_pct_{a}_vs_{b}"] = float(100 * area_per_point[missing].sum() / total_area)
    for a in pipelines:                          # against the best of the others
        best_other = np.min([d_surf[b] for b in pipelines if b != a], axis=0)
        missing = d_surf[a] > 2 * best_other + 1.0
        row[f"missing_area_pct_{a}"] = float(100 * area_per_point[missing].sum() / total_area)
        row[f"missing_area_mm2_{a}"] = float(area_per_point[missing].sum())

    return row


# --------------------------------------------------------------------------- report
def write_report(rows, out, pipelines):
    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["dataset", "patient", "structure"])
    numeric = df.select_dtypes("number").columns
    summary = df.groupby(["dataset", "structure"])[list(numeric)].median()
    summary.insert(0, "cases", df.groupby(["dataset", "structure"]).size())
    base = pipelines[0]
    wins = []
    for (dataset, structure), g in df.groupby(["dataset", "structure"]):
        row = dict(dataset=dataset, structure=structure, cases=len(g))
        for pl in pipelines[1:]:
            for metric, larger_better in PRECISION.items():
                v, f = g[f"{metric}_{base}"], g[f"{metric}_{pl}"]
                valid = v.notna() & f.notna()
                better = (f > v) if larger_better else (f < v)
                row[f"{metric}: {pl} better than {base}"] = float(better[valid].mean()) if valid.any() else np.nan
        wins.append(row)
    sliced = df.assign(slice_bin=pd.cut(df["slice_mm"], [lo for lo, _, _ in SLICE_BINS] + [np.inf],
                                        labels=[name for _, _, name in SLICE_BINS], right=True))
    by_slice = sliced.groupby(["dataset", "structure", "slice_bin"], observed=True)[list(numeric)].median()
    by_slice.insert(0, "cases", sliced.groupby(["dataset", "structure", "slice_bin"], observed=True).size())
    with pd.ExcelWriter(out) as xl:
        df.to_excel(xl, sheet_name="per_structure", index=False)
        summary.reset_index().to_excel(xl, sheet_name="summary", index=False)
        by_slice.reset_index().to_excel(xl, sheet_name="by_slice", index=False)
        pd.DataFrame(wins).to_excel(xl, sheet_name="wins", index=False)
    df.to_csv(Path(out).with_suffix(".csv"), index=False)

    cols = ["length_mm", "groups", "max_generation", "path_max_mm", "path_p90_mm", "radius_p05_mm",
            "outside_frac", "offset_rel_median", "abs_radius_err_median", "radius_err_median",
            "curvature_median", "short_leaf_frac", "spur_frac", "missing_area_pct", "n_crossings",
            "time_s", "dice"]
    for (dataset, structure), g in df.groupby(["dataset", "structure"]):
        print(f"\n{dataset} / {structure}: {len(g)} patients (medians; slice {g['slice_mm'].median():.2f} mm)")
        print(f"  {'metric':24s}" + "".join(f" {pl:>10s}" for pl in pipelines))
        for c in cols:
            if f"{c}_{base}" in g:
                print(f"  {c:24s}" + "".join(f" {pd.to_numeric(g[f'{c}_{pl}'], errors='coerce').median():10.3f}"
                                             for pl in pipelines))
        for a, b in itertools.combinations(pipelines, 2):
            print(f"  {a} vs {b}: matched {a} in {b} {g[f'matched_{a}_in_{b}'].median():.3f}, "
                  f"{b} in {a} {g[f'matched_{b}_in_{a}'].median():.3f}, F1 {g[f'matched_f1_{a}_{b}'].median():.3f}; "
                  f"distance {a}->{b} median {g[f'dist_median_{a}_to_{b}'].median():.2f} mm, "
                  f"p95 {g[f'dist_p95_{a}_to_{b}'].median():.2f} mm")
    print(f"\n-> {out}")


def write_run_info(output, inputs, failed):
    """<output>.run.json: when, with which command, commit and inputs the comparison was made."""
    import datetime
    import subprocess
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    info = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), command=sys.argv,
                git_commit=commit, inputs=inputs, failed=failed)
    Path(output).with_suffix(".run.json").write_text(json.dumps(info, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", required=True, help="datasets file (configs/analysis/*.json): roots and excluded patients")
    ap.add_argument("-o", "--output", default="centerline_comparison.xlsx")
    ap.add_argument("--reference", help="label whose 01_preprocessed.vtk is the reference surface (default: the first)")
    ap.add_argument("--reference-surfaces", help="take 01_preprocessed.vtk from DIR/<dataset>/<patient>/... instead, "
                                                 "e.g. thalweg's boundary (boundary_vs_reference.py --write-surfaces DIR)")
    ap.add_argument("--structures", nargs="*", help="only these structures (e.g. artery veins)")
    ap.add_argument("--patients", nargs="*", help="only these patients")
    ap.add_argument("--exclude", nargs="*", default=[], help="drop these patients (in every dataset)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    ap.add_argument("--workers", type=int, default=0, help="processes (default: the usable cores)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    from_file, pipelines, excluded = datasets_file.load(args.datasets)
    datasets = {name: d["runs"] for name, d in from_file.items()}          # name -> {label: root}
    patients_roots = {name: d["patients_root"] for name, d in from_file.items()}
    excluded |= set(args.exclude)
    reference = args.reference or pipelines[0]
    if reference not in pipelines:
        ap.error(f"--reference {reference} is not one of {pipelines}")
    if Path(args.output).exists() and not args.force:
        sys.exit(f"{args.output} exists: choose another -o (one folder per extraction) or use --force")

    ref_roots = {name: Path(args.reference_surfaces) / name for name in datasets} if args.reference_surfaces else {}
    jobs = []
    for name, roots in datasets.items():
        found = {pl: structure_dirs(root) for pl, root in roots.items()}
        ref_found = None
        if name in ref_roots:
            ref_found = structure_dirs(ref_roots[name]) if ref_roots[name].is_dir() else {}
        common = sorted(set.intersection(*(set(f) for f in found.values())))
        only = ", ".join(f"only {pl} {len(set(found[pl]) - set(common))}" for pl in pipelines)
        print(f"[{name}] {len(common)} patient x structure in all ({only})")
        for patient, structure in common:
            if args.structures and structure not in args.structures or args.patients and patient not in args.patients:
                continue
            if patient in excluded:
                print(f"  excluded {patient}/{structure}")
                continue
            folders = {pl: found[pl][patient, structure] for pl in pipelines}
            needed = ["supporting_files/05_branch_tree.vtk", "supporting_files/05_branch_tree_topology.json"]
            missing = [f"{pl}:{n}" for pl in pipelines for n in needed if not (folders[pl] / n).exists()]
            ref_dir = folders[reference] if ref_found is None else ref_found.get((patient, structure))
            if ref_dir is None or not (ref_dir / "supporting_files" / "01_preprocessed.vtk").exists():
                missing.append(f"{'reference root' if ref_found is not None else reference}:"
                               "supporting_files/01_preprocessed.vtk")
            if missing:
                print(f"  skip {patient}/{structure}: missing {missing}")
                continue
            jobs.append((name, patient, structure, folders, reference if ref_found is None else ref_dir,
                         patients_roots.get(name)))
    if not jobs:
        sys.exit("nothing to compare")

    from vescan_field.main import usable_cpus
    from vescan_field import threads
    workers = args.workers or usable_cpus()
    threads.KDTREE_WORKERS = 1 if workers > 1 else -1
    rows, failed = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(workers,)) as pool:
        futures = {pool.submit(compare, job): job for job in jobs}
        for n, fut in enumerate(as_completed(futures), start=1):
            job = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:                              # one bad case must not stop the rest
                failed.append(f"{job[0]}/{job[1]}/{job[2]}: {exc!r}")
            print(f"\r{n}/{len(jobs)} done, {len(failed)} failed", end="", flush=True)
    print()
    for f in failed:
        print("  failed", f)
    if rows:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_report(rows, args.output, pipelines)
        inputs = {name: {pl: str(Path(r).resolve()) for pl, r in roots.items()} for name, roots in datasets.items()}
        for name, root in ref_roots.items():
            inputs[name]["reference_root"] = str(Path(root).resolve())
        write_run_info(args.output, inputs, failed)


def _init_worker(workers):
    from vescan_field import threads
    threads.KDTREE_WORKERS = 1 if workers > 1 else -1
    os.environ.setdefault("OMP_NUM_THREADS", "1")


if __name__ == "__main__":
    main()
