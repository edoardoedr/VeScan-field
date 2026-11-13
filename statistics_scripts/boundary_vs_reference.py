#!/usr/bin/env python3
"""Where the margin's boundary (thalweg's, or field's) lies against the reference surface
01_preprocessed.vtk (NEXT.md 7.2).

    python statistics_scripts/boundary_vs_reference.py --datasets configs/analysis/server.json \
        [--boundary thalweg|field] [--write-surfaces DIR] [--structures ...] [--patients ...] -o OUT_DIR

Every dataset of the datasets file with a config for the --boundary pipeline (configs.<pipeline>):
the reference is 01_preprocessed.vtk of the vmtk run (--reference), the local radius comes from
the --boundary run.

compare_centerlines.py measures centering, radius error and length outside against
01_preprocessed.vtk: flying edges on the mask, windowed-sinc smoothing (factor 0.5), decimation to
100 000 points. That is VMTK's own input; thalweg and field read the segmentation network's margin
instead, and their radii read larger than the reference's section (thalweg +9 %, field +3.5 % on
the PD-1-Lung-00043 arteries). A nearest-wall radius cannot be too large unless the wall it sees
is outside the reference: this script measures that directly.

Per patient and structure (store and classes from the config, as the pipeline does):
  1. the margin: --boundary thalweg (default) on the store's own grid
     (vescan_thalweg.margin.union_margin), --boundary field resampled onto the CT grid
     (vescan_field.field.margin_from_store); restricted to its largest 26-connected component, as
     the centerlines and 01_preprocessed.vtk are;
  2. its sub-voxel zero crossings (vescan_field Boundary);
  3. their signed distance to 01_preprocessed.vtk of the reference run (vtkImplicitPolyDataDistance):
     POSITIVE when the boundary lies OUTSIDE the reference;
  4. the local vessel radius at each boundary point: the Radius of the nearest point of the
     --boundary run's centerline (05_branch_tree.vtk).

If the median distance is about 7-9 % of the radius, the radius excess comes from the
reference, not from the method.

Outputs in OUT_DIR (never overwritten without --force):
  boundary_vs_reference_<dataset>.csv       one row per patient, structure and radius bin
                                            (<1, 1-2, 2-3, >3 mm, all): n, median, q25, q75, mean of
                                            the distance (mm) and of distance / radius
  boundary_vs_reference_<dataset>_fine.csv  the same in 0.25 mm radius bins (for the figure)
  boundary_vs_reference_<dataset>.png       distance vs radius per structure (median of the cases,
                                            interquartile band across cases), in mm and in % of radius
  <same>.run.json                           command, date, git commit, inputs

--write-surfaces DIR also writes the boundary as a surface (flying edges at m = 0 on the margin,
largest component, no smoothing, no decimation, LPS) as
DIR/<dataset>/<patient>/final_export_<structure>/supporting_files/01_preprocessed.vtk: a
results-like tree that compare_centerlines.py --reference-surfaces DIR takes as the reference
instead (the "pipeline's own boundary" column). About 25-30 MB per structure on the CT grid.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import vtk
from scipy import ndimage
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # the repo root

import datasets_file                                                   # noqa: E402  (this folder)
from vescan_field.field import load_reference, margin_from_store
from vescan_field.main import usable_cpus
from vescan_field.stages.centerlines.boundary import Boundary
from vescan_field.store import Store
from vescan_field.surface import (close_if_open, ensure_outward_normals, keep_largest_connected_component,
                                  load_surface, point_array, points, save_surface)

BINS = [(0.0, 1.0, "<1"), (1.0, 2.0, "1-2"), (2.0, 3.0, "2-3"), (3.0, np.inf, ">3"), (0.0, np.inf, "all")]
FINE_STEP, FINE_MAX = 0.25, 8.0
logger = logging.getLogger("boundary_vs_reference")


# --------------------------------------------------------------------------- one structure
def largest_component(m):
    """The margin with every positive component but the largest made slightly negative, so that
    its zero crossings are those of the largest component only (26-connectivity)."""
    labels, n = ndimage.label(m > 0, structure=np.ones((3, 3, 3), bool))
    if n <= 1:
        return m
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = labels == int(np.argmax(sizes))
    return np.where(keep | (m <= 0), m, np.float32(-1e-3)).astype(np.float32)


def field_margin(patient_dir, store_name, classes, pad):
    """VeScan-field's margin: resampled onto the CT grid, cropped. Returns (margin, spacing,
    to_lps), with to_lps taking points in the crop's (z,y,x) mm."""
    ct_geom, mask = load_reference(patient_dir / f"{classes[0]}.nii.gz")
    store = Store(patient_dir / store_name)
    try:
        m, g, _, _ = margin_from_store(store, [store.class_index[c] for c in classes], ct_geom,
                                       mask.shape, pad_voxels=pad)
    finally:
        store.close()
    return m, g.spacing, g.mm_to_lps


def thalweg_margin(patient_dir, store_name, classes, pad=2):
    """VeScan-thalweg's margin: on the store's own grid (vescan_thalweg.margin.union_margin),
    cropped around the structure. Returns (margin, spacing = 1, to_lps), with to_lps taking
    points in the crop's (z,y,x) indices: LPS = origin + index @ directions (rankfield)."""
    from thalweg.store import open_store
    from vescan_thalweg.margin import union_margin
    store = open_store(patient_dir / store_name)
    try:
        m, geometry, _ = union_margin(store, classes)
    finally:
        if hasattr(store, "close"):
            store.close()
    sl = ndimage.find_objects((m > 0).astype(np.int8))
    if not sl or sl[0] is None:
        raise ValueError("empty structure")
    lo = np.array([max(s.start - pad, 0) for s in sl[0]])
    hi = np.array([min(s.stop + pad, n) for s, n in zip(sl[0], m.shape)])
    m = np.ascontiguousarray(m[tuple(slice(a, b) for a, b in zip(lo, hi))])
    origin = np.asarray(geometry.origin, float)
    dirs = np.asarray(geometry.directions, float)            # one row per array axis, LPS mm per index
    return m, np.ones(3), lambda idx: origin + (np.asarray(idx, float) + lo) @ dirs


def zero_surface(m, spacing, to_lps):
    """Flying edges at m = 0 on the margin crop, in LPS, largest component, outward normals."""
    img = vtk.vtkImageData()
    img.SetDimensions(m.shape[2], m.shape[1], m.shape[0])                  # VTK x = numpy's last axis
    img.SetSpacing(*spacing[::-1])
    img.GetPointData().SetScalars(numpy_to_vtk(np.ascontiguousarray(m).ravel(), deep=True))
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(img)
    fe.SetValue(0, 0.0)
    fe.ComputeNormalsOff()
    fe.ComputeGradientsOff()
    fe.ComputeScalarsOff()
    fe.Update()
    poly = fe.GetOutput()
    xyz = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(to_lps(xyz[:, ::-1]), deep=True))           # (x,y,z) -> crop (z,y,x) -> LPS
    poly.SetPoints(pts)
    largest = keep_largest_connected_component(poly)
    only_polys = vtk.vtkPolyData()                    # cleaning leaves a few degenerate lines/vertices
    only_polys.SetPoints(largest.GetPoints())
    only_polys.SetPolys(largest.GetPolys())
    return ensure_outward_normals(only_polys)


def signed_distance(surface, xyz):
    """Signed distance to a closed surface, positive outside (vtkImplicitPolyDataDistance)."""
    f = vtk.vtkImplicitPolyDataDistance()
    f.SetInput(surface)
    out = vtk.vtkDoubleArray()
    f.FunctionValue(numpy_to_vtk(np.ascontiguousarray(xyz, np.float64), deep=True), out)
    return vtk_to_numpy(out).copy()


def centerline_radius(path):
    poly = load_surface(path)
    r = point_array(poly, "Radius")
    if r is None:
        r = point_array(poly, "MaximumInscribedSphereRadius")
    return points(poly), np.asarray(r, float)


def bin_stats(d, rel, r, lo, hi):
    sel = (r >= lo) & (r < hi)
    if not sel.any():
        return dict(n=0)
    x, y = d[sel], rel[sel]
    return dict(n=int(sel.sum()), radius_median=float(np.median(r[sel])),
                dist_median=float(np.median(x)), dist_q25=float(np.percentile(x, 25)),
                dist_q75=float(np.percentile(x, 75)), dist_mean=float(x.mean()),
                rel_median=float(np.median(y)), rel_q25=float(np.percentile(y, 25)),
                rel_q75=float(np.percentile(y, 75)), rel_mean=float(y.mean()))


def measure(job):
    """All the rows of one (patient, structure)."""
    (dataset, patient, patient_dir, store_name, structure, classes, ref_dir, cl_dir, kind, pad, surf_dir) = job
    if kind == "field":
        m, spacing, to_lps = field_margin(patient_dir, store_name, classes, pad)
    else:
        m, spacing, to_lps = thalweg_margin(patient_dir, store_name, classes)
    m = largest_component(m)
    boundary = Boundary(m, spacing, 0.0)
    xyz = to_lps(boundary.points)

    reference = close_if_open(load_surface(ref_dir / "supporting_files" / "01_preprocessed.vtk"),
                              f"{patient}/{structure} 01_preprocessed.vtk")
    d = signed_distance(reference, xyz)
    cl_xyz, cl_r = centerline_radius(cl_dir / "supporting_files" / "05_branch_tree.vtk")
    if np.median(signed_distance(reference, cl_xyz)) > 0:      # the centerline must read inside
        raise RuntimeError("centerline reads outside the reference: normals inverted?")
    r = cl_r[cKDTree(cl_xyz).query(xyz)[1]]
    rel = d / np.maximum(r, 1e-6)

    base = dict(dataset=dataset, patient=patient, structure=structure, boundary=kind)
    rows = [dict(base, bin=name, **bin_stats(d, rel, r, lo, hi)) for lo, hi, name in BINS]
    fine = []
    for lo in np.arange(0.0, FINE_MAX, FINE_STEP):
        s = bin_stats(d, rel, r, lo, lo + FINE_STEP)
        if s["n"]:
            fine.append(dict(base, r_lo=lo, r_hi=lo + FINE_STEP, **s))

    if surf_dir is not None:
        out = surf_dir / f"final_export_{structure}" / "supporting_files" / "01_preprocessed.vtk"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_surface(zero_surface(m, spacing, to_lps), out)
    return rows, fine


# --------------------------------------------------------------------------- report
def figure(fine, path, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(fine)
    df = df[df.n >= 50]                                           # bins with too few points are noise
    structures = [s for s in ("artery", "veins", "airways") if s in set(df.structure)]
    fig, axes = plt.subplots(2, len(structures), figsize=(4.2 * len(structures), 6.6), squeeze=False)
    for j, st in enumerate(structures):
        g = df[df.structure == st].groupby("r_lo")
        x = np.array(sorted(g.groups)) + FINE_STEP / 2
        for i, (col, scale, ylabel) in enumerate((("dist_median", 1, "boundary - reference (mm)"),
                                                   ("rel_median", 100, "same, % of the local radius"))):
            q = g[col].quantile([0.25, 0.5, 0.75]).unstack().reindex(sorted(g.groups)) * scale
            keep = x >= (0.5 if col == "rel_median" else 0.0)      # % of a radius near 0 (tips) blows up
            ax = axes[i, j]
            ax.fill_between(x[keep], q[0.25][keep], q[0.75][keep], color="#2a78d6", alpha=0.25, lw=0)
            ax.plot(x[keep], q[0.5][keep], color="#2a78d6", lw=2)
            ax.axhline(0, color="#52514e", lw=0.8)
            ax.set_xlabel("local radius (mm)")
            ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(f"{st} ({df[df.structure == st].patient.nunique()} cases)")
    kind = df.boundary.iloc[0] if len(df) else "?"
    fig.suptitle(f"{dataset}: {kind} boundary vs 01_preprocessed.vtk (positive = {kind} outside); "
                 "median of the per-case medians, band = IQR across cases", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def write_run_info(path, inputs, failed):
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    info = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), command=sys.argv,
                git_commit=commit, inputs=inputs, failed=failed)
    path.write_text(json.dumps(info, indent=2))


def load_config(kind, path):
    """The dataset config of the boundary's pipeline (store, classes, patients) and its patient iterator."""
    if kind == "field":
        from vescan_field import config as cfgmod
        from vescan_field.main import iter_patients
    else:
        from vescan_thalweg import config as cfgmod
        from vescan_thalweg.main import iter_patients
    return cfgmod.load(path), iter_patients


def dataset_jobs(dataset, kind, config_path, ref_root, cl_root, surf_root, args, exclude, patients_root=None):
    """The jobs of one dataset. ref_root / cl_root: results roots (one folder per patient).
    patients_root: the patient folders (masks, store) instead of the config's patients_root_dir."""
    cfg, iter_patients = load_config(kind, config_path)
    if patients_root is not None:
        cfg.batch.patients_root_dir, cfg.batch.enabled = str(patients_root), True
    pad = cfg.pipeline.field.pad_voxels if kind == "field" else None
    structures = {s: c for s, c in cfg.enabled_structures().items()
                  if not args.structures or s in args.structures}
    jobs = []
    for patient, patient_dir, _, _ in iter_patients(cfg.batch):
        if args.patients and patient not in args.patients or patient in exclude:
            continue
        for structure, classes in structures.items():
            ref = ref_root / patient / f"final_export_{structure}"
            cl = cl_root / patient / f"final_export_{structure}"
            needed = [ref / "supporting_files" / "01_preprocessed.vtk", cl / "supporting_files" / "05_branch_tree.vtk"]
            missing = [str(p) for p in needed if not p.exists()]
            if missing:
                print(f"  skip {dataset}/{patient}/{structure}: missing {missing}")
                continue
            surf = None if surf_root is None else surf_root / dataset / patient
            jobs.append((dataset, patient, patient_dir, cfg.batch.store, structure, classes, ref, cl,
                         kind, pad, surf))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boundary", choices=("thalweg", "field"), default="thalweg",
                    help="whose margin gives the boundary (and whose centerline the local radius); default thalweg")
    ap.add_argument("--datasets", required=True, help="datasets file (configs/analysis/*.json)")
    ap.add_argument("--reference", default="vmtk", help="whose 01_preprocessed.vtk is the reference (default vmtk)")
    ap.add_argument("--write-surfaces", help="also write the boundary as a surface under DIR/<dataset>/<patient>/")
    ap.add_argument("--structures", nargs="*")
    ap.add_argument("--patients", nargs="*")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=0, help="processes (default: the usable cores)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-o", "--output", required=True, help="output folder")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    out = Path(args.output)
    surf_root = Path(args.write_surfaces) if args.write_surfaces else None
    exclude = set(args.exclude)
    kind = args.boundary
    jobs, inputs = [], {}
    datasets, labels, file_exclude = datasets_file.load(args.datasets)
    exclude |= file_exclude
    if args.reference not in labels or kind not in labels:
        ap.error(f"the datasets file needs the runs {args.reference!r} and {kind!r} (it has {labels})")
    for name, d in datasets.items():
        config = d["configs"].get(kind)
        if config is None:
            print(f"[{name}] no {kind} config: skipped")
            continue
        ref_root, cl_root = d["runs"][args.reference], d["runs"][kind]
        jobs += dataset_jobs(name, kind, config, ref_root, cl_root, surf_root, args, exclude, d["patients_root"])
        inputs[name] = dict(boundary=kind, config=str(config), reference_root=str(ref_root),
                            centerline_root=str(cl_root),
                            patients_root=str(d["patients_root"]) if d["patients_root"] else None)
    for name in inputs:
        csv = out / f"boundary_vs_reference_{name}.csv"
        if csv.exists() and not args.force:
            sys.exit(f"{csv} exists: choose another -o or use --force")
    if not jobs:
        sys.exit("nothing to measure")

    workers = args.workers or usable_cpus()
    rows, fine, failed = [], [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(workers,)) as pool:
        futures = {pool.submit(measure, job): job for job in jobs}
        for n, fut in enumerate(as_completed(futures), start=1):
            job = futures[fut]
            try:
                r, f = fut.result()
                rows += r
                fine += f
            except Exception as exc:                              # one bad case must not stop the rest
                failed.append(f"{job[0]}/{job[1]}/{job[4]}: {exc!r}")
            print(f"\r{n}/{len(jobs)} done, {len(failed)} failed", end="", flush=True)
    print()
    for f in failed:
        print("  failed", f)
    if not rows:
        sys.exit("every case failed")

    import pandas as pd
    out.mkdir(parents=True, exist_ok=True)
    all_rows, all_fine = pd.DataFrame(rows), pd.DataFrame(fine)
    for name in inputs:                                           # one set of files per dataset
        df = all_rows[all_rows.dataset == name].sort_values(["patient", "structure"])
        if df.empty:
            continue
        csv = out / f"boundary_vs_reference_{name}.csv"
        df.to_csv(csv, index=False)
        f = all_fine[all_fine.dataset == name]
        f.sort_values(["patient", "structure", "r_lo"]).to_csv(out / f"boundary_vs_reference_{name}_fine.csv", index=False)
        figure(f.to_dict("records"), out / f"boundary_vs_reference_{name}.png", name)
        write_run_info(csv.with_suffix(".run.json"),
                       dict(inputs[name], write_surfaces=str(surf_root / name) if surf_root else None),
                       [x for x in failed if x.startswith(f"{name}/")])

        summary = df.groupby(["structure", "bin"])[["dist_median", "rel_median"]].median()
        print(f"\n{name}: {kind} boundary, median over the cases of the per-case median "
              f"(positive = outside the reference)")
        print((summary.assign(rel_median=summary.rel_median * 100)
               .rename(columns={"dist_median": "distance mm", "rel_median": "% of radius"})).round(3).to_string())
    print(f"-> {out}/")


def _init_worker(workers):
    from vescan_field import threads
    threads.KDTREE_WORKERS = 1 if workers > 1 else -1
    os.environ.setdefault("OMP_NUM_THREADS", "1")


if __name__ == "__main__":
    main()
