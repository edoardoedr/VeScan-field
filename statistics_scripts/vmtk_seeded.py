#!/usr/bin/env python3
"""VMTK seeded at the thalweg tips vs its default seeds (NEXT.md 7.4), without the branch split.

    python statistics_scripts/vmtk_seeded.py --datasets configs/analysis/sample.json -o out.xlsx \
        [--structures ...] [--patients ...] [--workers N] [--save-centerlines DIR]

The question: is the surface VMTK leaves unreached (17-18 % on the arteries and veins) a matter of
its seeds or of the method? In Mike Halle's tests, VMTK seeded at all the field endpoints reached
50/51 and 55/57 of them.

Per patient and structure, on the vmtk run's 01_preprocessed.vtk (the surface VMTK works on), with
VeScan-vmtk's own functions (vescan_vmtk is imported, not changed) and its standard parameters:
  source   VMTK's start point (endpoints.detect: the widest point of the network), the same in both;
  default  targets = VMTK's automatic endpoints (the network leaves), as the pipeline does;
  seeded   targets = the tips of the thalweg tree (the far end of every leaf of 05_branch_tree.vtk,
           plus the root's free end), minus those farther from the surface than 1.5 x their branch
           radius (the filter VMTK applies to its own endpoints); each is snapped to the nearest
           surface vertex by the extraction itself.
Each run is vtkvmtkPolyDataCenterlines only (centerline.extract: capping, interior tessellation,
cost 1/R, pathological targets bisected out). No branch split, no groups: the raw centerline (one
source-to-target path per target, overlapping near the root) is compared as a set of lines.

Columns, for run in (default, seeded):
  targets_<run>, excluded_<run>     targets given; dropped by the bisection (each alone made the run
                                    return nothing: VMTK's "degenerate descent" failures);
  reached_<run>, reached_frac_<run> targets the centerline actually ends at (an end within
                                    1 mm + 2 x its radius of the target's surface point);
  time_<run>_s                      the extraction;
  length_<run>_mm                   the union of the paths (overlaps counted once, 0.5 mm grid);
  outside_frac_<run>                of the centerline, outside the surface;
  missing_area_pct_<run>            surface (% of the area) the run does not reach but thalweg does
                                    (distance > 2 x thalweg's + 1 mm), as compare_centerlines.py;
  missing_area_pct_thalweg_vs_<run> the reverse: surface thalweg does not reach but the run does;
  matched_<run>_in_thalweg, matched_thalweg_in_<run>   length within max(1 mm, lumen radius) of the other;
  dist_median_<run>_to_thalweg, dist_p95_<run>_to_thalweg
and missing_area_pct_both: surface thalweg reaches and neither VMTK run does (the best of the two).

Writes an .xlsx (per_case; summary = medians per dataset and structure) and <out>.run.json. Each
case is also appended to <out>.cases.jsonl as soon as it is done, failures included: if the run stops,
the same command resumes from there (only the missing cases are run; --retry-failed also reruns the
failed ones); --force starts again. Each case runs in its own process, killed after --timeout seconds
(default 3600; VMTK can stay inside one extraction indefinitely) and recorded as failed. With
--save-centerlines DIR, the two raw centerlines as DIR/<dataset>/<patient>/<structure>_{default,seeded}.vtp
(load them in Slicer next to the thalweg 05_branch_tree.vtk).
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))             # this folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # the repo root

import vtk  # noqa: E402  (vmtk's Python modules need vtk first)

import datasets_file  # noqa: E402
from compare_centerlines import load_tree, resample  # noqa: E402
from line_distance import distance as line_distance  # noqa: E402
from vescan_field.surface import (cell_areas, cell_point_ids, close_if_open, load_surface, point_array,  # noqa: E402
                                  points, points_inside, save_surface, triangles)

STEP_MM = 0.5
RUNS = ("default", "seeded")
logger = logging.getLogger("vmtk_seeded")


# --------------------------------------------------------------------------- seeds
def thalweg_tips(folder):
    """(tips, radii) in LPS: the far end of every leaf of the thalweg tree and the root's free end,
    with the leaf's (root's) average radius."""
    topo = json.loads((folder / "supporting_files" / "05_branch_tree_topology.json").read_text())
    nodes = {int(k): v for k, v in topo["nodes"].items()}
    children, parent = {}, {}
    for e in topo["edges"]:
        children.setdefault(e["parent"], []).append(e["child"])
        parent[e["child"]] = e["parent"]
    tree = load_tree(folder)
    cell_of = dict(zip(tree["groups"], tree["cells"]))
    xyz = tree["xyz"]
    tips, radii = [], []
    for g, n in nodes.items():
        if n.get("blanked") or n.get("virtual") or g not in cell_of:
            continue
        ends = xyz[[cell_of[g][0], cell_of[g][-1]]]
        free = []
        if not children.get(g):                         # a leaf: the end away from its parent
            p = parent.get(g)
            anchor = xyz[cell_of[p][-1]] if p in cell_of else ends[0]
            free.append(ends[np.argmax(np.linalg.norm(ends - anchor, axis=1))])
        if g not in parent or parent[g] not in nodes or nodes[parent[g]].get("virtual"):
            kid = next((c for c in children.get(g, []) if c in cell_of), None)   # a root: the end away from its child
            anchor = xyz[cell_of[kid][0]] if kid is not None else ends[-1]
            free.append(ends[np.argmax(np.linalg.norm(ends - anchor, axis=1))])
        for p in free:
            tips.append(p)
            radii.append(float(n.get("avg_radius") or 0.0))
    return np.array(tips), radii


# --------------------------------------------------------------------------- one run
def run_vmtk(surface, source, targets, extraction_cfg):
    """vtkvmtkPolyDataCenterlines from source to targets (LPS). Returns (centerline, excluded, seconds)."""
    from vescan_vmtk.stages.centerlines import centerline as C
    excluded = []
    original = C._pathological_targets

    def counting(run, ids):                            # records the bisection's verdict, unchanged
        bad = original(run, ids)
        excluded.extend(bad)
        return bad

    C._pathological_targets = counting
    t0 = time.time()
    try:
        line, _ = C.extract(surface, [list(source)] + [list(t) for t in targets], extraction_cfg)
    finally:
        C._pathological_targets = original
    return line, len(set(excluded)), time.time() - t0


def lines(poly):
    """The centerline as (xyz, cells, radius)."""
    xyz = points(poly)
    radius = point_array(poly, "Radius")
    return xyz, [c for c in cell_point_ids(poly) if len(c) >= 2], np.zeros(len(xyz)) if radius is None else radius


def samples_of(xyz, cells, radius):
    """Points every STEP_MM along the cells, overlapping paths counted once (0.5 mm grid)."""
    s = resample(dict(xyz=xyz, cells=cells, radius=radius, bifurcation=[False] * len(cells)))["x"]
    if not len(s):
        return s
    _, keep = np.unique(np.round(s / STEP_MM).astype(np.int64), axis=0, return_index=True)
    return s[np.sort(keep)]


def reached(target_positions, xyz, cells, radius):
    """Targets the centerline ends at: an end within 1 mm + 2 x its radius of the target."""
    if not cells:
        return 0
    ends = np.array([c[-1] for c in cells] + [c[0] for c in cells])
    pos, r = xyz[ends], radius[ends]
    d = np.linalg.norm(np.asarray(target_positions)[:, None, :] - pos[None], axis=2) - 2 * r[None]
    return int((d.min(axis=1) <= 1.0).sum())


def distance_to(points_, xyz, cells):
    return line_distance(points_, xyz, cells)


# --------------------------------------------------------------------------- one case
def measure(job):
    dataset, patient, structure, vmtk_dir, thalweg_dir, save_dir = job
    from vescan_vmtk.config import EndpointsConfig, ExtractionConfig
    from vescan_vmtk.stages.centerlines import endpoints as E

    row = dict(dataset=dataset, patient=patient, structure=structure)
    surface_path = vmtk_dir / "supporting_files" / "01_preprocessed.vtk"
    surface = close_if_open(load_surface(surface_path), f"{patient}/{structure} 01_preprocessed.vtk")

    default = E.detect(surface_path, EndpointsConfig())              # LPS, source first
    source = default[0]
    tips, tip_radii = thalweg_tips(thalweg_dir)
    row["thalweg_tips"] = len(tips)
    seeded = E.near_surface(surface, [list(t) for t in tips], tip_radii,
                            EndpointsConfig().endpoint_distance_radius_factor)
    row["thalweg_tips_far_from_surface"] = len(tips) - len(seeded)
    targets = {"default": default[1:], "seeded": seeded}

    spts = points(surface)
    tri = triangles(surface)
    area = np.bincount(tri.ravel(), np.repeat(cell_areas(surface) / 3, 3), minlength=len(spts))
    total = area.sum()
    th = load_tree(thalweg_dir)
    th_samples = samples_of(th["xyz"], th["cells"], th["radius"])
    d_surf_th = distance_to(spts, th["xyz"], th["cells"])
    lumen = cKDTree(spts)
    reached_by = {}
    for name in RUNS:
        row[f"targets_{name}"] = len(targets[name])
        try:
            line, excluded, seconds = run_vmtk(surface, source, targets[name], ExtractionConfig())
        except Exception as exc:                                       # a failed run is a result too
            row[f"error_{name}"] = repr(exc)
            continue
        xyz, cells, radius = lines(line)
        row[f"excluded_{name}"] = excluded
        row[f"time_{name}_s"] = round(seconds, 1)
        row[f"reached_{name}"] = reached(targets[name], xyz, cells, radius) if len(targets[name]) else 0
        row[f"reached_frac_{name}"] = row[f"reached_{name}"] / len(targets[name]) if len(targets[name]) else np.nan
        s = samples_of(xyz, cells, radius)
        row[f"length_{name}_mm"] = round(len(s) * STEP_MM, 1)
        row[f"outside_frac_{name}"] = float(1 - points_inside(surface, s, 1e-3).mean()) if len(s) else np.nan

        d_surf = distance_to(spts, xyz, cells)
        reached_by[name] = d_surf
        row[f"missing_area_pct_{name}"] = float(100 * area[d_surf > 2 * d_surf_th + 1.0].sum() / total)
        row[f"missing_area_pct_thalweg_vs_{name}"] = float(100 * area[d_surf_th > 2 * d_surf + 1.0].sum() / total)

        d = distance_to(s, th["xyz"], th["cells"])
        row[f"matched_{name}_in_thalweg"] = float((d <= np.maximum(lumen.query(s)[0], 1.0)).mean())
        row[f"dist_median_{name}_to_thalweg"] = float(np.median(d))
        row[f"dist_p95_{name}_to_thalweg"] = float(np.percentile(d, 95))
        d = distance_to(th_samples, xyz, cells)
        row[f"matched_thalweg_in_{name}"] = float((d <= np.maximum(lumen.query(th_samples)[0], 1.0)).mean())

        if save_dir is not None:
            out = save_dir / dataset / patient / f"{structure}_{name}.vtp"
            out.parent.mkdir(parents=True, exist_ok=True)
            save_surface(line, out)
    if len(reached_by) == 2:                                           # thalweg reaches it, neither VMTK run does
        best = np.minimum(reached_by["default"], reached_by["seeded"])
        row["missing_area_pct_both"] = float(100 * area[best > 2 * d_surf_th + 1.0].sum() / total)
    return row


# --------------------------------------------------------------------------- report
COLS = ["targets", "excluded", "reached_frac", "missing_area_pct", "missing_area_pct_thalweg_vs",
        "matched_in_thalweg", "matched_thalweg_in", "dist_median_to_thalweg", "length_mm", "outside_frac", "time_s"]


def column(metric, run):
    return {"missing_area_pct_thalweg_vs": f"missing_area_pct_thalweg_vs_{run}",
            "matched_in_thalweg": f"matched_{run}_in_thalweg", "matched_thalweg_in": f"matched_thalweg_in_{run}",
            "dist_median_to_thalweg": f"dist_median_{run}_to_thalweg", "length_mm": f"length_{run}_mm",
            "time_s": f"time_{run}_s"}.get(metric, f"{metric}_{run}")


def write_report(rows, out):
    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["dataset", "patient", "structure"])
    numeric = df.select_dtypes("number").columns
    summary = df.groupby(["dataset", "structure"])[list(numeric)].median()
    summary.insert(0, "cases", df.groupby(["dataset", "structure"]).size())
    with pd.ExcelWriter(out) as xl:
        df.to_excel(xl, sheet_name="per_case", index=False)
        summary.reset_index().to_excel(xl, sheet_name="summary", index=False)
    for (dataset, structure), g in df.groupby(["dataset", "structure"]):
        print(f"\n{dataset} / {structure}: {len(g)} cases (medians); thalweg tips {g['thalweg_tips'].median():.0f}")
        print(f"  {'metric':28s} {'default':>10s} {'seeded':>10s}")
        for m in COLS:
            vals = [pd.to_numeric(g[column(m, r)], errors="coerce").median() if column(m, r) in g else np.nan
                    for r in RUNS]
            print(f"  {m:28s}" + "".join(f" {v:10.3f}" for v in vals))
        if "missing_area_pct_both" in g:
            print(f"  surface thalweg reaches and neither VMTK run does: {g['missing_area_pct_both'].median():.2f} %")
        errors = [c for c in g if c.startswith("error_")]
        for c in errors:
            n = g[c].notna().sum()
            if n:
                print(f"  {c}: {n} case(s)")
    print(f"\n-> {out}")


def _plain(x):
    """numpy scalars and other non-JSON values, for the .cases.jsonl lines."""
    return x.item() if isinstance(x, np.generic) else str(x)


def load_partial(path):
    """The cases already finished by an earlier, interrupted run (a truncated last line is dropped)."""
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def write_run_info(path, datasets_path, failed):
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    path.write_text(json.dumps(dict(created=datetime.datetime.now().isoformat(timespec="seconds"),
                                    command=sys.argv, git_commit=commit,
                                    datasets=json.loads(Path(datasets_path).read_text()), failed=failed), indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", required=True, help="datasets file with the runs vmtk and thalweg")
    ap.add_argument("-o", "--output", required=True, help="output .xlsx")
    ap.add_argument("--structures", nargs="*")
    ap.add_argument("--patients", nargs="*")
    ap.add_argument("--save-centerlines", help="also write the two raw centerlines under DIR/<dataset>/<patient>/")
    ap.add_argument("--workers", type=int, default=0, help="processes (default: the usable cores)")
    ap.add_argument("--timeout", type=float, default=3600,
                    help="seconds per case before its process is killed and the case recorded as failed (0: none)")
    ap.add_argument("--retry-failed", action="store_true", help="when resuming, rerun the cases that failed")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    out = Path(args.output)
    if out.exists() and not args.force:
        sys.exit(f"{out} exists: choose another -o or use --force")

    datasets, labels, exclude = datasets_file.load(args.datasets)
    if not {"vmtk", "thalweg"} <= set(labels):
        sys.exit(f"the datasets file needs the runs vmtk and thalweg (it has {labels})")
    save_dir = Path(args.save_centerlines) if args.save_centerlines else None
    jobs = []
    for name, d in datasets.items():
        runs = {k: d["runs"][k] for k in ("vmtk", "thalweg")}
        cases, _ = datasets_file.common_cases(runs, args.structures, args.patients, exclude)
        print(f"[{name}] {len(cases)} patient x structure in vmtk and thalweg")
        for patient, structure, folders in cases:
            needed = [folders["vmtk"] / "supporting_files" / "01_preprocessed.vtk",
                      folders["thalweg"] / "supporting_files" / "05_branch_tree.vtk"]
            if not all(p.exists() for p in needed):
                print(f"  skip {patient}/{structure}: missing {[str(p) for p in needed if not p.exists()]}")
                continue
            jobs.append((name, patient, structure, folders["vmtk"], folders["thalweg"], save_dir))
    if not jobs:
        sys.exit("nothing to run")

    partial = out.with_suffix(".cases.jsonl")                         # one line per finished case
    if args.force and partial.exists():
        partial.unlink()
    lines = load_partial(partial)                                     # finished cases and failures
    if args.retry_failed:
        lines = [r for r in lines if "failed" not in r]
    done = {(r["dataset"], r["patient"], r["structure"]) for r in lines}
    if done:
        jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
        print(f"resuming: {len(done)} case(s) already in {partial}, {len(jobs)} to run")
    rows = [r for r in lines if "failed" not in r]
    failed = [f"{r['dataset']}/{r['patient']}/{r['structure']}: {r['failed']}" for r in lines if "failed" in r]

    from vescan_field.main import usable_cpus
    workers = args.workers or usable_cpus()
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text("".join(json.dumps(r, default=_plain) + "\n" for r in lines))  # drops a truncated line
    n_failed = 0
    with partial.open("a") as log:
        def record(line):
            log.write(json.dumps(line, default=_plain) + "\n")
            log.flush()

        for n, (job, ok, value) in enumerate(run_jobs(jobs, workers, args.timeout), start=1):
            if ok:
                rows.append(value)
                record(value)
            else:                                                       # one bad case must not stop the rest
                n_failed += 1
                failed.append(f"{job[0]}/{job[1]}/{job[2]}: {value}")
                record(dict(dataset=job[0], patient=job[1], structure=job[2], failed=value))
                print(f"\n  failed {job[0]}/{job[1]}/{job[2]}: {value}")
            print(f"\r{n}/{len(jobs)} done, {n_failed} failed", end="", flush=True)
    print()
    for f in failed:
        print("  failed", f)
    if rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        write_report(rows, out)
        write_run_info(out.with_suffix(".run.json"), args.datasets, failed)


def _child(job, conn):
    """One case in its own process; sends (True, row) or (False, error)."""
    _init_worker()
    try:
        conn.send((True, measure(job)))
    except BaseException as exc:                                       # noqa: BLE001  (reported, not raised)
        conn.send((False, repr(exc)))
    finally:
        conn.close()


def run_jobs(jobs, workers, timeout):
    """Yields (job, ok, row or error) as the cases finish. One process per case, so a case stuck in
    VTK's C++ code (which a Python timer cannot interrupt) can be killed after `timeout` seconds."""
    import multiprocessing as mp
    from multiprocessing.connection import wait
    pending, running = list(jobs), {}                                  # process -> (job, pipe end, start)
    try:
        while pending or running:
            while pending and len(running) < workers:
                job = pending.pop(0)
                recv, send = mp.Pipe(duplex=False)
                proc = mp.Process(target=_child, args=(job, send), daemon=True)
                proc.start()
                send.close()
                running[proc] = (job, recv, time.time())
            wait([c for _, c, _ in running.values()] + [p.sentinel for p in running], timeout=10)
            for proc, (job, conn, t0) in list(running.items()):
                if conn.poll():                                        # a result, or the child's end closed
                    try:
                        ok, value = conn.recv()
                    except EOFError:
                        proc.join(5)
                        ok, value = False, f"worker died (exit code {proc.exitcode})"
                elif not proc.is_alive():
                    ok, value = False, f"worker died (exit code {proc.exitcode})"
                elif timeout and time.time() - t0 > timeout:
                    proc.kill()
                    ok, value = False, f"timeout: killed after {timeout:.0f} s"
                else:
                    continue
                proc.join(5)
                conn.close()
                del running[proc]
                yield job, ok, value
    finally:
        for proc in running:                                           # Ctrl+C: no orphan VMTK runs
            proc.kill()


def _init_worker():
    from vescan_field import threads
    threads.KDTREE_WORKERS = 1
    os.environ.setdefault("OMP_NUM_THREADS", "1")


if __name__ == "__main__":
    main()
