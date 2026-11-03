"""Preprocessing of a vessel surface (from vescan/stages/preprocess.py, i.e. Slicer's
ExtractCenterlineLogic.preprocess): largest component, decimation, clean, triangulation,
consistent normals, outward normals.

Decimation to a NUMBER OF POINTS. pyfqmr (FastQuadric, the same algorithm as Slicer's
decimation) takes a triangle target, but stops earlier when its error threshold, which grows
with `aggressiveness`, allows no more collapses: at VeScan's aggressiveness (4) the arteries
of PD-1-Lung-00043 do not go below ~72.8k points whatever the target (5000, 20000, 80000),
while above some aggressiveness the target is hit (80003 for 80000). decimate_to_points
bisects for the MINIMUM aggressiveness that reaches N points: the least distorted mesh among
those with N points.
"""
from __future__ import annotations

import logging
import time

import vtk

from vescan_field.surface import (arrays_to_polydata, ensure_outward_normals,
                             keep_largest_connected_component, polydata_to_arrays)

logger = logging.getLogger(__name__)

AGGRESSIVENESS_RANGE = (1.0, 12.0)


def _simplify(verts, faces, target_faces, aggressiveness, max_iterations=100):
    import pyfqmr
    s = pyfqmr.Simplify()
    s.setMesh(verts, faces)
    s.simplify_mesh(target_count=target_faces, aggressiveness=aggressiveness,
                    max_iterations=max_iterations, preserve_border=True, verbose=False)
    v, f, _ = s.getMesh()
    return v, f


def decimate_to_points(poly, n_points, tolerance=0.05, steps=8):
    """Surface with ~n_points points (within +tolerance) at the minimum pyfqmr aggressiveness
    that gets there. Returns (polydata, info) with info = points, aggressiveness, passes, seconds."""
    t0 = time.time()
    n_in = poly.GetNumberOfPoints()
    if n_in <= n_points:
        return poly, dict(points=n_in, aggressiveness=None, passes=0, seconds=0.0)
    verts, faces = polydata_to_arrays(poly)
    target_faces = max(int(round(2 * n_points)), 4)       # Euler: triangles ~ 2 x points
    limit = n_points * (1 + tolerance)
    passes = 0

    def run(a, max_iterations=100):
        nonlocal passes
        passes += 1
        return _simplify(verts, faces, target_faces, a, max_iterations)

    lo, hi = AGGRESSIVENESS_RANGE
    best = run(hi)
    if len(best[0]) > limit:                               # not even at the maximum: more iterations
        best = run(hi, max_iterations=1000)
        if len(best[0]) > limit:
            logger.warning(f"decimation stopped at {len(best[0])} points ({n_points} requested)")
        lo = hi
    best_a = hi
    for _ in range(steps):                                 # minimum aggressiveness that reaches N
        if hi - lo < 0.05:
            break
        mid = 0.5 * (lo + hi)
        v, f = run(mid)
        if len(v) <= limit:
            hi, best, best_a = mid, (v, f), mid
        else:
            lo = mid
    out = arrays_to_polydata(*best)
    # exact value: pyfqmr with a rounded aggressiveness gives a different mesh
    info = dict(points=out.GetNumberOfPoints(), aggressiveness=best_a, passes=passes,
                seconds=round(time.time() - t0, 2))
    logger.info(f"decimation: {n_in} -> {info['points']} points ({n_points} requested), "
                f"aggressiveness {best_a:.3f}, {passes} passes, {info['seconds']} s")
    return out, info


def preprocess(poly, target_points=80000, tolerance=0.05, decimate=True,
               keep_largest_component=True):
    """Returns (preprocessed surface, decimation info or None)."""
    if keep_largest_component:
        poly = keep_largest_connected_component(poly)
    info = None
    if decimate:
        poly, info = decimate_to_points(poly, target_points, tolerance)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(poly)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cleaner.GetOutputPort())
    tri.PassLinesOff()
    tri.PassVertsOff()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(tri.GetOutputPort())
    normals.SetAutoOrientNormals(1)
    normals.SetFlipNormals(0)
    normals.SetConsistency(1)
    normals.SplittingOff()
    normals.Update()
    return ensure_outward_normals(normals.GetOutput()), info
