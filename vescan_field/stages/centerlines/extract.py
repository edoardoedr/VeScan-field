"""[B]-[C] Centerline of a class from its margin (from `analyze_class` of the earlier prototype)."""
from __future__ import annotations

import logging
import time

import numpy as np

from vescan_field.field import Geometry, crop_to_class, interp

from .boundary import Boundary
from .graph import field_graph
from .refine import resample_polyline, ridge_refine, smooth_polyline
from .skeleton import paths_to_segments, teasar

logger = logging.getLogger(__name__)


def extract(m_full: np.ndarray, geom: Geometry, delta=2.0, min_branch_mm=3.0,
            teasar_scale=1.5, teasar_const=0.5, min_voxels=30, step_mm=0.5,
            crop_pad_voxels=4, boundary_neighbours=8, graph_epsilon=1e-3,
            smooth_iterations_before=2, smooth_iterations_after=1, ridge_iterations=3, ridge_grid=5,
            largest_component_only=False):
    """Centerline of the margin `m_full` (positive inside, zero on the boundary); with
    largest_component_only, of its largest connected component only.

    Returns a dict with: segments (points_mm in the crop, radius, radius_lo, radius_hi,
    length_mm), roots_mm, the crop's margin and geom, timings, counts.
    """
    t = {}
    t0 = time.time()
    m, g = crop_to_class(m_full, geom, pad=crop_pad_voxels)
    sp = g.spacing
    b0 = Boundary(m, sp, 0.0, boundary_neighbours)
    b_in = Boundary(m, sp, +delta, boundary_neighbours)   # "tight" boundary: minimum radius
    b_out = Boundary(m, sp, -delta, boundary_neighbours)  # "loose" boundary: maximum radius
    t["boundary"] = time.time() - t0

    t0 = time.time()
    coords_mm = np.argwhere(m > 0) * sp
    Dn = b0.distance(coords_mm)
    t["distance"] = time.time() - t0

    t0 = time.time()
    Gw, Gl = field_graph(m, Dn, sp, epsilon=graph_epsilon)
    t["graph"] = time.time() - t0

    t0 = time.time()
    paths, mains, roots = teasar(coords_mm, Dn, Gw, Gl, scale=teasar_scale, const=teasar_const,
                                 min_branch_mm=min_branch_mm, min_voxels=min_voxels,
                                 largest_component_only=largest_component_only)
    segs = paths_to_segments(paths, extra_keys=roots)
    t["teasar"] = time.time() - t0
    logger.info(f"voxels {len(Dn)}, components with a tree {len(mains)}, segments {len(segs)}")

    t0 = time.time()
    out = []
    for s in segs:
        p = coords_mm[s]
        if len(p) >= 3:
            p = smooth_polyline(p, smooth_iterations_before)
            p = ridge_refine(p, b0, m, sp, iters=ridge_iterations, n=ridge_grid)
            p = smooth_polyline(p, smooth_iterations_after)
        p = resample_polyline(p, step_mm)
        mv = interp(m, p, sp)
        out.append(dict(ends=(int(s[0]), int(s[-1])),   # voxel-graph nodes at the ends
                        points_mm=p,
                        radius=b0.distance(p),
                        radius_lo=np.where(mv > delta, b_in.distance(p), 0.0),
                        radius_hi=b_out.distance(p),
                        length_mm=float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())))
    t["refine"] = time.time() - t0

    counts = dict(voxels=int(len(Dn)), components=len(mains), segments=len(out),
                  boundary_points=int(len(b0.points)), graph_edges=int(Gl.nnz),
                  total_length_mm=float(sum(s["length_mm"] for s in out)))
    return dict(segments=out, roots=list(roots), roots_mm=coords_mm[roots], coords_mm=coords_mm,
                main_paths=[coords_mm[p] for p in mains],
                margin=m, geom=g, timings=t, counts=counts)
