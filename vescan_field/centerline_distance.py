"""The nearest centerline segment to each point (of a surface), without VMTK.

With the radius it is vtkvmtkPolyBallLine::EvaluateFunction (vmtk/vtkVmtk/ComputationalGeometry/
vtkvmtkPolyBallLine.cxx), the metric VeScan uses to transfer labels: for each segment P0-P1
with radii r0-r1, t from the 4-component "complex" product (x.y - x3*y3) of (P1-P0, r1-r0)
and (p-P0, -r0), t clamped to [0, 1] (tolerance 1e-12), and
    f = |p - c(t)|^2 - r(t)^2.
The minimum wins, on a tie the first in (cell, segment) order. Weighting by the radius matters
after a bifurcation: a point on the wall of a wide trunk can be closer, in plain distance, to
the centerline of a thin branch next to it; with f the trunk wins.
Without the radius (r = 0) f is the squared distance to the segment (vtkStaticCellLocator).

Vectorised and EXACT: a segment can beat a point's best candidate f0 only if
    |p - mid| - L/2 < sqrt(f0 + r_max^2)
(f >= distance^2 - r_max^2, and distance >= |p - mid| - L/2), so for each radius band a
KD-tree of the midpoints queried with that radius is enough.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from vescan_field import threads

TOL = 1e-12                    # VTK_VMTK_DOUBLE_TOL
RADIUS_BANDS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf]


class Segments:
    """Every segment of a centerline, in (cell, segment) order."""

    def __init__(self, xyz, cells, radius=None):
        a = np.concatenate([ids[:-1] for ids in cells]) if cells else np.zeros(0, int)
        b = np.concatenate([ids[1:] for ids in cells]) if cells else np.zeros(0, int)
        self.cell = np.concatenate([np.full(len(ids) - 1, c) for c, ids in enumerate(cells)]).astype(int)
        self.sub = np.concatenate([np.arange(len(ids) - 1) for ids in cells]).astype(int)
        self.p0, self.p1 = xyz[a], xyz[b]
        r = np.zeros(len(xyz)) if radius is None else np.asarray(radius, float)
        self.r0, self.r1 = r[a], r[b]
        self.mid = 0.5 * (self.p0 + self.p1)
        self.half_length = 0.5 * np.linalg.norm(self.p1 - self.p0, axis=1)
        self.r_max = np.maximum(self.r0, self.r1)
        # point -> the segment starting and the one ending there (-1 if none)
        self.starting, self.ending = np.full(len(xyz), -1), np.full(len(xyz), -1)
        self.starting[a], self.ending[b] = np.arange(len(a)), np.arange(len(b))

    def evaluate(self, pts, seg):
        """(f, t) for the pairs point pts[k] - segment seg[k], operation by operation as VMTK."""
        p0, p1, r0, r1 = self.p0[seg], self.p1[seg], self.r0[seg], self.r1[seg]
        v0, v0r = p1 - p0, r1 - r0
        v1, v1r = pts - p0, -r0
        num = v0[:, 0] * v1[:, 0] + v0[:, 1] * v1[:, 1] + v0[:, 2] * v1[:, 2] - v0r * v1r
        den = v0[:, 0] * v0[:, 0] + v0[:, 1] * v0[:, 1] + v0[:, 2] * v0[:, 2] - v0r * v0r
        degenerate = np.abs(den) < TOL
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / den
        lo, hi = t < TOL, ~(t < TOL) & (1.0 - t < TOL)
        t = np.where(lo, 0.0, np.where(hi, 1.0, t))
        c = np.where(lo[:, None], p0, np.where(hi[:, None], p1, p0 + t[:, None] * v0))
        cr = np.where(lo, r0, np.where(hi, r1, r0 + t * v0r))
        d = pts - c
        f = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2] - cr * cr
        return np.where(degenerate, np.inf, f), t


def nearest_segments(points, xyz, cells, radius=None, chunk=20000):
    """For each point: (cell, segment in the cell, t on the segment, f) of the winning segment."""
    seg = Segments(xyz, cells, radius)
    points = np.asarray(points, float)
    n = len(points)
    out_seg = np.full(n, -1)
    out_t, out_f = np.zeros(n), np.full(n, np.inf)
    if len(seg.cell) == 0:
        return np.full(n, -1), np.full(n, -1), out_t, out_f

    # f0: an upper bound, from the segments of the nearest centerline point
    # (a ~0-length segment, e.g. a repeated point at the end of a cell, gives inf: the segments
    # touching the nearest vertices are tried until one is finite)
    k = min(8, len(xyz))
    vertices = cKDTree(xyz).query(points, k=k, workers=threads.KDTREE_WORKERS)[1].reshape(n, k)
    f0 = np.full(n, np.inf)
    for j in range(k):
        todo = np.isinf(f0)
        if not todo.any():
            break
        for touching in (seg.starting[vertices[todo, j]], seg.ending[vertices[todo, j]]):
            ok = touching >= 0
            rows = np.flatnonzero(todo)[ok]
            f0[rows] = np.minimum(f0[rows], seg.evaluate(points[rows], touching[ok])[0])

    bands, lo = [], 0.0
    for hi in RADIUS_BANDS:
        members = np.flatnonzero((seg.r_max >= lo) & (seg.r_max < hi))
        if len(members):
            bands.append((members, cKDTree(seg.mid[members]), seg.r_max[members].max(),
                          seg.half_length[members].max()))
        lo = hi

    for start in range(0, n, chunk):
        idx = np.arange(start, min(start + chunk, n))
        pair_pt, pair_seg = [], []
        for members, tree, r_max, half_len in bands:
            reach = np.sqrt(np.maximum(f0[idx] + r_max * r_max, 0.0)) + half_len + 1e-9
            for k, hits in enumerate(tree.query_ball_point(points[idx], reach, workers=threads.KDTREE_WORKERS)):
                if hits:
                    pair_pt.append(np.full(len(hits), idx[k]))
                    pair_seg.append(members[hits])
        if not pair_pt:
            continue
        pp, ss = np.concatenate(pair_pt), np.concatenate(pair_seg)
        f, t = seg.evaluate(points[pp], ss)
        order = np.lexsort((ss, f, pp))           # per point: minimum f, on a tie the first segment
        pp, ss, f, t = pp[order], ss[order], f[order], t[order]
        first = np.r_[True, pp[1:] != pp[:-1]]
        out_seg[pp[first]], out_f[pp[first]], out_t[pp[first]] = ss[first], f[first], t[first]

    found = out_seg >= 0
    cell, sub = np.full(n, -1), np.full(n, -1)
    cell[found], sub[found] = seg.cell[out_seg[found]], seg.sub[out_seg[found]]
    return cell, sub, out_t, out_f
