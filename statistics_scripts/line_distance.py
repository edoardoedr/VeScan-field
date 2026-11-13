"""Plain distance from points to a set of polylines, with bounded memory, for the comparison scripts.

vescan_field.centerline_distance.nearest_segments is exact (and gives VMTK's radius-weighted
metric, which surface_labels needs), but it gathers every segment within a reach that grows with
the point's distance from the line: for the surface a pipeline does not reach (points tens of mm
away) that is hundreds of millions of point-segment pairs, several GB per call. The comparisons
need only the plain distance, so here:
  1. every segment is sampled every STEP mm (its ends included);
  2. each point takes the nearest sample (KD-tree, memory linear in the points);
  3. the distance is the exact one to that sample's segment, and to the segments of its few
     nearest samples (k = K), so the result is exact except where the true nearest segment has
     none of the k nearest samples, which needs a point much closer to it than STEP / 2.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

STEP = 0.1         # mm between samples along each segment
K = 4              # nearest samples whose segments are evaluated exactly
CHUNK = 200_000    # points per batch


class Lines:
    def __init__(self, xyz, cells, step=STEP):
        xyz = np.asarray(xyz, float)
        a = np.concatenate([np.asarray(c[:-1]) for c in cells if len(c) >= 2]) if cells else np.zeros(0, int)
        b = np.concatenate([np.asarray(c[1:]) for c in cells if len(c) >= 2]) if cells else np.zeros(0, int)
        self.p0, self.p1 = xyz[a], xyz[b]
        length = np.linalg.norm(self.p1 - self.p0, axis=1)
        n = np.maximum(np.ceil(length / step).astype(int), 1) + 1          # samples per segment, ends included
        self.sample_seg = np.repeat(np.arange(len(a)), n)
        start = np.repeat(np.cumsum(n) - n, n)
        t = (np.arange(n.sum()) - start) / np.repeat(n - 1, n)
        self.samples = self.p0[self.sample_seg] + t[:, None] * (self.p1 - self.p0)[self.sample_seg]
        self.tree = cKDTree(self.samples) if len(self.samples) else None

    def distance(self, pts, workers=1):
        pts = np.asarray(pts, float)
        if self.tree is None:
            return np.full(len(pts), np.inf)
        k = min(K, len(self.samples))
        out = np.empty(len(pts))
        for s in range(0, len(pts), CHUNK):
            p = pts[s:s + CHUNK]
            _, idx = self.tree.query(p, k=k, workers=workers)
            idx = idx.reshape(len(p), k)
            best = np.full(len(p), np.inf)
            for j in range(k):
                seg = self.sample_seg[idx[:, j]]
                p0, v = self.p0[seg], self.p1[seg] - self.p0[seg]
                vv = np.einsum("ij,ij->i", v, v)
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.clip(np.einsum("ij,ij->i", p - p0, v) / vv, 0.0, 1.0)
                t = np.where(vv > 0, t, 0.0)
                best = np.minimum(best, np.linalg.norm(p - (p0 + t[:, None] * v), axis=1))
            out[s:s + CHUNK] = best
        return out


def distance(pts, xyz, cells, workers=1):
    """Distance (mm) from each point to the nearest of the polylines `cells` (point ids into xyz)."""
    return Lines(xyz, cells).distance(pts, workers)
