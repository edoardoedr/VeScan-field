"""Sub-voxel implicit boundary and distance from it (from the earlier prototype, field_vessels.py)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from vescan_field import threads


class Boundary:
    """Implicit surface {m = level}: zero crossings on the grid edges + normals from the gradient."""

    def __init__(self, m: np.ndarray, spacing: np.ndarray, level: float = 0.0, neighbours: int = 8):
        f = m - level
        self.spacing = spacing
        self.neighbours = neighbours     # crossings used for the distance (median)
        pts = []
        for ax in range(3):
            s0 = [slice(None)] * 3
            s1 = [slice(None)] * 3
            s0[ax], s1[ax] = slice(0, -1), slice(1, None)
            a, b = f[tuple(s0)], f[tuple(s1)]
            cross = (a > 0) != (b > 0)
            idx = np.argwhere(cross).astype(np.float64)
            av, bv = a[cross], b[cross]
            idx[:, ax] += av / (av - bv)
            pts.append(idx)
        idx = np.concatenate(pts)
        self.points = idx * spacing
        if len(idx):
            grad = np.gradient(f, *spacing)
            n = np.stack([ndimage.map_coordinates(g, idx.T, order=1) for g in grad], 1)
            self.normals = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
            self.tree = cKDTree(self.points)

    def distance(self, q: np.ndarray, k: int = None, chunk: int = 200_000) -> np.ndarray:
        """Distance (mm) from the boundary. For each of the k nearest crossings it uses the
        distance from its tangent plane (exact for a tube), or the Euclidean one if the point is
        too far sideways; the median over k damps the noise of the normals."""
        if len(self.points) == 0:
            return np.zeros(len(q))
        k = min(k or self.neighbours, len(self.points))
        out = np.empty(len(q))
        lat_max = 2.0 * self.spacing.max()
        for s in range(0, len(q), chunk):
            qq = q[s:s + chunk]
            dd, ii = self.tree.query(qq, k=k, workers=threads.KDTREE_WORKERS)
            dd, ii = dd.reshape(len(qq), k), ii.reshape(len(qq), k)
            diff = qq[:, None, :] - self.points[ii]
            dplane = np.abs(np.einsum("nkd,nkd->nk", diff, self.normals[ii]))
            lat = np.sqrt(np.maximum(dd ** 2 - dplane ** 2, 0.0))
            out[s:s + chunk] = np.median(np.where(lat <= lat_max, dplane, dd), 1)
        return out
