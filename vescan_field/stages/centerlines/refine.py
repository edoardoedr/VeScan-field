"""[C] Ridge refinement and polyline resampling (from the earlier prototype, field_vessels.py)."""
from __future__ import annotations

import numpy as np

from vescan_field.field import interp

from .boundary import Boundary


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def tangents(p):
    return _unit(np.gradient(p, axis=0)) if len(p) > 1 else np.array([[1.0, 0, 0]])


def normal_frames(t):
    ref = np.where(np.abs(t[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    u = _unit(np.cross(t, ref))
    return u, np.cross(t, u)


def smooth_polyline(p, it=2):
    q = p.copy()
    for _ in range(it):
        q[1:-1] = 0.25 * q[:-2] + 0.5 * q[1:-1] + 0.25 * q[2:]
    return q


def ridge_refine(p, bnd: Boundary, m, spacing, iters=3, n=5):
    """Move each point (except the ends) to the maximum of the boundary distance in the plane
    orthogonal to the tangent: a shrinking grid search."""
    if len(p) < 3:
        return p
    q = p.copy()
    t = tangents(q)
    u, v = normal_frames(t)
    g = np.linspace(-1, 1, n)
    A, B = (x.ravel() for x in np.meshgrid(g, g, indexing="ij"))
    h = float(spacing.max())
    inner = slice(1, -1)
    for _ in range(iters):
        cand = q[inner, None, :] + h * (A[None, :, None] * u[inner, None, :]
                                        + B[None, :, None] * v[inner, None, :])
        flat = cand.reshape(-1, 3)
        d = np.where(interp(m, flat, spacing) > 0, bnd.distance(flat), -np.inf)
        best = d.reshape(len(cand), -1).argmax(1)
        q[inner] = cand[np.arange(len(cand)), best]
        h /= 2.5
    return q


def resample_polyline(p, step):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    si = np.linspace(0, s[-1], max(int(np.ceil(s[-1] / step)) + 1, 2))
    return np.stack([np.interp(si, s, p[:, k]) for k in range(3)], 1)
