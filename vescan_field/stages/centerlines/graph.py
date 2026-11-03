"""[B] Graph of the voxels connected according to the field (from the earlier prototype, field_vessels.py)."""
from __future__ import annotations

import itertools

import numpy as np
from scipy import sparse

OFFSETS = [o for o in itertools.product((-1, 0, 1), repeat=3) if o > (0, 0, 0)]  # 13


def field_graph(m: np.ndarray, Dn: np.ndarray, spacing: np.ndarray, level: float = 0.0,
                epsilon: float = 1e-3):
    """Nodes = voxels with m>level. An edge between neighbours (26) if the field interpolated at
    the midpoint is > level too: diagonal-only contacts through negative field do not count."""
    inside = m > level
    nid = -np.ones(m.shape, np.int64)
    nid[inside] = np.arange(int(inside.sum()))
    rows, cols, w, ln = [], [], [], []
    for o in OFFSETS:
        base = [slice(1 if oi < 0 else 0, n - (1 if oi > 0 else 0)) for oi, n in zip(o, m.shape)]

        def sh(arr, c):
            return arr[tuple(slice(b.start + ci, b.stop + ci) for b, ci in zip(base, c))]

        ok = sh(inside, (0, 0, 0)) & sh(inside, o)
        corners = list(itertools.product(*[(0, oi) if oi else (0,) for oi in o]))
        if len(corners) > 2:  # diagonals: trilinear value at the midpoint
            mid = sum(sh(m, c) for c in corners) / len(corners)
            ok &= mid > level
        a, b = sh(nid, (0, 0, 0))[ok], sh(nid, o)[ok]
        length = float(np.linalg.norm(np.array(o) * spacing))
        dmid = 0.5 * (Dn[a] + Dn[b])
        rows.append(a), cols.append(b)
        ln.append(np.full(len(a), length))
        w.append(length / (dmid ** 2 + epsilon))   # paths far from the boundary are cheaper
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    N = len(Dn)
    Gw = sparse.csr_matrix((np.concatenate(w), (rows, cols)), shape=(N, N))
    Gl = sparse.csr_matrix((np.concatenate(ln), (rows, cols)), shape=(N, N))
    return Gw, Gl
