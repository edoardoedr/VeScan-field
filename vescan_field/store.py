"""[A] The haversack ranked store (`lung_vessels.duckn.zip`), read with rankfield (M. Halle,
Apache-2.0; https://github.com/mhalle/rankfield), the library that defines the format.

rankfield opens the store (`rankfield.store`: parts in paint order, each a lazy RankField on its
zarr planes, its geometry in LPS) and decodes it (`rankfield.deficit`). This module adds what
VeScan needs on top: the class names and extents from the root's seg segments, and the deficits
of every class on a box of the model grid only, decoding just that box's planes.

Deficit d_k = l_k - max_j l_j of class k: 0 where it wins, minus its gap where it trails, -clip
where it is not named. Geometry: LPS = origin + index @ directions, index in array order (z,y,x).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


def box_deficits(code, lo, hi) -> np.ndarray:
    """Deficits of all classes of RankField `code` on the box [lo, hi) of its grid, (K, z, y, x)
    float32: rankfield.deficit on a RankField holding only the box's planes (each plane read once)."""
    import rankfield as rf
    sl = (slice(None),) + tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    meta = {k: v for k, v in code.meta.items() if k != "shape"}      # the box's shape is its planes'
    box = dataclasses.replace(code, ranks=np.asarray(code.ranks[sl]), support=np.asarray(code.support[sl]),
                              tail=None, tails=None, meta=meta)       # the deficit reads ranks and support
    return np.stack([rf.deficit(box, k) for k in range(code.classes)]).astype(np.float32, copy=False)


class Store:
    """One part of the store (default: part 0): metadata, classes, geometry, box reads."""

    def __init__(self, path, part: int = 0):
        import rankfield.store as rs
        self.path = Path(path)
        root = rs.open_group(self.path)
        self._root = root
        parts = dict(zip(rs.part_indices(root), rs.read_parts(root)))
        if part not in parts:
            raise ValueError(f"{self.path.name} has no part {part} (parts: {sorted(parts)})")
        self.code = parts[part].field                      # lazy: planes read by box
        self.meta = self.code.meta
        self.K = self.code.classes
        self.clip = self.code.clip
        self.shape = tuple(int(v) for v in self.code.support.shape[1:])

        g = self.code.geometry
        self.origin = np.asarray(g.origin, float)
        # columns = LPS vectors of the array axes (z,y,x); their length is the spacing
        self.axes = np.asarray(g.directions, float).T

        # names -> channel, and extent (inclusive indices) from the root's segments
        labels = list(self.code.labels)
        self.class_index, self.extent = {}, {}
        for s in root.attrs.asdict()["duckn"]["extensions"]["seg"]["segments"]:
            if s["label_values"][0] not in labels:
                continue                                    # a segment of another part
            c = labels.index(s["label_values"][0])
            self.class_index[s["name"]] = c
            if "extent" in s:
                self.extent[c] = np.array(s["extent"], int).reshape(3, 2)

    def close(self):
        self._root.store.close()

    # ------------------------------------------------------------------ geometry
    def index_to_lps(self, idx_zyx: np.ndarray) -> np.ndarray:
        return self.origin + np.asarray(idx_zyx, float) @ self.axes.T

    def lps_to_index(self, lps: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.axes, (np.asarray(lps, float) - self.origin).T).T

    @property
    def spacing(self) -> np.ndarray:
        return np.linalg.norm(self.axes, axis=0)

    # ------------------------------------------------------------------ reading
    def class_box(self, c: int, pad: int = 3):
        """Box [lo, hi) of class c on the model grid, with padding."""
        if c in self.extent:
            ext = self.extent[c]
            lo, hi = ext[:, 0], ext[:, 1] + 1
        else:
            nz = np.argwhere(np.asarray(self.code.ranks[0]) == c + 1)
            if not len(nz):
                raise ValueError(f"class {c} not in the store")
            lo, hi = nz.min(0), nz.max(0) + 1
        return np.maximum(lo - pad, 0), np.minimum(hi + pad, np.array(self.shape))

    def deficits(self, lo, hi) -> np.ndarray:
        """Deficits of all K classes on the box, (K, z, y, x) float32."""
        return box_deficits(self.code, lo, hi)

    def labelmap(self, lo=None, hi=None) -> np.ndarray:
        """The model labels (ranks[0] - 1), exact: do not derive them from argmax(deficit)."""
        if lo is None:
            return np.asarray(self.code.ranks[0]).astype(np.int16) - 1
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        return np.asarray(self.code.ranks[(0,) + sl]).astype(np.int16) - 1
