"""[A] Reader of the haversack ranked store (`lung_vessels.duckn.zip`) with zarr + numpy only.

The only module that knows the format. It follows the README inside the store (sec. 2, 4, 6):
- `ranks[j]` = class + 1 of the j-th best class (0 = absent), `ranks[0]` is the argmax;
- `support[j-1]` = byte of the gap (in logits) of class `ranks[j]` from the winner, decoded
  with the `levels(meta)` table; a class that is not named is `-clip`;
- geometry: LPS = space_origin + sum_a index[a] * space_direction[a], axes in array order.

`levels` reproduces `rankfield.code.levels` (rankfield 0.3.11, Apache-2.0, M. Halle): rankfield
needs Python >= 3.12 and cannot be installed in the vmtk-build env.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SUPPORT_MAX = 255


def levels(meta: dict) -> np.ndarray:
    """Table byte -> gap from the winner (logits), 256 float32: as rankfield.levels."""
    curve = str(meta.get("gap_curve", "uniform"))
    rng = float(meta.get("gap_range", meta["clip"]))
    s = np.arange(256, dtype=np.float64) / SUPPORT_MAX
    if curve == "uniform":
        g = (1.0 - s) * rng
    elif curve == "log":
        origin = float(meta.get("gap_origin", 0.5))
        g = origin * np.expm1((1.0 - s) * np.log1p(rng / origin))
    else:
        raise ValueError(f"unknown gap_curve: {curve}")
    return g.astype(np.float32)


def decode_deficit(ranks: np.ndarray, support: np.ndarray, k: int, lut: np.ndarray,
                   clip: float) -> np.ndarray:
    """Deficit d_k = l_k - max_j l_j of one class (0 where it wins, -clip where it is not named)."""
    d = np.where(ranks[0] == k + 1, np.float32(0.0), np.float32(-clip))
    for j in range(1, ranks.shape[0]):
        hit = ranks[j] == k + 1
        d[hit] = -lut[support[j - 1][hit]]
    return d


class Store:
    """One part (parts/0) of the store: metadata, classes, geometry, box reads."""

    def __init__(self, path, part: str = "parts/0"):
        import zarr
        self.path = Path(path)
        self._zs = zarr.storage.ZipStore(str(path), mode="r")
        root = zarr.open_group(self._zs, mode="r")
        self.part = root[part]
        self.meta = self.part.attrs["duckn"]["extensions"]["ranked"]
        if self.meta.get("mode", "ranked") != "ranked":
            raise ValueError(f"store is not ranked: {self.meta.get('mode')}")
        self.K = int(self.meta["classes"])
        self.clip = float(self.meta["clip"])
        self.lut = levels(self.meta)
        self.ranks = self.part["ranks"]
        self.support = self.part["support"]
        self.shape = tuple(self.ranks.shape[1:])

        g = self.ranks.attrs["duckn"]
        if g.get("space", "left-posterior-superior") != "left-posterior-superior":
            raise ValueError(f"space is not LPS: {g.get('space')}")
        self.origin = np.array(g["space_origin"], float)
        # columns = LPS vectors of the array axes (z,y,x); their length is the spacing
        self.axes = np.array([a["space_direction"] for a in g["axes"] if a["kind"] == "space"],
                             float).T

        # names -> channel, and extent (inclusive indices) from the root's segments
        labels = list(self.meta["labels"])
        self.class_index, self.extent = {}, {}
        for s in root.attrs["duckn"]["extensions"]["seg"]["segments"]:
            c = labels.index(s["label_values"][0])
            self.class_index[s["name"]] = c
            if "extent" in s:
                self.extent[c] = np.array(s["extent"], int).reshape(3, 2)

    def close(self):
        self._zs.close()

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
            nz = np.argwhere(np.asarray(self.ranks[0]) == c + 1)
            if not len(nz):
                raise ValueError(f"class {c} not in the store")
            lo, hi = nz.min(0), nz.max(0) + 1
        return np.maximum(lo - pad, 0), np.minimum(hi + pad, np.array(self.shape))

    def read_box(self, lo, hi):
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        return np.asarray(self.ranks[(slice(None),) + sl]), np.asarray(self.support[(slice(None),) + sl])

    def deficits(self, lo, hi) -> np.ndarray:
        """Deficits of all K classes on the box, (K, z, y, x) float32."""
        ranks, support = self.read_box(lo, hi)
        return np.stack([decode_deficit(ranks, support, k, self.lut, self.clip)
                         for k in range(self.K)])

    def labelmap(self, lo=None, hi=None) -> np.ndarray:
        """The model labels (ranks[0] - 1), exact: do not derive them from argmax(deficit)."""
        if lo is None:
            return np.asarray(self.ranks[0]).astype(np.int16) - 1
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        return np.asarray(self.ranks[(0,) + sl]).astype(np.int16) - 1
