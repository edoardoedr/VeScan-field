"""[A] CT grid geometry and the margin of a class from the store, interpolated onto the CT grid.

Internal coordinates: mm in the crop's array order (z,y,x). Exports in LPS.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- geometry
@dataclass
class Geometry:
    spacing: np.ndarray                                                   # mm, order (z,y,x)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))       # LPS
    direction: np.ndarray = field(default_factory=lambda: np.eye(3))     # as SimpleITK
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, int))  # crop index

    def mm_to_lps(self, p_mm: np.ndarray) -> np.ndarray:
        idx = p_mm / self.spacing + self.offset          # indices (z,y,x) in the full volume
        ijk_mm = (idx * self.spacing)[:, ::-1]           # (x,y,z) in mm
        return self.origin + ijk_mm @ self.direction.T

    def index_to_lps(self, idx_zyx: np.ndarray) -> np.ndarray:
        """Crop indices (z,y,x) -> LPS."""
        return self.mm_to_lps(np.asarray(idx_zyx, float) * self.spacing)

    def crop_mm_to_full_mm(self, p_mm: np.ndarray) -> np.ndarray:
        return p_mm + self.offset * self.spacing

    def cropped(self, lo) -> "Geometry":
        return Geometry(self.spacing, self.origin, self.direction, self.offset + np.asarray(lo, int))

    def crop_origin_lps(self) -> np.ndarray:
        return self.index_to_lps(np.zeros((1, 3)))[0]


def load_reference(path):
    """Geometry and array (z,y,x) of an image (CT or segmentation) via SimpleITK."""
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    geom = Geometry(np.array(img.GetSpacing(), float)[::-1],
                    np.array(img.GetOrigin(), float),
                    np.array(img.GetDirection(), float).reshape(3, 3))
    return geom, sitk.GetArrayFromImage(img)


def crop_to_class(m: np.ndarray, geom: Geometry, pad: int = 4):
    sl = ndimage.find_objects((m > 0).astype(np.int8))
    if not sl or sl[0] is None:
        raise ValueError("empty class")
    lo = np.array([max(s.start - pad, 0) for s in sl[0]])
    hi = np.array([min(s.stop + pad, n) for s, n in zip(sl[0], m.shape)])
    crop = tuple(slice(a, b) for a, b in zip(lo, hi))
    return np.ascontiguousarray(m[crop]), geom.cropped(lo)


def interp(vol: np.ndarray, p_mm: np.ndarray, spacing, order=1, cval=None):
    mode = "nearest" if cval is None else "constant"
    return ndimage.map_coordinates(vol, (p_mm / spacing).T, order=order,
                                   mode=mode, cval=0.0 if cval is None else cval)


# --------------------------------------------------------------------------- store -> CT grid
def margin_from_store(store, classes, ct_geom: Geometry, ct_shape, mask=None,
                      pad_voxels: int = 3, slab: int = 16):
    """Margin of a set of classes S, m_S = max_{c in S} d_c - max_{k not in S} d_k
    (for one class: m_c = d_c - max_{k!=c} d_k), on the CT grid, cropped around S.

    The deficits of all classes are decoded on S's box on the model grid, interpolated
    (trilinear) at the LPS points of the CT voxels and combined AFTER the interpolation, as
    store_to_probabilities in batch_segment.py. Works in slabs of `slab` CT planes.
    If `mask` (full CT array, z,y,x) is given, argmax in S is compared with it (QA).
    Returns (margin float32, crop Geometry, qa dict, timings dict).
    """
    classes = [int(classes)] if np.isscalar(classes) else [int(c) for c in classes]
    t = {}
    t0 = time.time()
    boxes = [store.class_box(c, pad_voxels) for c in classes]
    lo_m = np.min([b[0] for b in boxes], 0)
    hi_m = np.max([b[1] for b in boxes], 0)
    deficits = store.deficits(lo_m, hi_m)                  # (K, z, y, x) float32 on the box
    t["store_decode"] = time.time() - t0

    # CT box: corners of the model box -> continuous CT indices
    corners = np.array([[a, b, c] for a in (lo_m[0], hi_m[0] - 1)
                        for b in (lo_m[1], hi_m[1] - 1) for c in (lo_m[2], hi_m[2] - 1)], float)
    ct_idx = lps_to_index(ct_geom, store.index_to_lps(corners))
    lo_c = np.maximum(np.floor(ct_idx.min(0)).astype(int) - pad_voxels, 0)
    hi_c = np.minimum(np.ceil(ct_idx.max(0)).astype(int) + 1 + pad_voxels, np.array(ct_shape))
    g = ct_geom.cropped(lo_c)
    shape = tuple(int(x) for x in hi_c - lo_c)
    logger.info(f"model box {lo_m}-{hi_m}, CT box {lo_c}-{hi_c} ({np.prod(shape) / 1e6:.1f} M voxels)")

    t0 = time.time()
    margin = np.empty(shape, np.float32)
    winner = np.empty(shape, np.bool_) if mask is not None else None
    yy, xx = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    for z0 in range(0, shape[0], slab):
        z1 = min(z0 + slab, shape[0])
        zz = np.arange(z0, z1)[:, None, None] + np.zeros_like(yy)[None]
        idx = np.stack([zz, np.broadcast_to(yy, zz.shape), np.broadcast_to(xx, zz.shape)], -1)
        mi = store.lps_to_index(g.index_to_lps(idx.reshape(-1, 3))) - lo_m   # in the model box
        d_in = np.full(len(mi), -np.inf, np.float32)
        other = np.full(len(mi), -np.inf, np.float32)
        for k in range(deficits.shape[0]):
            v = ndimage.map_coordinates(deficits[k], mi.T, order=1, mode="nearest")
            np.maximum(d_in if k in classes else other, v, out=d_in if k in classes else other)
        margin[z0:z1] = (d_in - other).reshape(z1 - z0, *shape[1:])
        if winner is not None:
            # argmax in S <=> d_S >= max of the others (ties: negligible)
            winner[z0:z1] = (d_in > other).reshape(z1 - z0, *shape[1:])
    t["resample_ct"] = time.time() - t0

    qa = {}
    if mask is not None:
        crop = tuple(slice(a, b) for a, b in zip(lo_c, hi_c))
        ref = mask[crop] > 0
        outside = int((mask > 0).sum() - ref.sum())
        tp = int((winner & ref).sum())
        fp = int((winner & ~ref).sum())
        fn = int((~winner & ref).sum()) + outside
        qa = dict(mask_voxels=int(ref.sum()) + outside, field_voxels=int(winner.sum()),
                  dice=2 * tp / max(2 * tp + fp + fn, 1),
                  disagree_frac_of_mask=(fp + fn) / max(tp + fn, 1),
                  mask_voxels_outside_box=outside)
        logger.info(f"QA argmax vs mask: dice {qa['dice']:.4f}, differing voxels "
                    f"{fp + fn} ({100 * qa['disagree_frac_of_mask']:.2f}% of the mask)")
    return margin, g, qa, t


def lps_to_index(geom: Geometry, lps: np.ndarray) -> np.ndarray:
    """LPS -> continuous indices (z,y,x) in the full volume of `geom` (ignores the crop offset)."""
    ijk = np.linalg.solve(geom.direction, (np.asarray(lps, float) - geom.origin).T).T
    return (ijk / geom.spacing[::-1])[:, ::-1]
