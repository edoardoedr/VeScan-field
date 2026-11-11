"""A structure's margin from the ranked store, on the model's own grid, for thalweg.

A structure is a set S of store classes (batch.structures.<s>.classes; e.g. airways = lumen +
wall). Its margin is the one VeScan field uses (vescan_field.field.margin_from_store), here on the
store's native grid instead of the CT's:

    m_S = max_{c in S} d_c - max_{k not in S} d_k        (d_k: deficit of class k, rankfield)

positive inside, zero on the boundary, clamped to +-clip as rankfield's margin (for one class it is
rankfield's own margin, checked equal on PD-1-Lung-00043). The result is
the (margin, geometry, ref) triple that thalweg.centerlines.centerline_graph(margin=...) takes.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def union_margin(store, classes):
    """(margin float32, rankfield Geometry, StructureRef of the first class) of the union of
    `classes` (store names) in `store` (thalweg.store.FieldStore). All classes must be in one part."""
    import rankfield as rf

    refs = [store.ref(c) for c in classes]
    parts = {r.part for r in refs}
    if len(parts) != 1:
        raise ValueError(f"classes {classes} are in different store parts {sorted(parts)}: "
                         f"their union has no common grid")
    if len(refs) == 1:
        m, geometry, ref = store.margin(classes[0])
        return m, geometry, ref
    code = store.field(refs[0].part)
    inside = {code.labels.index(r.label_value) for r in refs}
    d_in = d_out = None
    for ch in range(code.classes):
        d = rf.deficit(code, ch).astype(np.float32)
        if ch in inside:
            d_in = d if d_in is None else np.maximum(d_in, d)
        else:
            d_out = d if d_out is None else np.maximum(d_out, d)
    m = d_in - d_out if d_out is not None else d_in
    # clamped to +-clip as rankfield's margin (the rendering field; deficits keep the true gap up to
    # the byte curve's far end): for one class this is rf.margin exactly
    clip = float(code.clip)
    m = np.clip(m, -clip, clip)
    logger.info(f"union margin of {classes} on part {refs[0].part}, grid {m.shape}")
    return m.astype(np.float32), code.geometry, refs[0]


def write_margin_image(path, m, geometry):
    """The margin as NIfTI (debug), with the store grid's placement (origin + direction rows)."""
    import SimpleITK as sitk
    dirs = np.asarray(geometry.directions, float)          # one row per array axis (z,y,x order)
    spacing = np.linalg.norm(dirs, axis=1)
    img = sitk.GetImageFromArray(np.asarray(m, np.float32))
    # SimpleITK axes are (x,y,z) = array axes reversed; direction columns = unit axis vectors
    img.SetSpacing(tuple(float(s) for s in spacing[::-1]))
    img.SetOrigin(tuple(float(v) for v in np.asarray(geometry.origin, float)))
    img.SetDirection(tuple(float(v) for v in (dirs[::-1] / spacing[::-1, None]).T.ravel()))
    sitk.WriteImage(img, str(path))
