"""Centerline labels onto a surface (from vescan/stages/transfer_centerline_labels.py).

Each surface point takes its values from the nearest centerline cell under VMTK's
radius-weighted metric (vescan_field.centerline_distance, no vtkvmtk). An array stored in the
centerline's POINT data (LobeLabelPoint, AnatomicalSegmentPoint) is taken from the nearest
centerline point on the same segment (t >= 0.5 -> the next one): categorical values, no
interpolation. Cell and point follow the same choice, so they never disagree.

drop_unlabeled first removes the cells without a label (0 everywhere): an unclassified cell with
a huge radius steals surface (on a real airway tree 204 centerline points took 35% of the
surface). Only for labels where 0 means "unclassified", not for GroupId.

fill_islands: patches of a label smaller than 0.25% of the area are filled with the prevailing
value around them (wall points nearer to a centerline beyond a fissure). 0.25% falls in an empty
gap (islands up to 206-319 mm2, real regions from 804 mm2 up). Protected values: the crossings,
which on the surface are exactly small patches.
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from scipy.sparse import csgraph

from vescan_field.centerline_distance import nearest_segments
from vescan_field.surface import (_make_array, cell_point_ids, extract_lines, points, vertex_graph)

logger = logging.getLogger(__name__)

MAX_ISLAND_AREA_FRACTION = 0.0025


def source_arrays(centerline, names):
    """name -> (numpy array, per point?, VTK type). Cell data first, then point data."""
    from vtk.util.numpy_support import vtk_to_numpy
    out, missing = {}, []
    for name in names:
        for data, per_point in ((centerline.GetCellData(), False), (centerline.GetPointData(), True)):
            arr = data.GetArray(name)
            if arr is not None:
                out[name] = (vtk_to_numpy(arr), per_point, arr.GetDataType())
                break
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"the centerline has no arrays {missing}")
    return out


def labelled_cells(cells, arrays, unlabeled=0):
    """Cells with at least one value other than `unlabeled` in one of the arrays."""
    keep = []
    for c, ids in enumerate(cells):
        for values, per_point, _ in arrays.values():
            if np.any((values[ids] if per_point else values[c]) != unlabeled):
                keep.append(c)
                break
    return keep


def transfer(surface, centerline, names, drop_unlabeled=False, unlabeled=0, fill=False,
             max_island_area_fraction=MAX_ISLAND_AREA_FRACTION, max_island_passes=5, protected=None):
    """New surface with the centerline's `names` arrays as point data."""
    import vtk
    arrays = source_arrays(centerline, names)
    cells = cell_point_ids(centerline)
    if drop_unlabeled:
        keep = labelled_cells(cells, arrays, unlabeled)
        if keep and len(keep) < len(cells):
            logger.info(f"{len(cells) - len(keep)} of {len(cells)} unlabelled cells ignored")
            centerline = extract_lines(centerline, keep)
            arrays, cells = source_arrays(centerline, names), cell_point_ids(centerline)

    radius = centerline.GetPointData().GetArray("Radius")
    if radius is not None:
        from vtk.util.numpy_support import vtk_to_numpy
        radius = vtk_to_numpy(radius)
    cell, sub, t, _ = nearest_segments(points(surface), points(centerline), cells, radius)
    lengths = np.array([len(ids) for ids in cells])
    k = np.clip(sub + (t >= 0.5), 0, lengths[cell] - 1)
    point = np.array([cells[c][i] for c, i in zip(cell, k)])

    out = vtk.vtkPolyData()
    out.DeepCopy(surface)
    for name, (values, per_point, vtk_type) in arrays.items():
        out.GetPointData().AddArray(_make_array(name, values[point if per_point else cell], vtk_type))
    if fill:
        n = fill_islands(out, list(arrays), max_island_area_fraction, protected or {}, max_island_passes)
        if n:
            logger.info(f"{n} points filled in islands under {100 * max_island_area_fraction:.2f}% of the area")
    return out


def fill_islands(surface, names, max_area_fraction=MAX_ISLAND_AREA_FRACTION, protected=None, max_passes=5):
    """Fill the small islands of each array in place, up to max_passes passes. Regions are
    visited in order of first point and updated in sequence, as VeScan does."""
    from vtk.util.numpy_support import vtk_to_numpy
    protected = protected or {}
    adj, point_area, total = vertex_graph(surface)
    if total <= 0:
        return 0
    max_area = total * max_area_fraction
    rows = np.repeat(np.arange(adj.shape[0]), np.diff(adj.indptr))
    filled_total = 0
    for name in names:
        arr = surface.GetPointData().GetArray(name)
        if arr is None or arr.GetNumberOfComponents() != 1:
            continue
        values = vtk_to_numpy(arr)                       # a view: writes go straight into the array
        guard = protected.get(name, set())
        for _ in range(max_passes):
            same = values[rows] == values[adj.indices]
            graph = adj.copy()
            graph.data = same
            graph.eliminate_zeros()
            n_comp, comp = csgraph.connected_components(graph, directed=False)
            order = np.argsort(comp, kind="stable")      # points of each region, in order
            starts = np.searchsorted(comp[order], np.arange(n_comp))
            ends = np.r_[starts[1:], len(order)]
            first_point = order[starts]
            area = np.bincount(comp, weights=point_area, minlength=n_comp)
            filled = 0
            for r in np.argsort(first_point):            # regions in order of first point
                region = order[starts[r]:ends[r]]
                v = values[region[0]]
                if v in guard or area[r] > max_area:
                    continue
                nb = np.concatenate([adj.indices[adj.indptr[i]:adj.indptr[i + 1]] for i in region])
                ring = Counter(values[nb[values[nb] != v]].tolist())
                if not ring:
                    continue                             # detached piece: nothing around
                values[region] = ring.most_common(1)[0][0]
                filled += len(region)
            filled_total += filled
            if not filled:
                break
        arr.Modified()
    return filled_total
