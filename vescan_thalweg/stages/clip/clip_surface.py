"""Clip of the vessel surface on the cut centerline (from vescan/stages/clip_vessel.py).

Each surface point goes to the side of the centerline (kept or dropped) it is nearer to:
scalar = distance to kept - distance to dropped (distance to the polyline, not to its points:
no steps), and vtkClipPolyData cuts at zero inside the triangles, so the border is smooth and
does not follow the triangulation. Then the largest component (drops the slivers where kept and
dropped are ambiguous, near a bifurcation) and, if requested, the caps on the holes.

The surface's arrays (the labels of stage surface_labels) are kept: a cut triangle keeps its cell
values, a new point on a cut edge takes the point values of the nearer original point (labels
are not interpolated), and a cap takes the values of the border it closes. The normals are
dropped: they would be wrong on the caps.

Instead of Slicer's ClipVessel (vtkvmtkPolyDataCenterlineGroupsClipper): with hundreds of kept
groups it became unusable (20+ minutes). VMTK's flow extensions (CFD only) are not ported.
"""
from __future__ import annotations

import logging

import numpy as np
import vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from vescan_thalweg.centerline_distance import nearest_segments
from vescan_thalweg.surface import _make_array, cap_holes, cell_point_ids, keep_largest_connected_component, points

logger = logging.getLogger(__name__)


def distance_to(pts, centerline):
    """Distance (mm) of each point from the centerline polyline."""
    f = nearest_segments(pts, points(centerline), cell_point_ids(centerline))[3]
    return np.sqrt(np.maximum(f, 0.0))


def clip_surface(surface, kept, dropped, cap=True):
    """Clipped surface (largest component, with caps if cap), with the surface's arrays."""
    if dropped.GetNumberOfPoints() == 0:
        logger.info("nothing to clip: surface unchanged")
        return surface
    pts = points(surface)
    side = distance_to(pts, kept) - distance_to(pts, dropped)     # < 0: on the kept side
    logger.info(f"{int((side <= 0).sum())}/{len(pts)} points on the kept side")

    scalar = vtk.vtkPolyData()
    scalar.DeepCopy(surface)
    arr = numpy_to_vtk(side, deep=True)
    arr.SetName("KeptDroppedSignedDistance")
    scalar.GetPointData().SetScalars(arr)
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(scalar)
    clipper.SetValue(0.0)
    clipper.GenerateClipScalarsOff()
    clipper.SetInsideOut(1)                                         # keeps scalar <= 0
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(clipper.GetOutputPort())
    cleaner.Update()
    out = keep_largest_connected_component(cleaner.GetOutput(), clean=False)
    nearest = cKDTree(pts).query(points(out))[1]
    src, dst = surface.GetPointData(), vtk.vtkPointData()
    for i in range(src.GetNumberOfArrays()):
        arr = src.GetArray(i)
        if arr is not None and arr is not src.GetNormals():
            dst.AddArray(_make_array(arr.GetName(), vtk_to_numpy(arr)[nearest], arr.GetDataType()))
    out.GetPointData().ShallowCopy(dst)
    return cap_holes(out, keep_arrays=True) if cap else out
