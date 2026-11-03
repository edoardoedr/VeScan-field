"""Binary mask -> closed surface, as Slicer's "Closed surface" representation
(vtkBinaryLabelmapToClosedSurfaceConversionRule). From vescan/stages/convert_segmentations.py:

  1. crop to the mask's bounding box + 1 voxel; if it touches the volume border, pad 1 voxel
     on every side (the surface is closed at the border too);
  2. iso-contour with vtkDiscreteFlyingEdges3D (default) or vtkSurfaceNets3D, in IJK;
  3. vtkDecimatePro decimation (optional, decimation_factor > 0) and
     vtkWindowedSincPolyDataFilter smoothing, with Slicer's formulas;
  4. IJK -> world with the mask's geometry, directly in LPS (VeScan goes through RAS and
     then flips x and y: a 180-degree rotation about z, the same result);
  5. consistent normals (flying edges only, as Slicer).
"""
from __future__ import annotations

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

METHODS = ("flying-edges", "surface-nets")


def _crop(mask, margin=1):
    """Crop to the mask + margin; pad 1 voxel if it touches the border. Returns (crop, ijk offset)."""
    kk, jj, ii = np.nonzero(mask)
    lo = np.array([kk.min(), jj.min(), ii.min()]) - margin
    hi = np.array([kk.max(), jj.max(), ii.max()]) + margin
    shape = np.array(mask.shape)
    touches = (lo < 0).any() or (hi >= shape).any()
    lo_c, hi_c = np.maximum(lo, 0), np.minimum(hi, shape - 1)
    crop = mask[lo_c[0]:hi_c[0] + 1, lo_c[1]:hi_c[1] + 1, lo_c[2]:hi_c[2] + 1].astype(np.int32)
    if touches:
        crop = np.pad(crop, 1, mode="constant", constant_values=0)
        lo_c = lo_c - 1
    return crop, (int(lo_c[2]), int(lo_c[1]), int(lo_c[0]))      # (i, j, k)


def _vtk_image(array):
    nz, ny, nx = array.shape
    scalars = numpy_to_vtk(array.ravel(order="C"), deep=True, array_type=vtk.VTK_INT)
    scalars.SetName("ImageScalars")
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.SetOrigin(0.0, 0.0, 0.0)
    image.GetPointData().SetScalars(scalars)
    return image


def ijk_to_lps(geom):
    """4x4 matrix IJK (x,y,z) -> LPS from the Geometry (spacing in z,y,x order)."""
    m = np.eye(4)
    m[:3, :3] = geom.direction @ np.diag(geom.spacing[::-1])
    m[:3, 3] = geom.origin
    return m


def convert_mask(mask, geom, method="flying-edges", smoothing_factor=0.5, decimation_factor=0.0,
                 compute_normals=True, surface_nets_internal_smoothing=False):
    """Closed surface (LPS) of a mask (z,y,x) on `geom`'s grid; None if empty."""
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}' (allowed {METHODS})")
    if not np.any(mask):
        return None
    crop, (i0, j0, k0) = _crop(mask != 0)
    image = _vtk_image(crop)

    if method == "flying-edges":
        contour = vtk.vtkDiscreteFlyingEdges3D()
        contour.ComputeGradientsOff()
        contour.ComputeNormalsOff()
    else:
        contour = vtk.vtkSurfaceNets3D()
        contour.SmoothingOff()
        if surface_nets_internal_smoothing:
            contour.SmoothingOn()
            n = 15.0 * smoothing_factor ** 2 + 9.0 * smoothing_factor
            contour.SetNumberOfIterations(int(np.floor(n)))
    contour.SetInputData(image)
    contour.SetValue(0, 1)
    contour.Update()
    result = contour.GetOutput()
    if result.GetNumberOfPolys() == 0:
        return None

    if decimation_factor > 0.0:
        dec = vtk.vtkDecimatePro()
        dec.SetInputData(result)
        dec.SetFeatureAngle(60)
        dec.SplittingOff()
        dec.PreserveTopologyOn()
        dec.SetMaximumError(1)
        dec.SetTargetReduction(decimation_factor)
        dec.Update()
        result = dec.GetOutput()

    if smoothing_factor > 0 and not (method == "surface-nets" and surface_nets_internal_smoothing):
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(result)
        smoother.SetNumberOfIterations(int(20 + smoothing_factor * 40))
        smoother.SetPassBand(10.0 ** (-4.0 * smoothing_factor))
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        result = smoother.GetOutput()

    offset = np.eye(4)
    offset[:3, 3] = [i0, j0, k0]
    matrix = vtk.vtkMatrix4x4()
    matrix.DeepCopy((ijk_to_lps(geom) @ offset).ravel().tolist())
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(result)
    tf.SetTransform(transform)

    if compute_normals and method == "flying-edges":
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(tf.GetOutputPort())
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.Update()
        result = normals.GetOutput()
    else:
        tf.Update()
        result = tf.GetOutput()
    result.GetPointData().RemoveArray("ImageScalars")
    return result
