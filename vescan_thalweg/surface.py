"""VTK surface utilities shared by every stage: I/O with the LPS/RAS space in the header,
components, triangles, areas, cell extraction, normal orientation, numpy views of arrays.

From vescan/io.py, vescan/stages/preprocess.py, build_lobe_segments.py and lobe_reachability.py,
with the same behaviour; polydata_to_arrays/arrays_to_polydata and signed_volume are
vectorised (no Python loop per triangle).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- I/O
def load_surface(path):
    """Read .vtk (legacy), .vtp (XML) or .stl."""
    ext = os.path.splitext(str(path))[1].lower()
    readers = {".stl": vtk.vtkSTLReader, ".vtp": vtk.vtkXMLPolyDataReader, ".vtk": vtk.vtkPolyDataReader}
    if ext not in readers:
        raise ValueError(f"unsupported extension '{ext}' (.stl, .vtk, .vtp)")
    reader = readers[ext]()
    reader.SetFileName(str(path))
    if ext == ".vtk":
        reader.ReadAllScalarsOn()
        reader.ReadAllFieldsOn()
    reader.Update()
    poly = reader.GetOutput()
    if poly.GetNumberOfPoints() == 0:
        raise ValueError(f"empty or unreadable surface: {path}")
    return poly


def save_surface(poly, path, coordinate_space="LPS"):
    """Write .vtk (with the space in the header as Slicer does: "3D Slicer output. SPACE=LPS",
    so detect_coordinate_space finds it again) or .vtp."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".vtk":
        writer = vtk.vtkPolyDataWriter()
        writer.SetHeader(f"3D Slicer output. SPACE={coordinate_space}")
    elif ext == ".vtp":
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        raise ValueError(f"unsupported extension for writing '{ext}' (.vtk, .vtp)")
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.Write()


def detect_coordinate_space(path):
    """"LPS", "RAS" or None, from the header of a .vtk written by Slicer or save_surface."""
    if os.path.splitext(str(path))[1].lower() != ".vtk":
        return None
    with open(path, "rb") as f:
        header = b"".join(f.readline() for _ in range(2)).decode("ascii", errors="ignore")
    for space in ("LPS", "RAS"):
        if f"SPACE={space}" in header:
            return space
    return None


def flip_lps_ras(poly):
    """LPS <-> RAS: negate x and y (a 180-degree rotation about z, not a reflection)."""
    transform = vtk.vtkTransform()
    transform.Scale(-1.0, -1.0, 1.0)
    f = vtk.vtkTransformPolyDataFilter()
    f.SetTransform(transform)
    f.SetInputData(poly)
    f.Update()
    return f.GetOutput()


# --------------------------------------------------------------------------- array <-> polydata
def polydata_to_arrays(poly):
    """(vertices (n,3) float64, triangles (m,3) int32), triangulating first."""
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    out = tri.GetOutput()
    verts = vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)
    faces = vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int32)
    return verts, faces


def arrays_to_polydata(verts, faces):
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(verts, dtype=np.float64), deep=True))
    poly.SetPoints(points)
    cells = vtk.vtkCellArray()
    offsets = np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64)
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(np.ascontiguousarray(faces, dtype=np.int64).ravel(), deep=True))
    poly.SetPolys(cells)
    return poly


def _make_array(name, values, vtk_type):
    arr = numpy_to_vtk(np.ascontiguousarray(values), deep=True, array_type=vtk_type)
    arr.SetName(name)
    return arr


def add_cell_array(poly, name, values, vtk_type=vtk.VTK_INT):
    poly.GetCellData().AddArray(_make_array(name, values, vtk_type))


def add_point_array(poly, name, values, vtk_type=vtk.VTK_INT):
    poly.GetPointData().AddArray(_make_array(name, values, vtk_type))


def cell_array(poly, name):
    """A cell array as numpy (None if missing)."""
    arr = poly.GetCellData().GetArray(name)
    return None if arr is None else vtk_to_numpy(arr)


def point_array(poly, name):
    """A point array as numpy (None if missing)."""
    arr = poly.GetPointData().GetArray(name)
    return None if arr is None else vtk_to_numpy(arr)


def points(poly):
    return vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)


def cell_point_ids(poly):
    """For each cell, its point ids in cell order (list of arrays).
    For centerline polylines (cells = lines), as GetCell(i).GetPointIds()."""
    cells = poly.GetLines() if poly.GetNumberOfLines() else poly.GetPolys()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    conn = vtk_to_numpy(cells.GetConnectivityArray())
    return [conn[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def extract_lines(poly, cell_ids):
    """The chosen cells of a centerline as a new polyline polydata, with every point and cell
    array (same VTK type); points renumbered in order."""
    cells = cell_point_ids(poly)
    chosen = [cells[c] for c in cell_ids]
    ids = np.concatenate(chosen) if chosen else np.zeros(0, int)
    out = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(points(poly)[ids]), deep=True))
    out.SetPoints(pts)
    lines = vtk.vtkCellArray()
    offsets = np.r_[0, np.cumsum([len(c) for c in chosen])].astype(np.int64)
    lines.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(len(ids), dtype=np.int64), deep=True))
    out.SetLines(lines)
    for src, dst, sel in ((poly.GetPointData(), out.GetPointData(), ids),
                          (poly.GetCellData(), out.GetCellData(), np.asarray(cell_ids, int))):
        for i in range(src.GetNumberOfArrays()):
            arr = src.GetArray(i)
            if arr is None:
                continue
            values = vtk_to_numpy(arr)[sel]
            dst.AddArray(_make_array(arr.GetName(), values, arr.GetDataType()))
    return out


def add_cell_id_array(poly, name="CellId"):
    """CellId 0..N-1: one number per cell (to colour every piece differently in Slicer)."""
    add_cell_array(poly, name, np.arange(poly.GetNumberOfCells(), dtype=np.int32))


def vertex_graph(poly):
    """(neighbours as CSR, area of each point, total area) of a triangle surface: two points are
    neighbours if they share a triangle; a point's area is 1/3 of each of its triangles."""
    from scipy import sparse
    tri = triangles(poly)
    n = poly.GetNumberOfPoints()
    rows = np.concatenate([tri[:, i] for i, j in ((0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2))])
    cols = np.concatenate([tri[:, j] for i, j in ((0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2))])
    adj = sparse.csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(n, n))
    adj.sum_duplicates()
    area = cell_areas(poly)
    point_area = np.bincount(tri.ravel(), weights=np.repeat(area / 3.0, 3), minlength=n)
    return adj, point_area, float(area.sum())


def boundary_loops(poly):
    """The open boundaries as loops of point ids, each in the direction its edges have in the
    triangles (so a fan cap [p_j, center, p_j+1] gets the right orientation)."""
    tri = triangles(poly)
    directed = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    key = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse.ravel()] == 1]
    following = dict(zip(boundary[:, 0].tolist(), boundary[:, 1].tolist()))
    loops, seen = [], set()
    for start in boundary[:, 0].tolist():
        if start in seen:
            continue
        loop, p = [], start
        while p not in seen and p in following:
            seen.add(p)
            loop.append(p)
            p = following[p]
        if len(loop) >= 3:
            loops.append(np.array(loop))
    return loops


def cap_holes(poly, keep_arrays=False):
    """Close every open boundary as vtkvmtkCapPolyData (zero displacement): one point at the
    boundary's edge-length-weighted barycenter, and a triangle fan. Geometry only, as VMTK
    (which drops the arrays), unless keep_arrays: then each cap triangle takes the cell values of
    the triangle across its boundary edge, and each cap center the point values of its nearest
    boundary point."""
    pts, tri = points(poly), triangles(poly)
    new_pts, new_tri = [pts], [tri]
    point_src, cell_src = [np.arange(len(pts))], [np.arange(len(tri))]
    next_id = len(pts)
    edge_cell = {}
    if keep_arrays:
        for c, (i, j, k) in enumerate(tri.tolist()):
            edge_cell[i, j] = edge_cell[j, k] = edge_cell[k, i] = c
    for loop in boundary_loops(poly):
        a, b = pts[loop], pts[np.roll(loop, -1)]
        w = np.linalg.norm(b - a, axis=1)
        keep = w >= 1e-12
        center = ((a[keep] + b[keep]) / 2.0 * w[keep, None]).sum(axis=0) / w[keep].sum()
        new_pts.append(center[None])
        new_tri.append(np.stack([loop, np.full(len(loop), next_id), np.roll(loop, -1)], axis=1))
        next_id += 1
        if keep_arrays:
            point_src.append([loop[np.argmin(np.linalg.norm(a - center, axis=1))]])
            cells = []
            for edge in zip(loop.tolist(), np.roll(loop, -1).tolist()):
                # a loop closed at a pinched vertex ends on an edge no triangle has: the previous one's
                cells.append(edge_cell.get(edge, cells[-1] if cells else None))
            cell_src.append(cells)
    out = arrays_to_polydata(np.vstack(new_pts), np.vstack(new_tri))
    if keep_arrays:
        for src, dst, sel in ((poly.GetPointData(), out.GetPointData(), np.concatenate(point_src)),
                              (poly.GetCellData(), out.GetCellData(), np.concatenate(cell_src))):
            for i in range(src.GetNumberOfArrays()):
                arr = src.GetArray(i)
                if arr is not None:
                    dst.AddArray(_make_array(arr.GetName(), vtk_to_numpy(arr)[sel], arr.GetDataType()))
    return out


def representative_cells(group_of_cell, cell_ids):
    """Group -> its cell with the most points (on a tie, the first): a branch traced by several
    centerlines is measured once (as VeScan's build_graph/crossings)."""
    best = {}
    for cell, group in enumerate(group_of_cell):
        group = int(group)
        n = len(cell_ids[cell])
        if group not in best or n > best[group][1]:
            best[group] = (cell, n)
    return {group: cell for group, (cell, _) in best.items()}


def points_inside(surface, pts, tolerance):
    """Which points (n,3) lie inside a closed surface (vtkSelectEnclosedPoints).
    CheckSurface stays off: on a surface it judges open it fails instead of warning; closure is
    checked beforehand with close_if_open."""
    if len(pts) == 0:
        return np.zeros(0, bool)
    query = vtk.vtkPolyData()
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_to_vtk(np.ascontiguousarray(pts, dtype=np.float64), deep=True))
    query.SetPoints(vpts)
    checker = vtk.vtkSelectEnclosedPoints()
    checker.SetSurfaceData(surface)
    checker.SetTolerance(tolerance)
    checker.SetInputData(query)
    checker.Update()
    return vtk_to_numpy(checker.GetOutput().GetPointData().GetArray("SelectedPoints")).astype(bool)


# --------------------------------------------------------------------------- triangles
def triangles(poly):
    """Connectivity (m,3) of a triangle-only surface."""
    conn = vtk_to_numpy(poly.GetPolys().GetConnectivityArray())
    n = poly.GetNumberOfCells()
    if conn.size != 3 * n:
        raise ValueError(f"expected a triangle-only surface: {conn.size} indices for {n} cells")
    return conn.reshape(n, 3)


def cell_areas(poly):
    pts = vtk_to_numpy(poly.GetPoints().GetData())
    tri = triangles(poly)
    p0, p1, p2 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def extract_cells(poly, cell_mask):
    """The selected cells as their own polydata (geometry only, points renumbered)."""
    tri = triangles(poly)[cell_mask]
    used = np.unique(tri)
    remap = np.full(poly.GetNumberOfPoints(), -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    pts = vtk_to_numpy(poly.GetPoints().GetData())[used]
    return arrays_to_polydata(pts, remap[tri])


def signed_volume(poly):
    """Enclosed volume, signed by orientation: positive with outward normals."""
    pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    tri = triangles(poly)
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def ensure_outward_normals(poly):
    """Flip the surface if it is inside out (negative signed volume): vtkPolyDataNormals'
    AutoOrientNormals is unreliable with non-manifold edges (see VeScan's preprocess)."""
    volume = signed_volume(poly)
    if volume >= 0.0:
        return poly
    logger.info(f"surface inside out (signed volume {volume:.1f} mm3): flipping it")
    reverse = vtk.vtkReverseSense()
    reverse.SetInputData(poly)
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.Update()
    return reverse.GetOutput()


# --------------------------------------------------------------------------- components and boundaries
def keep_largest_connected_component(poly, clean=True):
    """Only the largest connected component, then (if clean) vtkCleanPolyData."""
    labeler = vtk.vtkPolyDataConnectivityFilter()
    labeler.SetInputData(poly)
    labeler.SetExtractionModeToAllRegions()
    labeler.ColorRegionsOn()
    labeler.Update()
    n_regions = labeler.GetNumberOfExtractedRegions()
    if n_regions > 1:
        rid = vtk_to_numpy(labeler.GetOutput().GetPointData().GetArray("RegionId"))
        counts = np.sort(np.bincount(rid))[::-1]
        logger.info(f"{n_regions} components (points, largest first): {counts[:5].tolist()}"
                    f"{' ...' if n_regions > 5 else ''}")
    largest = vtk.vtkPolyDataConnectivityFilter()
    largest.SetInputData(poly)
    largest.SetExtractionModeToLargestRegion()
    largest.Update()
    if not clean:
        return largest.GetOutput()
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(largest.GetOutputPort())
    cleaner.Update()
    return cleaner.GetOutput()


def count_open_edge_points(poly):
    """Points on boundary or non-manifold edges (0 = closed surface)."""
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(poly)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    return edges.GetOutput().GetNumberOfPoints()


def close_if_open(poly, name=""):
    """vtkCleanPolyData only if the surface is not closed (duplicate points not merged): cleaning
    an already closed mesh can create a non-manifold edge (VeScan's lobe_reachability)."""
    if count_open_edge_points(poly) > 0:
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(poly)
        cleaner.Update()
        poly = cleaner.GetOutput()
        n_open = count_open_edge_points(poly)
        if n_open:
            logger.warning(f"{name} not closed ({n_open} points on open/non-manifold edges)")
    return poly
