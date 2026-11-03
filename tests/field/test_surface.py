"""Shared surface utilities (vescan_field.surface)."""
import numpy as np
import vtk

from vescan_field import surface


def cube(size=2.0):
    src = vtk.vtkCubeSource()
    src.SetXLength(size)
    src.SetYLength(size)
    src.SetZLength(size)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(src.GetOutputPort())
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.Update()
    return clean.GetOutput()


def test_signed_volume_and_outward_normals():
    c = cube(2.0)
    assert np.isclose(surface.signed_volume(c), 8.0)
    rev = vtk.vtkReverseSense()
    rev.SetInputData(c)
    rev.ReverseCellsOn()
    rev.Update()
    inside_out = rev.GetOutput()
    assert np.isclose(surface.signed_volume(inside_out), -8.0)
    fixed = surface.ensure_outward_normals(inside_out)
    assert np.isclose(surface.signed_volume(fixed), 8.0)


def test_arrays_roundtrip_and_areas():
    c = cube(2.0)
    v, f = surface.polydata_to_arrays(c)
    back = surface.arrays_to_polydata(v, f)
    assert back.GetNumberOfPoints() == c.GetNumberOfPoints()
    assert (surface.triangles(back) == f).all()
    assert np.isclose(surface.cell_areas(back).sum(), 24.0)       # 6 faces of 4 mm2


def test_extract_cells_and_closedness():
    c = cube(2.0)
    assert surface.count_open_edge_points(c) == 0
    mask = np.zeros(c.GetNumberOfCells(), bool)
    mask[:2] = True                                                 # one face (2 triangles)
    patch = surface.extract_cells(c, mask)
    assert patch.GetNumberOfCells() == 2 and patch.GetNumberOfPoints() == 4
    assert np.isclose(surface.cell_areas(patch).sum(), 4.0)
    assert surface.count_open_edge_points(patch) == 4


def test_save_load_space_header(tmp_path):
    path = tmp_path / "c.vtk"
    surface.save_surface(cube(), path, coordinate_space="LPS")
    assert surface.detect_coordinate_space(path) == "LPS"
    assert surface.load_surface(path).GetNumberOfCells() == 12


def test_keep_largest_component():
    a, b = cube(2.0), cube(1.0)
    shift = vtk.vtkTransform()
    shift.Translate(10, 0, 0)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(b)
    tf.SetTransform(shift)
    app = vtk.vtkAppendPolyData()
    app.AddInputData(a)
    app.AddInputConnection(tf.GetOutputPort())
    app.Update()
    out = surface.keep_largest_connected_component(app.GetOutput())
    assert out.GetNumberOfPoints() == a.GetNumberOfPoints()


def test_points_inside_and_cells():
    c = cube(2.0)                                                    # [-1, 1]^3
    pts = np.array([[0, 0, 0], [0.9, 0, 0], [1.5, 0, 0], [0, 0, -3]], float)
    assert surface.points_inside(c, pts, 1e-4).tolist() == [True, True, False, False]
    line = vtk.vtkPolyData()
    p = vtk.vtkPoints()
    for x in range(5):
        p.InsertNextPoint(x, 0, 0)
    line.SetPoints(p)
    lines = vtk.vtkCellArray()
    for ids in ([0, 1, 2], [3, 4]):
        lines.InsertNextCell(len(ids))
        for i in ids:
            lines.InsertCellPoint(i)
    line.SetLines(lines)
    cells = surface.cell_point_ids(line)
    assert [c.tolist() for c in cells] == [[0, 1, 2], [3, 4]]
    # group 7 on two cells: the representative is the one with more points
    assert surface.representative_cells([7, 7], cells) == {7: 0}
