"""nearest_segments: the same as vtkvmtkPolyBallLine::EvaluateFunction's loop."""
import numpy as np
import pytest

from vescan_thalweg.centerline_distance import TOL, nearest_segments


def brute_force(p, xyz, cells, radius):
    """VMTK's C++ loop, line by line."""
    best = (np.inf, -1, -1, 0.0)
    for c, ids in enumerate(cells):
        for i in range(len(ids) - 1):
            p0, p1 = xyz[ids[i]], xyz[ids[i + 1]]
            r0, r1 = (radius[ids[i]], radius[ids[i + 1]]) if radius is not None else (0.0, 0.0)
            v0, v0r, v1, v1r = p1 - p0, r1 - r0, p - p0, -r0
            num = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2] - v0r * v1r
            den = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2] - v0r * v0r
            if abs(den) < TOL:
                continue
            t = num / den
            if t < TOL:
                t, cp, cr = 0.0, p0, r0
            elif 1.0 - t < TOL:
                t, cp, cr = 1.0, p1, r1
            else:
                cp, cr = p0 + t * v0, r0 + t * v0r
            d = p - cp
            f = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - cr * cr
            if f < best[0]:
                best = (f, c, i, t)
    return best


def random_tree(seed):
    rng = np.random.default_rng(seed)
    cells, pts = [], []
    for c in range(6):
        start = rng.uniform(-20, 20, 3)
        steps = np.cumsum(rng.normal(0, 1.5, (rng.integers(3, 12), 3)), axis=0)
        line = start + np.vstack([[0, 0, 0], steps])
        if c == 2:
            line = np.vstack([line, line[-1:]])          # repeated point: degenerate segment
        cells.append(np.arange(len(pts), len(pts) + len(line)))
        pts.extend(line)
    xyz = np.array(pts)
    return xyz, cells, rng.uniform(0.3, 6.0, len(xyz)), rng.uniform(-30, 30, (300, 3))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("with_radius", [True, False])
def test_same_as_vmtk_loop(seed, with_radius):
    xyz, cells, radius, query = random_tree(seed)
    r = radius if with_radius else None
    cell, sub, t, f = nearest_segments(query, xyz, cells, r, chunk=97)
    for k, p in enumerate(query):
        bf, bc, bs, bt = brute_force(p, xyz, cells, r)
        assert (cell[k], sub[k]) == (bc, bs)
        assert f[k] == pytest.approx(bf, abs=1e-9) and t[k] == pytest.approx(bt, abs=1e-12)


def test_without_radius_is_distance_to_segment():
    xyz = np.array([[0.0, 0, 0], [10, 0, 0]])
    cell, sub, t, f = nearest_segments(np.array([[5.0, 3, 0], [-4, 0, 0]]), xyz, [np.array([0, 1])])
    assert np.allclose(f, [9.0, 16.0]) and np.allclose(t, [0.5, 0.0])


def test_wide_trunk_beats_closer_thin_branch():
    """Why the poly-ball: a point on the wall of a wide trunk, closer in plain distance to the
    centerline of a thin branch next to it, goes to the trunk."""
    xyz = np.array([[0.0, 0, -10], [0, 0, 10], [7, 0, -10], [7, 0, 10]])
    cells = [np.array([0, 1]), np.array([2, 3])]
    radius = np.array([5.0, 5.0, 0.5, 0.5])
    p = np.array([[4.0, 0, 0]])                          # 4 mm from the trunk, 3 from the branch
    assert nearest_segments(p, xyz, cells, radius)[0][0] == 0
    assert nearest_segments(p, xyz, cells, None)[0][0] == 1


def test_same_as_vtkvmtk_if_available():
    vg = pytest.importorskip("vmtk.vtkvmtkComputationalGeometryPython")
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    xyz, cells, radius, query = random_tree(3)
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(xyz, deep=True))
    poly.SetPoints(pts)
    lines = vtk.vtkCellArray()
    for ids in cells:
        lines.InsertNextCell(len(ids))
        for i in ids:
            lines.InsertCellPoint(int(i))
    poly.SetLines(lines)
    arr = numpy_to_vtk(radius, deep=True)
    arr.SetName("Radius")
    poly.GetPointData().AddArray(arr)
    ball = vg.vtkvmtkPolyBallLine()
    ball.SetInput(poly)
    ball.SetPolyBallRadiusArrayName("Radius")
    cell, sub, _, f = nearest_segments(query, xyz, cells, radius)
    for k, p in enumerate(query):
        value = ball.EvaluateFunction(p)
        assert (cell[k], sub[k]) == (ball.GetLastPolyBallCellId(), ball.GetLastPolyBallCellSubId())
        assert f[k] == pytest.approx(value, abs=1e-9)
