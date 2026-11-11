"""Stage surfaces: mask -> surface conversion, decimation to N points, fissures."""
import numpy as np
import pytest

from vescan_thalweg import surface
from vescan_thalweg.field import Geometry
from vescan_thalweg.stages.surfaces import lobe_segments
from vescan_thalweg.stages.surfaces.convert import convert_mask
from vescan_thalweg.stages.surfaces.preprocess import decimate_to_points, preprocess

SP = np.array([0.5, 0.5, 0.5])


def sphere_mask(r_mm=10.0, n=60):
    c = (n - 1) / 2
    z, y, x = np.ogrid[:n, :n, :n]
    return ((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2) * SP[0] ** 2 <= r_mm ** 2


@pytest.fixture(scope="module")
def sphere():
    return convert_mask(sphere_mask(), Geometry(SP))


def test_convert_sphere(sphere):
    assert surface.count_open_edge_points(sphere) == 0
    v = surface.signed_volume(sphere)
    assert v > 0                                                   # outward normals
    assert abs(v - 4 / 3 * np.pi * 10.0 ** 3) / (4 / 3 * np.pi * 10.0 ** 3) < 0.03


def test_convert_touching_border_is_closed():
    mask = np.zeros((20, 20, 20), bool)
    mask[:10, 5:15, 5:15] = True                                   # touches the first slice
    poly = convert_mask(mask, Geometry(SP))
    assert surface.count_open_edge_points(poly) == 0


def test_convert_world_coordinates():
    """With origin and spacing: the sphere center lands where the geometry says."""
    g = Geometry(np.array([1.0, 0.8, 0.6]), origin=np.array([-10.0, 20.0, 30.0]))
    mask = np.zeros((30, 30, 30), bool)
    z, y, x = np.ogrid[:30, :30, :30]
    mask[((z - 15) * 1.0) ** 2 + ((y - 15) * 0.8) ** 2 + ((x - 15) * 0.6) ** 2 <= 36] = True
    from vtk.util.numpy_support import vtk_to_numpy
    c = vtk_to_numpy(convert_mask(mask, g).GetPoints().GetData()).mean(0)
    assert np.allclose(c, [-10 + 15 * 0.6, 20 + 15 * 0.8, 30 + 15 * 1.0], atol=0.1)


def test_decimate_to_points(sphere):
    n_in = sphere.GetNumberOfPoints()
    target = n_in // 5
    out, info = decimate_to_points(sphere, target, tolerance=0.05)
    assert out.GetNumberOfPoints() <= target * 1.05
    assert out.GetNumberOfPoints() >= target * 0.9
    assert info["aggressiveness"] is not None and info["passes"] >= 2
    same, info2 = decimate_to_points(sphere, n_in + 10)  # already below the target
    assert same is sphere and info2["passes"] == 0


def test_preprocess_keeps_closed_outward(sphere):
    out, info = preprocess(sphere, target_points=sphere.GetNumberOfPoints() // 4)
    assert surface.signed_volume(out) > 0
    assert surface.count_open_edge_points(out) == 0
    assert info["points"] <= sphere.GetNumberOfPoints() // 4 * 1.05


def test_fissure_between_adjacent_blocks():
    """Two adjacent blocks of the same label map: contact ~ area of the shared face."""
    g = Geometry(np.array([1.0, 1.0, 1.0]))
    labels = np.zeros((40, 40, 40), np.int8)
    labels[5:20, 5:35, 5:35] = 1                                   # "RUL"
    labels[20:35, 5:35, 5:35] = 2                                  # "RML", below
    lobes = {"RUL": convert_mask(labels == 1, g), "RML": convert_mask(labels == 2, g)}
    combined, fissures = lobe_segments.build_combined_lobes(lobes)
    assert len(fissures) == 1 and fissures[0]["contact_value"] == 60
    # shared face 30 x 30 mm2; smoothing rounds the rim, so a bit less
    assert 600 < fissures[0]["area_mm2"] < 900
    from vtk.util.numpy_support import vtk_to_numpy
    cd = combined.GetCellData()
    assert set(np.unique(vtk_to_numpy(cd.GetArray("LobeSegment_Artery")))) == {40, 41}
    assert set(np.unique(vtk_to_numpy(cd.GetArray("Fissure")))) == {0, 70}
    leg = lobe_segments.legend(fissures)
    assert "Fissure" in leg["arrays"] and leg["fissures"][0]["lobe_a"] == "RUL"
