"""The union margin on the real store (PD-1-Lung-00043; skipped without it)."""
from pathlib import Path

import numpy as np
import pytest

STORE = Path(__file__).resolve().parents[2] / "Data" / "anti_pd_1_lung" / "PD-1-Lung-00043" / "lung_vessels.duckn.zip"
pytestmark = pytest.mark.skipif(not STORE.exists(), reason="PD-1-Lung-00043 store not present")


@pytest.fixture(scope="module")
def store():
    from thalweg.store import open_store
    return open_store(STORE)


def test_one_class_is_rankfields_margin(store):
    import rankfield as rf

    from vescan_thalweg.margin import union_margin
    ref = store.ref("lung_arteries")
    code = store.field(ref.part)
    expected = rf.margin(code, code.labels.index(ref.label_value))
    m, geometry, r = union_margin(store, ["lung_arteries"])
    assert r.label_value == ref.label_value
    np.testing.assert_array_equal(m, expected)


def test_union_contains_its_classes(store):
    from vescan_thalweg.margin import union_margin
    lumen, _, _ = union_margin(store, ["lung_airways"])
    wall, _, _ = union_margin(store, ["lung_airways_wall"])
    union, geometry, ref = union_margin(store, ["lung_airways", "lung_airways_wall"])
    assert ref.label_value == store.ref("lung_airways").label_value
    assert np.all(union >= np.maximum(lumen, wall) - 1e-5)          # m_S >= m_c for every c in S
    inside = union > 0
    assert np.all(inside[(lumen > 0) | (wall > 0)])                    # the union holds both classes
    assert inside.sum() > (lumen > 0).sum()
    assert union.min() >= -8.0 - 1e-6 and union.max() <= 8.0 + 1e-6   # clamped to +-clip


def test_margin_image_roundtrip(store, tmp_path):
    import SimpleITK as sitk

    from thalweg.kernel.field import to_world

    from vescan_thalweg.margin import union_margin, write_margin_image
    m, geometry, _ = union_margin(store, ["lung_arteries"])
    path = tmp_path / "m.nii.gz"
    write_margin_image(path, m, geometry)
    img = sitk.ReadImage(str(path))
    idx = np.array([[10, 20, 30], [100, 150, 200]])                    # array order (z,y,x)
    for i in idx:
        lps = img.TransformContinuousIndexToPhysicalPoint([float(v) for v in i[::-1]])
        np.testing.assert_allclose(lps, to_world(geometry, i), atol=1e-3)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(img), m)
