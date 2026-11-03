"""Synthetic phantom (from the earlier prototype): Y-shaped "arterial" tree (radii 3.0 / 1.5 /
1.0 mm) + a "vein" (r 1.5 mm) touching a branch. Logits = k * signed distance + noise; the
margin is computed directly from the logits, as from the store."""
import numpy as np
import pytest

from vescan_field.stages.centerlines.extract import extract
from vescan_field.field import Geometry

SP = np.array([0.7, 0.7, 0.7])
SHAPE = (100, 100, 100)
K = 8.0  # logits per mm (+-2 logits ~ +-0.25 mm)
J = np.array([35.0, 35, 35])
ART = [("trunk", (8, 35, 35), J, 3.0),
       ("branch_1.5", J, (62, 51, 35), 1.5),
       ("branch_1.0", J, (62, 19, 35), 1.0)]


def capsule_sdf(P, a, b, r):
    a, b = np.array(a, float), np.array(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return r - np.linalg.norm(P - (a + t[..., None] * ab), axis=-1)


def phantom(seed=0):
    P = np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(SHAPE, SP)], indexing="ij"), -1)
    sdf_a = np.max([capsule_sdf(P, a, b, r) for _, a, b, r in ART], 0)
    d = np.cross([27.0, 16.0, 0] / np.linalg.norm([27.0, 16.0, 0]), [0, 0, 1.0])
    c = J + 0.5 * np.array([27.0, 16.0, 0]) + 3.0 * d                      # axis distance = 1.5+1.5
    sdf_v = capsule_sdf(P, c - [0, 0, 25], c + [0, 0, 25], 1.5)
    rng = np.random.default_rng(seed)
    logits = np.stack([np.zeros(SHAPE), K * sdf_a, K * sdf_v]).astype(np.float32)
    logits += rng.normal(0, 0.3, logits.shape).astype(np.float32)
    margin_art = logits[1] - np.maximum(logits[0], logits[2])
    return margin_art, sdf_v


@pytest.fixture(scope="module")
def result():
    m, sdf_v = phantom()
    return extract(m, Geometry(SP)), sdf_v


def _branch_of(p):
    mid = p[len(p) // 2]
    return max(ART, key=lambda x: capsule_sdf(mid[None], x[1], x[2], 0)[0])


def test_tree_shape(result):
    res, _ = result
    assert res["counts"]["components"] == 1
    long_segs = [s for s in res["segments"] if s["length_mm"] > 5]
    names = [_branch_of(res["geom"].crop_mm_to_full_mm(s["points_mm"]))[0] for s in long_segs]
    # the trunk is split in two if the root (max D) falls mid-way (degree 2)
    assert sorted(set(names)) == ["branch_1.0", "branch_1.5", "trunk"]
    assert names.count("branch_1.5") == names.count("branch_1.0") == 1
    assert names.count("trunk") in (1, 2)


def test_radii(result):
    res, _ = result
    for s in res["segments"]:
        if s["length_mm"] <= 5:
            continue
        p = res["geom"].crop_mm_to_full_mm(s["points_mm"])
        keep = slice(len(p) // 5, len(p) - len(p) // 5)  # away from junctions and ends
        name, _, _, r_true = _branch_of(p)
        r = np.median(s["radius"][keep])
        assert abs(r - r_true) < 0.15, (name, r, r_true)
        assert np.median(s["radius_lo"][keep]) <= r <= np.median(s["radius_hi"][keep])


def test_centerline_on_axis(result):
    res, _ = result
    for s in res["segments"]:
        if s["length_mm"] <= 5:
            continue
        p = res["geom"].crop_mm_to_full_mm(s["points_mm"])
        keep = slice(len(p) // 5, len(p) - len(p) // 5)
        _, a, b, r = _branch_of(p)
        off_axis = r - capsule_sdf(p[keep], a, b, r)   # distance from the capsule axis
        assert np.median(off_axis) < 0.2


def test_vein_not_in_artery_tree(result):
    res, sdf_v = result
    pts = np.concatenate([res["geom"].crop_mm_to_full_mm(s["points_mm"]) for s in res["segments"]])
    from vescan_field.field import interp
    assert np.all(interp(sdf_v, pts, SP) < 0)
