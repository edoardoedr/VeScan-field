"""Oriented graph on the Y phantom: root in the trunk, one junction, two child branches."""
import numpy as np
import pytest

from vescan_field.stages.centerlines import topology
from vescan_field.stages.centerlines.extract import extract
from vescan_field.field import Geometry
from .test_phantom import SP, _branch_of, phantom


@pytest.fixture(scope="module")
def oriented():
    m, _ = phantom()
    res = extract(m, Geometry(SP))
    graph = topology.orient(res)
    return res, graph


def test_graph_is_tree_from_root(oriented):
    res, graph = oriented
    assert len(graph["roots"]) == 1
    assert len(graph["edges"]) == len(graph["segments"]) - len(graph["root_segments"])
    assert all(s["component"] == 0 for s in graph["segments"])
    junctions = [p for p in graph["points"] if p["type"] == "junction"]
    assert len(junctions) == 1 and junctions[0]["degree"] == 3


def test_generations_and_parents(oriented):
    res, graph = oriented
    by_name = {}
    for s, g in zip(res["segments"], graph["segments"]):
        name = _branch_of(res["geom"].crop_mm_to_full_mm(s["points_mm"]))[0]
        by_name.setdefault(name, []).append(g)
    # the trunk can be split in two by the root (degree 2): always generation 0
    assert all(g["generation"] == 0 for g in by_name["trunk"])
    trunk_ids = {g["id"] for g in by_name["trunk"]}
    for name in ("branch_1.5", "branch_1.0"):
        (g,) = by_name[name]
        assert g["generation"] == 1 and g["parent"] in trunk_ids and g["children"] == []


def test_inflow_root_on_trunk_end():
    """Root moved to the end of the trunk: whole trunk gen 0, branches gen 1."""
    m, _ = phantom()
    res = extract(m, Geometry(SP))
    graph = topology.orient(res, inflow=True)
    root = next(p for p in graph["points"] if p["type"] == "root")
    assert root["degree"] == 1
    names = {}
    for s, g in zip(res["segments"], graph["segments"]):
        names.setdefault(_branch_of(res["geom"].crop_mm_to_full_mm(s["points_mm"]))[0], []).append(g)
    (trunk,) = names["trunk"]                          # no longer split by the old root
    assert trunk["generation"] == 0 and trunk["parent"] == -1
    for name in ("branch_1.5", "branch_1.0"):
        (g,) = names[name]
        assert g["generation"] == 1 and g["parent"] == trunk["id"]


def test_merge_degree2_keeps_junctions():
    """Chain a-b-c with a junction at c (degree 3): joined only at b (degree 2)."""
    def seg(a, b):
        p = np.array([[0.0, 0, a], [0, 0, b]])
        return dict(ends=(a, b), points_mm=p, radius=np.ones(2), radius_lo=np.ones(2),
                    radius_hi=np.ones(2), length_mm=abs(b - a))
    segs = [seg(0, 1), seg(1, 2), seg(2, 3), seg(2, 4), seg(2, 5)]
    out = topology.merge_degree2(segs)
    ends = sorted(tuple(sorted(s["ends"])) for s in out)
    assert ends == [(0, 2), (2, 3), (2, 4), (2, 5)]
    merged = next(s for s in out if set(s["ends"]) == {0, 2})
    assert merged["length_mm"] == 2 and len(merged["points_mm"]) == 3


def test_polylines_start_at_start_point(oriented):
    res, graph = oriented
    pos = {p["id"]: np.array(p["position_lps"]) for p in graph["points"]}
    for s in res["segments"]:
        p = res["geom"].mm_to_lps(s["points_mm"][[0, -1]])
        assert np.allclose(p[0], pos[s["start"]], atol=1e-3)
        assert np.allclose(p[1], pos[s["end"]], atol=1e-3)


def test_largest_component_only():
    """An island away from the Y (as a small detached vessel piece): its own tree by default,
    none with largest_component_only (as the surface's keep_largest_component)."""
    m, _ = phantom()
    m = m.copy()
    island = np.s_[2:12, 2:8, 2:8]                       # z, y, x: a small box in a corner
    assert (m[island] <= 0).all()
    m[island] = 2.0
    every = extract(m, Geometry(SP))
    largest = extract(m, Geometry(SP), largest_component_only=True)
    assert every["counts"]["components"] == 2 and largest["counts"]["components"] == 1
    assert len(largest["segments"]) < len(every["segments"])
    reference = extract(phantom()[0], Geometry(SP))
    assert len(largest["segments"]) == len(reference["segments"])
