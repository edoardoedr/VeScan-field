"""The centerlines stage on the synthetic Y phantom of tests/field/test_phantom.py: a trunk (r 3.0 mm)
forking into branches of r 1.5 and 1.0 mm, plus a vein (r 1.5 mm) touching a branch. thalweg traces
the margin, the stage writes the 05_ files in both bifurcation modes."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vescan_thalweg.config import CenterlinesConfig, PointBifurcationsConfig
from vescan_thalweg.group_tree import GroupTree
from vescan_thalweg.paths import PipelinePaths
from vescan_thalweg.surface import cell_array, load_surface
from vescan_thalweg.stages import centerlines

SP = np.array([0.7, 0.7, 0.7])
SHAPE = (100, 100, 100)
K = 8.0  # logits per mm
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
    c = J + 0.5 * np.array([27.0, 16.0, 0]) + 3.0 * d
    sdf_v = capsule_sdf(P, c - [0, 0, 25], c + [0, 0, 25], 1.5)
    rng = np.random.default_rng(seed)
    logits = np.stack([np.zeros(SHAPE), K * sdf_a, K * sdf_v]).astype(np.float32)
    logits += rng.normal(0, 0.3, logits.shape).astype(np.float32)
    return np.clip(logits[1] - np.maximum(logits[0], logits[2]), -8, 8).astype(np.float32), sdf_v


class FakeStore:
    """What centerline_graph reads from a store when the margin is given: path and the opened-store
    duck type (open_store returns it as is)."""
    path = "phantom"
    structures = ()

    def margin(self, name, part=None):
        raise AssertionError("the margin is passed in")


def _ctx(tmp_path, m):
    geometry = SimpleNamespace(origin=(0.0, 0.0, 0.0), directions=np.diag(SP))   # world = index * spacing
    ref = SimpleNamespace(scheme=None, part=0, label_value=1)
    return SimpleNamespace(structure="artery", classes=["lung_arteries"], store=FakeStore(), margin=m,
                           geometry=geometry, ref=ref, paths=PipelinePaths(tmp_path, debug=True), timings={})


def _branch_of(p):
    mid = p[len(p) // 2]
    return max(ART, key=lambda x: capsule_sdf(mid[None], x[1], x[2], 0)[0])


@pytest.fixture(scope="module")
def margin():
    return phantom()


@pytest.fixture(scope="module", params=["vmtk", "point"])
def run(request, margin, tmp_path_factory):
    m, sdf_v = margin
    tmp = tmp_path_factory.mktemp(request.param)
    ctx = _ctx(tmp, m)
    cfg = CenterlinesConfig(bifurcations=request.param, point=PointBifurcationsConfig())
    summary = centerlines.run(ctx, cfg)
    return request.param, ctx, summary, sdf_v


def test_one_bifurcation_two_children(run):
    mode, ctx, summary, _ = run
    tree = GroupTree.load(ctx.paths.branch_tree_topology)
    topo = json.loads(ctx.paths.branch_tree_topology.read_text())
    blanked = [n for n in topo["nodes"].values() if n["blanked"]]
    assert len(blanked) == 1, (mode, summary["vescan"])
    (b,) = blanked
    assert len(tree.children[b["group_id"]]) == 2
    assert summary["vescan"]["max_generation"] == 1
    assert len(topo["roots"]) == 1


def test_branch_radii_and_axis(run):
    mode, ctx, _, _ = run
    poly = load_surface(ctx.paths.branch_tree)
    blank = cell_array(poly, "IsBifurcation")
    radius = np.array([poly.GetPointData().GetArray("Radius").GetValue(i) for i in range(poly.GetNumberOfPoints())])
    seen = set()
    for c in range(poly.GetNumberOfCells()):
        if blank[c]:
            continue
        ids = [poly.GetCell(c).GetPointId(i) for i in range(poly.GetCell(c).GetNumberOfPoints())]
        p = np.array([poly.GetPoint(i) for i in ids])
        if np.linalg.norm(np.diff(p, axis=0), axis=1).sum() < 5:
            continue
        keep = slice(len(p) // 5, len(p) - len(p) // 5)                   # away from junctions and ends
        name, a, b, r_true = _branch_of(p)
        seen.add(name)
        assert abs(np.median(radius[ids][keep]) - r_true) < 0.15, (mode, name)
        assert np.median(r_true - capsule_sdf(p[keep], a, b, r_true)) < 0.2, (mode, name)   # on the axis
    assert seen == {"trunk", "branch_1.5", "branch_1.0"}


def test_root_is_the_trunk_inlet(run):
    mode, ctx, _, _ = run
    topo = json.loads(ctx.paths.branch_tree_topology.read_text())
    root = topo["nodes"][str(topo["roots"][0])]
    assert root["generation"] == 0 and not root["blanked"]
    assert abs(root["avg_radius"] - 3.0) < 0.3, mode


def test_vein_not_in_artery_tree(run):
    _, ctx, _, sdf_v = run
    poly = load_surface(ctx.paths.branch_tree)
    pts = np.array([poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())])
    from scipy import ndimage
    assert np.all(ndimage.map_coordinates(sdf_v, (pts / SP).T, order=1) < 0)


def test_debug_files(run):
    mode, ctx, _, _ = run
    assert ctx.paths.thalweg_graph.exists()
    assert ctx.paths.thalweg_split.exists() == (mode == "vmtk")


def test_adjacency_matches_thalwegs_vmtk_port(margin, tmp_path):
    """SplitCenterline.adjacency (indexed) == thalweg.vmtk.utilities.find_adjacent_centerline_group_ids."""
    from thalweg.vmtk import utilities as U

    from vescan_thalweg.config import TraceConfig, VmtkBifurcationsConfig
    from vescan_thalweg.stages.centerlines import split_vmtk
    from vescan_thalweg.stages.centerlines.trace import trace
    graph, name = trace(_ctx(tmp_path, margin[0]), TraceConfig())
    cl = split_vmtk.vmtk_split(graph, name, VmtkBifurcationsConfig())
    split = split_vmtk.SplitCenterline(cl)
    adj = split.adjacency()
    for g in split.group_ids:
        up, down = U.find_adjacent_centerline_group_ids(cl, g)
        assert adj[g] == {"upstream": up, "downstream": down}
