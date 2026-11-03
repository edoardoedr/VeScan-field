"""VeScan-style groups on the Y phantom: branches + one minimal bifurcation (point before/after)."""
import numpy as np
import pytest

from vescan_field.stages.centerlines import topology, tree
from vescan_field.stages.centerlines.extract import extract
from vescan_field.field import Geometry
from .test_phantom import SP, phantom


@pytest.fixture(scope="module")
def groups():
    m, _ = phantom()
    res = extract(m, Geometry(SP))
    graph = topology.orient(res)
    return tree.build_groups(res, graph, halfwidth_mm=0.5)


def test_one_minimal_bifurcation(groups):
    gr, adj, roots = groups
    bif = [g for g in gr.values() if g["blanked"]]
    assert len(bif) == 1
    (b,) = bif
    assert b["n_candidate_cells"] == 2                       # one cell per child
    for p, _ in b["cells"]:
        L = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        assert 0.9 < L < 1.1                                  # 0.5 mm before + 0.5 mm after
    assert len(adj[b["group_id"]]["upstream"]) == 1 and len(adj[b["group_id"]]["downstream"]) == 2


def test_adjacency_and_roots(groups):
    gr, adj, roots = groups
    assert len(roots) == 1 and not adj[roots[0]]["upstream"]
    for gid, a in adj.items():
        for d in a["downstream"]:
            assert gid in adj[d]["upstream"]
    # every group reachable from the root
    seen, stack = set(), list(roots)
    while stack:
        x = stack.pop()
        seen.add(x)
        stack.extend(adj[x]["downstream"])
    assert seen == set(gr)


def test_generations_like_build_graph(groups):
    """Same rule as build_graph.compute_generations: +1 after a blanked group."""
    gr, adj, roots = groups
    gen = {r: 0 for r in roots}
    queue = list(roots)
    while queue:
        x = queue.pop(0)
        for d in adj[x]["downstream"]:
            gen[d] = gen[x] + (1 if gr[x]["blanked"] else 0)
            queue.append(d)
    (b,) = [g for g in gr.values() if g["blanked"]]
    assert gen[b["group_id"]] == 0
    assert [gen[d] for d in adj[b["group_id"]]["downstream"]] == [1, 1]


def test_branches_and_bifurcation_are_contiguous(groups):
    gr, adj, roots = groups
    (b,) = [g for g in gr.values() if g["blanked"]]
    parent = gr[adj[b["group_id"]]["upstream"][0]]
    for (p, _), child in zip(b["cells"], adj[b["group_id"]]["downstream"]):
        assert np.allclose(p[0], parent["positions"][-1])     # the parent branch ends at the point before
        assert np.allclose(p[-1], gr[child]["positions"][0])  # the child starts at the point after
