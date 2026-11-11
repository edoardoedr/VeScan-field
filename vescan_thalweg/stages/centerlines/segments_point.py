"""bifurcations = "point": thalweg's graph -> the oriented segments vescan_field's tree.build_groups
reads (the junction is the bifurcation, as in VeScan field).

thalweg's edges already run away from the root (TubeGraph.tree) and are already split at every
node, so the conversion is bookkeeping: one segment per edge, its polyline resampled to step_mm
(LPS, as thalweg gives it), parent = the edge arriving at its start node, and the node types
field's topology.orient would give (root, junction = degree >= 3, end, node).
"""
from __future__ import annotations

import numpy as np


class _LPS:
    """Identity geometry: thalweg's points are already LPS mm (tree.build_groups calls mm_to_lps)."""

    @staticmethod
    def mm_to_lps(p):
        return np.asarray(p, float)


def _resample(P, R, step):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if s[-1] <= 0:
        return P[[0, -1]], R[[0, -1]]
    si = np.linspace(0, s[-1], max(int(np.ceil(s[-1] / step)) + 1, 2))
    return np.stack([np.interp(si, s, P[:, k]) for k in range(3)], 1), np.interp(si, s, R)


def to_segments(graph, name, step_mm):
    """(res, graph_dict) in the format of vescan_field's centerlines.extract + topology.orient."""
    tree = graph.tree(name)
    deg = graph.degree()
    index = {eid: i for i, eid in enumerate(tree.order)}             # breadth-first from the root
    segs = []
    for eid in tree.order:
        e = graph.edges[eid]
        P = graph.edge_points(e)
        R = graph.edge_radius(e)
        R = np.where(R > 0, R, 0.0)                                  # -1 = no inside point found
        P, R = _resample(P, R, step_mm)
        up = tree.parent.get(e.start_node)
        segs.append(dict(points_mm=P, radius=R, start=e.start_node, end=e.end_node,
                         parent=index[up] if up is not None else -1, children=[], generation=-1,
                         component=0, length_mm=float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())))
    for i, s in enumerate(segs):
        if s["parent"] >= 0:
            segs[s["parent"]]["children"].append(i)
    for s in segs:                                                    # +1 at each junction crossed
        par = s["parent"]
        s["generation"] = 0 if par < 0 else segs[par]["generation"] + (deg[s["start"]] >= 3)

    points = []
    for nd in sorted((nd for nd in graph.nodes if nd.structure == name), key=lambda nd: nd.id):
        d = deg.get(nd.id, 0)
        kind = ("root" if nd.id == tree.root else "end" if d == 1 else "junction" if d >= 3 else "node")
        points.append(dict(id=nd.id, type=kind, degree=d, position_lps=[float(v) for v in nd.position]))
    res = dict(segments=segs, geom=_LPS())
    gdict = dict(space="LPS", roots=[tree.root], points=points,
                 root_segments=[i for i, s in enumerate(segs) if s["parent"] < 0])
    return res, gdict
