"""[D] From the oriented graph (topology.orient) to VeScan's groups: branches and bifurcations.

Format: VeScan's build_graph one (`groups`, `adjacency`, `roots`), from which export_tree
writes the topology JSON, markups and generations as VeScan does.

Rule (decided 2026-10-25): the bifurcation is a point, the junction; the bifurcation (blanked)
group is the minimal one around it, from "one point before" on the parent to "one point after"
on each child, `halfwidth_mm` (default = resampling step) from the junction. Close junctions
are not merged: bifurcations stay binary as TEASAR finds them. The branch is the rest of the
segment. Generations as build_graph: parent branch g, bifurcation g, children g+1.

Root: each component has one root; the segments leaving the root (2 if it falls mid-vessel)
become generation-0 groups and the largest (by downstream tree) is VeScan's root, the others
its direct children, with no bifurcation between them: so the component stays one for
lobe_reachability (which keeps the largest fragment per root).
"""
from __future__ import annotations

import numpy as np


def cut_polyline(P, R, s0, s1):
    """The part of the polyline between arc lengths s0 < s1 (ends interpolated)."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    inner = (s > s0) & (s < s1)
    ss = np.r_[s0, s[inner], s1]
    pts = np.stack([np.interp(ss, s, P[:, k]) for k in range(3)], 1)
    return pts, np.interp(ss, s, R)


def _group(gid, blanked, cells):
    """A group in build_graph's format; `cells` = list of (LPS points, radii)."""
    lengths = [float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p, _ in cells]
    best = int(np.argmax(lengths))
    p, r = cells[best]
    return dict(group_id=gid, cell_id=-1, n_candidate_cells=len(cells), blanked=blanked,
                positions=[tuple(float(c) for c in x) for x in p], radii=[float(x) for x in r],
                length=lengths[best], avg_radius=float(np.mean(r)), generation=-1,
                cells=cells)


def build_groups(res: dict, graph: dict, halfwidth_mm: float = 0.5, short_segment_divisor: float = 3.0):
    """build_graph-style groups, adjacency and roots from the oriented segments of `res`.

    Returns (groups, adjacency, roots). Each group also has `cells`: the cells to write in
    05_branch_tree.vtk (1 for a branch, 1 per child for a bifurcation).
    """
    g = res["geom"]
    segs = res["segments"]
    points = graph["points"]
    junction = {p["id"] for p in points if p["type"] == "junction"}

    # branch: the segment without the bifurcation parts at the ends that are junctions
    lps, cut = [], []
    for s in segs:
        P = g.mm_to_lps(s["points_mm"])
        L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        h = min(halfwidth_mm, L / short_segment_divisor)   # very short segments: part of them stays a branch
        s0 = h if s["start"] in junction else 0.0
        s1 = L - h if s["end"] in junction else L
        lps.append((P, s["radius"], L))
        cut.append((s0, s1))

    groups, adjacency = {}, {}
    branch_gid = {}
    for i, s in enumerate(segs):
        if s["component"] < 0:
            continue
        P, R, _ = lps[i]
        gid = len(groups)
        groups[gid] = _group(gid, False, [cut_polyline(P, R, *cut[i])])
        adjacency[gid] = dict(upstream=[], downstream=[])
        branch_gid[i] = gid

    # bifurcations: one per junction, one cell per child (point before -> junction -> point after)
    children_at = {}
    parent_at = {}
    for i, s in enumerate(segs):
        if i not in branch_gid:
            continue
        if s["start"] in junction and s["parent"] >= 0:
            children_at.setdefault(s["start"], []).append(i)
            parent_at[s["start"]] = s["parent"]
    for j, kids in children_at.items():
        par = parent_at[j]
        Pp, Rp, Lp = lps[par]
        before = cut_polyline(Pp, Rp, cut[par][1], Lp)       # from the point before to the junction
        cells = []
        for k in kids:
            Pk, Rk, _ = lps[k]
            after = cut_polyline(Pk, Rk, 0.0, cut[k][0])     # from the junction to the point after
            cells.append((np.vstack([before[0], after[0][1:]]), np.r_[before[1], after[1][1:]]))
        gid = len(groups)
        groups[gid] = _group(gid, True, cells)
        adjacency[gid] = dict(upstream=[branch_gid[par]], downstream=[branch_gid[k] for k in kids])
        adjacency[branch_gid[par]]["downstream"].append(gid)
        for k in kids:
            adjacency[branch_gid[k]]["upstream"].append(gid)

    # roots: per component, the segment from the root with the longest downstream tree
    def subtree_length(gid):
        total, stack = 0.0, [gid]
        while stack:
            x = stack.pop()
            total += groups[x]["length"]
            stack.extend(adjacency[x]["downstream"])
        return total

    roots = []
    by_comp = {}
    for i in graph["root_segments"]:
        by_comp.setdefault(segs[i]["component"], []).append(branch_gid[i])
    for comp in sorted(by_comp):
        gids = sorted(by_comp[comp], key=subtree_length, reverse=True)
        main = gids[0]
        for other in gids[1:]:                              # siblings from the same root
            adjacency[main]["downstream"].append(other)
            adjacency[other]["upstream"].append(main)
        roots.append(main)
    return groups, adjacency, roots
