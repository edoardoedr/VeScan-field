"""Centerline graph: points (root, junctions, ends) and segments oriented from the root.

The graph is the one TEASAR already produces: each segment goes from a node of degree != 2 (or
a root) to the next. Here it is walked from each component's root, fixing direction, parent,
children and generation. No bifurcation zones: VMTK-style groups are step [D] (tree.py).

Generation: 0 for the segments leaving the root; +1 at each junction crossed (point of
degree >= 3). A root of degree 2 (mid-vessel) is not a junction.

Root: TEASAR starts from the widest point of each component, which for the arteries falls on
the pulmonary trunk bifurcation and for the airways mid-trachea. With `inflow=True` the root
moves to the end where the flow comes in: the nearest one (along the tree) among the ends
whose terminal segment has radius >= inflow_min_radius_fraction * the maximum radius.
The widest point becomes an ordinary junction again, with its bifurcation.
"""
from __future__ import annotations

import heapq
import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

_FLIP = ("points_mm", "radius", "radius_lo", "radius_hi")


def inflow_root(r, segs, inc, min_radius_fraction=0.5):
    """The inflow end of the component rooted (at its widest point) at `r`, or `r` itself."""
    r_max = max(float(np.max(segs[i]["radius"])) for i in inc[r])
    dist, heap, best = {r: 0.0}, [(0.0, r)], None
    while heap:                                     # Dijkstra over the segments (length in mm)
        d, v = heapq.heappop(heap)
        if d > dist[v]:
            continue
        if v != r and len(inc[v]) == 1:
            (i,) = inc[v]
            if np.median(segs[i]["radius"]) >= min_radius_fraction * r_max:
                best = v
                break
        for i in inc[v]:
            w = segs[i]["ends"][1] if segs[i]["ends"][0] == v else segs[i]["ends"][0]
            nd = d + segs[i]["length_mm"]
            if nd < dist.get(w, np.inf):
                dist[w] = nd
                heapq.heappush(heap, (nd, w))
    return r if best is None else best


def inflow_fork(r, segs, inc, coords_mm, radii=2.0):
    """Segments to drop: the "fork" TEASAR puts at the inflow end when the tube is cut (e.g.
    the trachea at the edge of the field of view): leaves that start at the first junction
    after the root and end within `radii` radii of the root."""
    if len(inc.get(r, [])) != 1:
        return set()
    (i0,) = inc[r]
    j = segs[i0]["ends"][1] if segs[i0]["ends"][0] == r else segs[i0]["ends"][0]
    if len(inc[j]) < 3:
        return set()
    r_j = max(float(np.median(segs[i]["radius"])) for i in inc[j])
    drop = set()
    for i in inc[j]:
        w = segs[i]["ends"][1] if segs[i]["ends"][0] == j else segs[i]["ends"][0]
        if i != i0 and len(inc[w]) == 1 and \
                np.linalg.norm(coords_mm[w] - coords_mm[r]) < radii * r_j:
            drop.add(i)
    return drop if len(inc[j]) - len(drop) >= 2 else set()


def _flip(s):
    for k in _FLIP:
        s[k] = s[k][::-1].copy()
    s["ends"] = s["ends"][::-1]


def merge_degree2(segs, keep=()):
    """Join the two segments meeting at a node of degree 2 not in `keep` (an old mid-vessel
    root is left with degree 2 when the root moves to the inflow end)."""
    segs = [dict(s) for s in segs]
    alive = set(range(len(segs)))
    inc: dict[int, list[int]] = {}
    for i, s in enumerate(segs):
        for v in s["ends"]:
            inc.setdefault(v, []).append(i)
    for v in list(inc):
        ids = inc[v]                                # read now: the joins update inc[w]
        if len(ids) != 2 or v in keep or ids[0] == ids[1]:
            continue
        a, b = (segs[i] for i in ids)
        if a["ends"][1] != v:
            _flip(a)
        if b["ends"][0] != v:
            _flip(b)
        for k in _FLIP:
            a[k] = np.concatenate([a[k], b[k][1:]])
        a["ends"] = (a["ends"][0], b["ends"][1])
        a["length_mm"] += b["length_mm"]
        alive.discard(ids[1])
        w = b["ends"][1]
        inc[w] = [ids[0] if i == ids[1] else i for i in inc[w]]
    return [segs[i] for i in sorted(alive)]


def orient(res: dict, inflow: bool = False, inflow_min_radius_fraction: float = 0.5,
           inflow_fork_radii: float = 2.0) -> dict:
    """Orient the segments of `res` (from centerlines.extract) in place and return the graph.

    Each segment gets: start, end (point ids), parent (segment id, -1 from the root),
    children, generation, component. The polylines are flipped to run from `start` (towards
    the root) to `end`. With `inflow` the root is the inflow end (see inflow_root), and
    `res["roots"]` / `res["roots_mm"]` are updated.
    """
    segs = res["segments"]
    inc: dict[int, list[int]] = {}                 # voxel node -> incident segments
    for i, s in enumerate(segs):
        for v in s["ends"]:
            inc.setdefault(v, []).append(i)
    if inflow:
        res["roots"] = [inflow_root(r, segs, inc, inflow_min_radius_fraction) if r in inc else r
                        for r in res["roots"]]
        res["roots_mm"] = res["coords_mm"][res["roots"]]
        drop = set().union(*(inflow_fork(r, segs, inc, res["coords_mm"], inflow_fork_radii) for r in res["roots"]))
        if drop:
            lengths = ", ".join(f"{segs[i]['length_mm']:.1f} mm" for i in sorted(drop))
            logger.info(f"dropped {len(drop)} fork segments at the inflow ({lengths})")
            segs = [s for i, s in enumerate(segs) if i not in drop]
    segs = res["segments"] = merge_degree2(segs, keep=set(res["roots"]))
    inc = {}
    for i, s in enumerate(segs):
        for v in s["ends"]:
            inc.setdefault(v, []).append(i)

    pid: dict[int, int] = {}                       # voxel node -> point id (visit order)
    points = []

    def point(v, comp):
        if v not in pid:
            pid[v] = len(points)
            points.append(dict(id=len(points), voxel=v, degree=len(inc.get(v, [])), component=comp))
        return pid[v]

    for s in segs:
        s.update(start=-1, end=-1, parent=-1, children=[], generation=-1, component=-1)
    roots = []
    for comp, r in enumerate(res["roots"]):
        if r not in inc:
            continue
        roots.append(point(r, comp))
        queue = deque((r, i, -1, 0) for i in inc[r])
        while queue:
            v, i, parent, gen = queue.popleft()
            s = segs[i]
            if s["component"] >= 0:
                continue
            if s["ends"][0] != v:                  # flip the polyline: from v outwards
                _flip(s)
            w = s["ends"][1]
            s.update(start=point(v, comp), end=point(w, comp), parent=parent,
                     generation=gen, component=comp)
            if parent >= 0:
                segs[parent]["children"].append(i)
            nxt = gen + (len(inc[w]) >= 3)
            queue.extend((w, j, i, nxt) for j in inc[w] if j != i)

    lost = [i for i, s in enumerate(segs) if s["component"] < 0]
    if lost:
        logger.warning(f"{len(lost)} segments not reached from the roots")

    root_set = set(roots)
    for p in points:
        p["type"] = ("root" if p["id"] in root_set else "end" if p["degree"] == 1
                     else "junction" if p["degree"] >= 3 else "node")
    pos = res["geom"].mm_to_lps(res["coords_mm"][[p["voxel"] for p in points]]) if points else []
    # radius at the point: from the first sample of the segments starting or ending there
    rad = {p["id"]: [] for p in points}
    for s in segs:
        if s["start"] >= 0:
            rad[s["start"]].append(float(s["radius"][0]))
            rad[s["end"]].append(float(s["radius"][-1]))
    for p, x in zip(points, pos):
        p["position_lps"] = [round(float(c), 4) for c in x]
        p["radius_mm"] = round(max(rad[p["id"]]), 4) if rad[p["id"]] else None
        del p["voxel"]

    segments = [dict(id=i, start=s["start"], end=s["end"], parent=s["parent"],
                     children=s["children"], generation=s["generation"], component=s["component"],
                     length_mm=round(s["length_mm"], 4),
                     radius_mean_mm=round(float(np.mean(s["radius"])), 4),
                     radius_median_mm=round(float(np.median(s["radius"])), 4))
                for i, s in enumerate(segs)]
    gens = [s["generation"] for s in segs if s["generation"] >= 0]
    logger.info(f"graph: {len(points)} points, {len(roots)} roots, "
                f"{sum(p['type'] == 'junction' for p in points)} junctions, max generation {max(gens, default=-1)}")
    return dict(space="LPS",
                roots=roots,
                root_segments=[i for i, s in enumerate(segs) if s["parent"] < 0 and s["component"] >= 0],
                points=points,
                segments=segments,
                edges=[dict(parent=s["parent"], child=i) for i, s in enumerate(segs) if s["parent"] >= 0])
