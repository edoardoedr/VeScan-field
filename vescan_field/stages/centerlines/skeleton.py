"""[B] Seedless centerline (TEASAR) and segments between junctions (from the earlier prototype, field_vessels.py)."""
from __future__ import annotations

import logging

import numpy as np
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def teasar(coords_mm, Dn, Gw, Gl, scale=1.5, const=0.5, min_branch_mm=3.0, min_voxels=30,
           largest_component_only=False):
    """Seed-free: root = thickest voxel of each component; the geodesically farthest uncovered
    voxel is added repeatedly, joined to the tree by the weighted shortest path; a tube of
    radius scale*D+const around each branch is covered. With largest_component_only, only the
    component with the most voxels gets a tree (as the surface's keep_largest_component).
    Returns (paths, main path per component, roots as node indices)."""
    ncomp, labels = csgraph.connected_components(Gl, directed=False)
    components = range(ncomp)
    if largest_component_only and ncomp > 1:
        sizes = np.bincount(labels, minlength=ncomp)
        largest = int(np.argmax(sizes))
        dropped = sizes[np.arange(ncomp) != largest]
        logger.info(f"largest component only: {sizes[largest]} voxels kept, {len(dropped)} components "
                    f"dropped ({int(dropped.sum())} voxels, {int((dropped >= min_voxels).sum())} "
                    f"with at least {min_voxels})")
        components = [largest]
    all_paths, mains, comp_sizes, roots = [], [], [], []
    for comp in components:
        nodes = np.flatnonzero(labels == comp)
        if len(nodes) < min_voxels:
            continue
        sw, sl = Gw[nodes][:, nodes], Gl[nodes][:, nodes]
        D = Dn[nodes]
        pts = coords_mm[nodes]
        r = int(np.argmax(D))
        glen = csgraph.dijkstra(sl, directed=False, indices=r)
        _, pred = csgraph.dijkstra(sw, directed=False, indices=r, return_predecessors=True)
        tree = cKDTree(pts)
        valid = np.isfinite(glen)
        on_tree = np.zeros(len(nodes), bool)
        on_tree[r] = True
        comp_paths = []
        # "the farthest valid one": valid nodes can only decrease, so one pass over the nodes
        # in decreasing distance is enough (same result as argmax)
        order = np.lexsort((np.arange(len(nodes)), -np.where(valid, glen, -1.0)))
        pos = 0
        while True:
            while pos < len(order) and not valid[order[pos]]:
                pos += 1
            if pos == len(order):
                break
            t = int(order[pos])
            path = [t]
            while not on_tree[path[-1]] and pred[path[-1]] >= 0:
                path.append(int(pred[path[-1]]))
            path = np.array(path)
            seg_len = float(np.linalg.norm(np.diff(pts[path], axis=0), axis=1).sum())
            hits = tree.query_ball_point(pts[path], scale * D[path] + const)
            valid[np.unique(np.concatenate([np.asarray(h, int) for h in hits]))] = False
            valid[path] = False
            if seg_len >= min_branch_mm or not comp_paths:
                on_tree[path] = True
                comp_paths.append(nodes[path[::-1]])  # from the junction towards the end
        all_paths += comp_paths
        mains.append(comp_paths[0])
        comp_sizes.append(len(nodes))
        roots.append(int(nodes[r]))
    order = np.argsort(comp_sizes)[::-1]
    return all_paths, [mains[i] for i in order], [roots[i] for i in order]


def paths_to_segments(paths, extra_keys=()):
    """Split the union of the paths into segments between nodes of degree != 2 (bifurcations/
    ends) and at the `extra_keys` nodes (the roots: a graph to walk must start at a node)."""
    adj: dict[int, set] = {}
    for p in paths:
        for a, b in zip(p[:-1].tolist(), p[1:].tolist()):
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    key = {n for n, nb in adj.items() if len(nb) != 2} | (set(extra_keys) & set(adj))
    segs, seen = [], set()
    for s in key:
        for nb in adj[s]:
            if (s, nb) in seen:
                continue
            seg = [s, nb]
            seen.update({(s, nb), (nb, s)})
            while seg[-1] not in key:
                a, b = seg[-2], seg[-1]
                nxt = next(x for x in adj[b] if x != a)
                seen.update({(b, nxt), (nxt, b)})
                seg.append(nxt)
            segs.append(np.array(seg))
    return segs
