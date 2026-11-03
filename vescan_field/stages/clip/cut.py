"""Cut of the centerline after N bifurcations (from vescan/stages/cut_graph.py).

The groups with 0 <= generation <= max_generations are kept (0 = trunk only). One cut point per
edge kept parent -> dropped child, at the last point of the parent's longest cell (where the
child would have continued).

With generation_lobes (a subset of the lobes) the limit applies only to the groups that reach no
other lobe: a group that reaches a lobe outside the subset is kept whatever its generation. This
includes the shared trunks that lead both inside and outside the subset, so the kept tree stays
connected (the lobes a group reaches contain those of its children).
"""
from __future__ import annotations

import numpy as np

from vescan_field.group_tree import GroupTree
from vescan_field.surface import cell_array, cell_point_ids, extract_lines, points


def kept_groups(tree: GroupTree, max_generations, exempt=frozenset()):
    """exempt: groups kept at any generation (those that reach a lobe outside generation_lobes)."""
    kept = {g for g, n in tree.nodes.items() if 0 <= n["generation"] <= max_generations or g in exempt}
    if not kept:
        raise ValueError(f"no group with generation <= {max_generations} (the root is 0)")
    return kept


def groups_outside(anatomy, lobes, generation_lobes):
    """Groups of the anatomy centerline (cell arrays GroupId, Reaches_<LOBE>) that reach a lobe of
    `lobes` that is not in generation_lobes."""
    group_of_cell = cell_array(anatomy, "GroupId").astype(int)
    out = set()
    for lobe in lobes:
        if lobe in generation_lobes:
            continue
        reaches = cell_array(anatomy, f"Reaches_{lobe}")
        if reaches is None:
            raise ValueError(f"generation_lobes: no Reaches_{lobe} array in the anatomy centerline")
        out |= set(group_of_cell[reaches > 0].tolist())
    return out


def cut_centerline(poly, tree: GroupTree, max_generations, exempt=frozenset()):
    """Returns (cut centerline, kept cells, dropped cells, cut points)."""
    kept = kept_groups(tree, max_generations, exempt)
    group_of_cell = cell_array(poly, "GroupId").astype(int)
    keep_cell = np.isin(group_of_cell, list(kept))
    kept_cells, dropped_cells = np.flatnonzero(keep_cell), np.flatnonzero(~keep_cell)

    # longest cell of each group (Length per cell; on a tie the first)
    length = cell_array(poly, "Length")
    best = {}
    for cell, g in enumerate(group_of_cell):
        value = float(length[cell]) if length is not None else 0.0
        if int(g) not in best or value > best[int(g)][1]:
            best[int(g)] = (cell, value)
    cells, xyz = cell_point_ids(poly), points(poly)
    cuts = [dict(parent_group_id=e["parent"], dropped_group_id=e["child"],
                 position=tuple(xyz[cells[best[e["parent"]][0]][-1]]))
            for e in tree.edges
            if e["parent"] in kept and e["child"] not in kept and e["parent"] in best]
    return extract_lines(poly, kept_cells), kept_cells, dropped_cells, cuts
