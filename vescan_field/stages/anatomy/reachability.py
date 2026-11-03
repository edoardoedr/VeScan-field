"""[VeScan stage 9] Which lobes each group (branch or bifurcation) of the tree leads to.

1. Containment (the seed only): a branch belongs directly to a lobe if at least one of its
   points is inside the lobe surface and the fraction inside is >= containment_threshold
   (0.1, low on purpose: segmented trees often stop before the periphery, and only the distal
   part of a branch enters the lobe it feeds). The proximal trunks lie in the mediastinum,
   outside every lobe: that is expected, step 2 handles them.
2. Reachability: bottom-up walk (post-order); the lobes a group reaches are its direct ones
   joined with its children's. More than one lobe = shared trunk.

Labels (N = number of ACTIVE lobes in this run, not always 5):
  LobeMask / NumReachableLobes / Reaches_<LOBE>  from reachability (subtree): Reaches is
      0 (does not get there), 1 (only there), 2 (there and elsewhere)
  LobeLabel (per group, LOCAL)  0 none/fragment, 1..N inside that lobe only, N+1 outside every
      lobe (trunk), N+2 its points touch more than one lobe, or it is downstream of one that
      does: a vessel that has crossed stays marked. Local and not per subtree because a group
      entirely inside one lobe must not be painted as trunk just because a distant descendant
      enters another lobe (real case, groups 344-350 of an arterial tree).
  LobeLabelPoint (per point)  the same scale point by point, without propagation: N+2 only
      where a point is inside two lobes at once (rare, on the fissures). The point where the
      value changes along the centerline is where the vessel enters or leaves a lobe.

Fragments (components detached from the main one) are not classified: they would be assigned
to the lobe they happen to be near. No-lobe regions: the maximal subtrees that reach no active
lobe; stage 10 needs them when a lobe is excluded (its territory still goes to the right side).
They are "substantial" from 25 mm: on 7 real runs the artefacts reach 20.9 mm, the smallest
real territory 29.8 mm.
"""
from __future__ import annotations

import logging

import numpy as np

from vescan_field.group_tree import GroupTree
from vescan_field.surface import cell_array, cell_point_ids, points, points_inside

logger = logging.getLogger(__name__)

GROUP_ID, BLANKED = "GroupId", "IsBifurcation"


def classify_group(lobes):
    if not lobes:
        return "none"
    if len(lobes) == 1:
        return f"exclusive:{next(iter(lobes))}"
    return "shared:" + "+".join(sorted(lobes))


def lobes_by_fraction(group_points, groups, inside, threshold):
    """For each group in `groups`: the fraction of its points inside each lobe, and the lobes with
    at least one point inside and fraction >= threshold. `inside`: lobe -> bool per point."""
    fractions = {g: {} for g in groups}
    for lobe, inside_lobe in inside.items():
        for g in groups:
            ids = group_points.get(g, ())
            if len(ids):
                fractions[g][lobe] = int(inside_lobe[ids].sum()) / len(ids)
    direct = {g: {lobe for lobe, f in fr.items() if f > 0 and f >= threshold} for g, fr in fractions.items()}
    return direct, fractions


def propagate(tree: GroupTree, direct):
    """Reached lobes = direct ones + the children's, bottom-up."""
    reachable = {}
    for g in tree.post_order():
        lobes = set(direct.get(g, ()))
        for c in tree.children.get(g, []):
            lobes |= reachable.get(c, set())
        reachable[g] = lobes
    return reachable


def no_lobe_regions(tree: GroupTree, reachable, fragments, min_length_mm):
    """The maximal subtrees that reach no lobe, with the lobes reached by the nearest ancestor
    that reaches any (the anatomical anchor). Returns (regions, region of each group, regions
    below each group)."""
    regions, region_of, visited = [], {}, set()
    stack = [(r, frozenset()) for r in tree.roots]
    while stack:
        g, anchor = stack.pop()
        if g in visited or g in fragments:
            continue
        visited.add(g)
        if not reachable.get(g):
            members = tree.subtree(g, fragments)
            regions.append(dict(root=g, root_generation=tree.nodes.get(g, {}).get("generation", 0),
                                groups=sorted(members), n_groups=len(members),
                                length_mm=round(sum(tree.nodes[m].get("length", 0.0)
                                                    for m in members if m in tree.nodes), 3),
                                parent_reachable_lobes=sorted(anchor)))
            for m in members:
                region_of.setdefault(m, len(regions) - 1)
                visited.add(m)
            continue
        here = frozenset(reachable[g])
        stack.extend((c, here) for c in tree.children.get(g, []) if c not in visited)
    for r in regions:
        r["substantial"] = r["length_mm"] >= min_length_mm

    below = {}
    for g in tree.post_order():
        s = {region_of[g]} if g in region_of else set()
        for c in tree.children.get(g, []):
            s |= below.get(c, set())
        below[g] = s
    return regions, region_of, below


def group_lobe_labels(tree: GroupTree, lobe_order, own, fragments):
    """LobeLabel: top-down, with the crossing propagated to every descendant."""
    n = len(lobe_order)
    labels, tainted, visited = {}, {}, set()
    stack = list(tree.roots)
    while stack:
        g = stack.pop()
        if g in visited:
            continue
        visited.add(g)
        mine = own.get(g, set())
        is_tainted = tainted.get(g, False) or len(mine) > 1
        labels[g] = n + 2 if is_tainted else (n + 1 if not mine else 1 + lobe_order.index(next(iter(mine))))
        for c in tree.children.get(g, []):
            if c not in visited:
                tainted[c] = tainted.get(c, False) or is_tainted
                stack.append(c)
    for g in fragments:
        labels[g] = 0
    return labels


def point_lobe_labels(inside, lobe_order, fragment_points):
    """LobeLabelPoint: 0 fragment, 1..N inside that lobe only, N+1 outside, N+2 inside several lobes."""
    n = len(lobe_order)
    stack = np.stack([inside[lobe] for lobe in lobe_order])
    count = stack.sum(axis=0)
    labels = np.where(count == 0, n + 1, np.where(count > 1, n + 2, 1 + stack.argmax(axis=0)))
    labels[list(fragment_points)] = 0
    return labels.astype(np.int32)


def label_reachability(poly, tree: GroupTree, lobes, tolerance=1e-4, threshold=0.1,
                       min_region_mm=25.0):
    """Returns (cell arrays {name: values}, LobeLabelPoint array, legend as VeScan's JSON).
    `lobes`: lobe -> closed surface, in the order of the active lobes."""
    lobe_order = list(lobes)
    group_of_cell = cell_array(poly, GROUP_ID).astype(int)
    blanked_cell = cell_array(poly, BLANKED)
    cells = cell_point_ids(poly)

    group_points, blanked = {}, {}
    for cell, g in enumerate(group_of_cell):
        g = int(g)
        blanked[g] = bool(blanked_cell[cell]) if blanked_cell is not None else False
        group_points.setdefault(g, []).append(cells[cell])
    group_points = {g: np.concatenate(ids) for g, ids in group_points.items()}

    fragments = tree.fragments()
    if fragments:
        logger.info(f"{len(fragments)} groups in detached fragments ({tree.length(fragments):.1f} mm): not classified")

    # each centerline point tested once per lobe (cells share no points)
    xyz = points(poly)
    inside = {lobe: points_inside(surf, xyz, tolerance) for lobe, surf in lobes.items()}

    branches = [g for g in blanked if not blanked[g] and g not in fragments]
    direct, fractions = lobes_by_fraction(group_points, branches, inside, threshold)
    reachable = propagate(tree, direct)
    regions, region_of, below = no_lobe_regions(tree, reachable, fragments, min_region_mm)

    # LobeLabel looks at the own points of EVERY group, bifurcations included
    everyone = [g for g in blanked if g not in fragments]
    own, own_fractions = lobes_by_fraction(group_points, everyone, inside, threshold)
    group_labels = group_lobe_labels(tree, lobe_order, own, fragments)
    # a fragment group can have no cell (a stub dropped by the tree's minimum length)
    fragment_points = {int(i) for g in fragments if g in group_points for i in group_points[g]}
    point_labels = point_lobe_labels(inside, lobe_order, fragment_points)

    def per_cell(fn):
        return np.array([fn(int(g)) for g in group_of_cell], np.int32)

    reach = lambda g: reachable.get(g, set())
    arrays = {"LobeMask": per_cell(lambda g: sum(1 << i for i, l in enumerate(lobe_order) if l in reach(g))),
              "NumReachableLobes": per_cell(lambda g: len(reach(g))),
              "LobeLabel": per_cell(lambda g: group_labels.get(g, 0))}
    for lobe in lobe_order:
        arrays[f"Reaches_{lobe}"] = per_cell(lambda g, lobe=lobe: 0 if lobe not in reach(g)
                                             else (1 if len(reach(g)) == 1 else 2))

    n = len(lobe_order)
    legend = {
        "lobe_order": lobe_order,
        "lobe_bit": {lobe: bit for bit, lobe in enumerate(lobe_order)},
        "containment_threshold": threshold,
        "no_lobe_regions": regions,
        "lobe_label_legend": {
            "0": "none",
            **{str(i + 1): f"exclusive:{lobe}" for i, lobe in enumerate(lobe_order)},
            str(n + 1): "trunk (this group's own points are outside every lobe)",
            str(n + 2): "crossing (this group's own points touch >1 lobe, or it is downstream of a group that does)",
        },
        "lobe_label_point_legend": {
            "0": "none (disconnected fragment)",
            **{str(i + 1): f"exclusive:{lobe}" for i, lobe in enumerate(lobe_order)},
            str(n + 1): "trunk (this exact point is outside every lobe)",
            str(n + 2): "overlap (this exact point simultaneously tests inside more than one lobe surface - "
                        "not propagated downstream like the group-level 'crossing' above; expected to be "
                        "rare, only at fissure overlaps). The point where a vessel actually enters/leaves a "
                        "lobe instead shows up as a plain value change between consecutive points along the "
                        "'LobeLabelPoint' point array.",
        },
        "groups": {
            str(g): {
                "blanked": node["blanked"],
                "generation": node["generation"],
                "direct_lobes": sorted(direct.get(g, ())),
                "lobe_fractions": fractions.get(g, {}),
                "reachable_lobes": sorted(reach(g)),
                "classification": classify_group(reach(g)),
                "own_lobes": sorted(own.get(g, ())),
                "own_lobe_fractions": own_fractions.get(g, {}),
                "lobe_label": group_labels.get(g, 0),
                "no_lobe_region": region_of.get(g),
                "no_lobe_regions_below": sorted(below.get(g, ())),
            }
            for g, node in tree.nodes.items()
        },
    }
    shared = sum(1 for g in tree.nodes if len(reach(g)) > 1)
    logger.info(f"lobes: {sum(1 for d in direct.values() if d)}/{len(direct)} branches with a direct lobe, "
                f"{shared} shared trunks, {sum(1 for g in tree.nodes if not reach(g))} groups without a lobe")
    return arrays, point_labels, legend
