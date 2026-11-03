"""[VeScan stage 10] The anatomical segment of each group and each point of the centerline.

Three levels, read from the stage-9 legend (reachability):
  1. LobeLabel = crossing -> crossing (the group touches several lobes, or is downstream).
  2. LobeLabel = one lobe -> that lobar label ("inside a lobe, you are that lobe's vessel"):
     before the subtree, because a "comb" of branches all inside RUL has a subtree reaching
     all 5 lobes (real case on the veins) and would end up as trunk.
  3. outside every lobe -> from the subtree, by LUNGS reached: one -> main right/left (or lobar
     if it reaches a single lobe of that lung), both -> trunk/trachea. Veins have no trunk:
     the confluence group is put on a side cell by cell, comparing the cell's mean x with the
     patient's midline.

Per point (AnatomicalSegmentPoint) the boundaries fall where they happen: a main vessel becomes
lobar at the point where the centerline enters the lobe, a crossing starts at the point where
the vessel changes lobe. A point outside every lobe is never lobar (it is "right artery up to
the lobe"); a subtree reaching no lobe stays 0 (e.g. a stub as wide as the trunk attached to
the bifurcation: an artefact, not a main artery).

Excluded lobes (LobeScheme): stage 9's no-lobe regions are placed on a side from their anchor
(the side of the ancestor that reaches lobes), so the territory of an excluded lobe stays main
right/left instead of 0. With all 5 lobes nothing changes.
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from vescan_field.lobes import BOTH_SIDES, LOBE_ORDER, RIGHT_LOBES, SIDE_LEFT, SIDE_RIGHT, sides_of

from . import labels as L

logger = logging.getLogger(__name__)


class LobeScheme:
    """The active lobes of the stage-9 run and the side of its no-lobe regions.

    Two scales: LobeLabel is numbered 1..N on the ACTIVE lobes, AnatomicalSegment on the 5
    canonical ones; the active order is read from the legend (`lobe_order`), never assumed."""

    def __init__(self, legend):
        self.lobes = list(legend.get("lobe_order") or LOBE_ORDER)
        self.n = len(self.lobes)
        self.trunk_label, self.crossing_label = self.n + 1, self.n + 2
        self.covered = sides_of(self.lobes)
        self.partial = set(self.lobes) != set(LOBE_ORDER)
        self.regions = legend.get("no_lobe_regions") or []
        self.region_side = self._region_sides()

    def _region_sides(self):
        """Side of the substantial no-lobe regions, in order:
          1. the anchor reaches one lung and the other has active lobes -> that lung;
          2. one lung has no active lobes: the most proximal unplaced region (and those at the
             same generation) is that lung, which can only hang from the trunk;
          3. the remaining ones with the anchor on one lung -> that lung;
          4. otherwise no side (they stay 0)."""
        if not self.partial:
            return {}
        uncovered = BOTH_SIDES - self.covered
        sides, ambiguous = {}, []
        for i, r in enumerate(self.regions):
            if not r.get("substantial"):
                continue
            anchor = sides_of(r.get("parent_reachable_lobes") or ())
            if len(anchor) == 1 and (BOTH_SIDES - anchor) <= self.covered:
                sides[i] = next(iter(anchor))
            elif anchor:
                ambiguous.append(i)
        if len(uncovered) == 1 and ambiguous:
            top = min(self.regions[i].get("root_generation", 0) for i in ambiguous)
            for i in ambiguous:
                if self.regions[i].get("root_generation", 0) == top:
                    sides[i] = next(iter(uncovered))
        for i in ambiguous:
            anchor = sides_of(self.regions[i].get("parent_reachable_lobes") or ())
            if i not in sides and len(anchor) == 1:
                sides[i] = next(iter(anchor))
        return sides

    def sides(self, group):
        """The lungs the group's subtree reaches (lobes + placed regions)."""
        if group is None:
            return set()
        s = sides_of(group.get("reachable_lobes") or ())
        for i in group.get("no_lobe_regions_below") or ():
            if i in self.region_side:
                s.add(self.region_side[i])
        return s

    def lobe_of(self, lobe_label):
        return self.lobes[lobe_label - 1] if 1 <= lobe_label <= self.n else None

    def describe(self):
        if not self.partial:
            return f"all {self.n} lobes"
        missing = [lobe for lobe in LOBE_ORDER if lobe not in self.lobes]
        return f"{self.n}/{len(LOBE_ORDER)} lobes ({', '.join(self.lobes)}; excluded {', '.join(missing)})"


def group_label(group, vessel_type, scheme):
    """The label of a group (cell)."""
    own = scheme.lobe_of(group["lobe_label"])
    if group["lobe_label"] == scheme.crossing_label:
        return L.crossing_value(vessel_type)
    if own is not None:
        return L.lobe_value(own, vessel_type)
    lobes, sides = set(group["reachable_lobes"]), scheme.sides(group)
    if not sides:
        return 0
    if len(lobes) == 1 and sides == sides_of(lobes):     # committed to a single lobe
        return L.lobe_value(next(iter(lobes)), vessel_type)
    if len(sides) == 1:
        return L.side_value(next(iter(sides)), vessel_type)
    if vessel_type == "vein":
        return L.side_value(SIDE_RIGHT, vessel_type)    # provisional: side per cell later
    return L.trunk_value(vessel_type)


def extra_lobar_label(group, vessel_type, scheme):
    """A point outside every lobe: trunk if the subtree feeds both lungs, main if one, 0 if
    none; None for a vein straddling the midline (geometric side)."""
    sides = scheme.sides(group)
    if not sides:
        return 0
    if len(sides) == 1:
        return L.side_value(next(iter(sides)), vessel_type)
    return None if vessel_type == "vein" else L.trunk_value(vessel_type)


def crossing_start(point_lobe_labels, n):
    """Index of the point where a group crosses: the first one in a lobe other than the first
    lobe met, or inside two lobes at once. Points outside every lobe in between do not count."""
    first = None
    for i, v in enumerate(point_lobe_labels):
        if v == n + 2:
            return i
        if not 1 <= v <= n:
            continue
        if first is None:
            first = v
        elif v != first:
            return i
    return None


def straddles_midline(group, scheme):
    """Venous confluence group: outside every lobe and with a subtree on both sides."""
    return group["lobe_label"] == scheme.trunk_label and scheme.sides(group) == BOTH_SIDES


def _side_by_x(x, midline):
    """LPS: more negative x = further right."""
    if midline is None:
        return SIDE_RIGHT
    return SIDE_RIGHT if x < midline else SIDE_LEFT


def midline_x(cell_x, cell_labels, vessel_type, cell_groups, excluded):
    """The patient's midline: mean of the mean x of the right lobar cells and that of the left
    ones (not the barycenter, which shifts with the imbalance between sides: 19.4 mm on a real
    patient). A side without lobar cells uses its main right/left cells, except the
    confluence's (provisional side). None if a side has neither."""
    lobar, main = {SIDE_RIGHT: [], SIDE_LEFT: []}, {SIDE_RIGHT: [], SIDE_LEFT: []}
    side_of_value = {L.side_value(s, vessel_type): s for s in (SIDE_RIGHT, SIDE_LEFT)}
    for cell, value in enumerate(cell_labels):
        offset = int(value) - L.BLOCK_START[vessel_type] - L.OFFSET_LOBE_BASE
        if 0 <= offset < len(LOBE_ORDER):
            lobar[SIDE_RIGHT if LOBE_ORDER[offset] in RIGHT_LOBES else SIDE_LEFT].append(cell_x[cell])
        elif int(cell_groups[cell]) not in excluded and int(value) in side_of_value:
            main[side_of_value[int(value)]].append(cell_x[cell])
    means = []
    for side in (SIDE_RIGHT, SIDE_LEFT):
        samples = lobar[side] or main[side]
        if not samples:
            return None
        means.append(sum(samples) / len(samples))
    return (means[0] + means[1]) / 2


def classify(legend, group_of_cell, cells, xyz, point_lobe_labels, vessel_type):
    """Returns (AnatomicalSegment per cell, AnatomicalSegmentPoint per point, label per group)."""
    scheme = LobeScheme(legend)
    logger.info(f"lobe scheme: {scheme.describe()}")
    groups = legend["groups"]
    group_labels = {int(g): group_label(grp, vessel_type, scheme) for g, grp in groups.items()}
    cell_labels = np.array([group_labels.get(int(g), 0) for g in group_of_cell], np.int32)

    midline = None
    if vessel_type == "vein":
        cell_x = [sum(xyz[ids, 0].tolist()) / len(ids) for ids in cells]
        straddling = {int(g) for g, grp in groups.items() if straddles_midline(grp, scheme)}
        midline = midline_x(cell_x, cell_labels, vessel_type, group_of_cell, straddling)
        if straddling and midline is None:
            logger.warning("no side with committed cells, venous confluence left as it is")
        elif straddling:
            right, left = L.side_value(SIDE_RIGHT, vessel_type), L.side_value(SIDE_LEFT, vessel_type)
            sides_by_group = {}
            for cell, g in enumerate(group_of_cell):
                if int(g) in straddling and cell_labels[cell] in (right, left):
                    cell_labels[cell] = L.side_value(_side_by_x(cell_x[cell], midline), vessel_type)
                    sides_by_group.setdefault(int(g), []).append(int(cell_labels[cell]))
            group_labels.update({g: Counter(v).most_common(1)[0][0] for g, v in sides_by_group.items()})

    point_labels = np.zeros(len(xyz), np.int32)
    n = scheme.n
    for cell, g in enumerate(group_of_cell):
        group = groups.get(str(int(g)))
        ids = cells[cell]
        lobe_labels = [int(v) for v in point_lobe_labels[ids]]
        inherited = group is not None and group["lobe_label"] == n + 2 and len(group["own_lobes"]) <= 1
        start = 0 if inherited else crossing_start(lobe_labels, n)
        start = len(ids) if start is None else start
        for k, (pid, v) in enumerate(zip(ids, lobe_labels)):
            if k >= start:
                value = L.crossing_value(vessel_type)
            elif v == 0:
                value = 0
            elif 1 <= v <= n:
                value = L.lobe_value(scheme.lobes[v - 1], vessel_type)
            else:
                value = extra_lobar_label(group, vessel_type, scheme)
                if value is None:
                    value = L.side_value(_side_by_x(xyz[pid, 0], midline), vessel_type)
            point_labels[pid] = value
    return cell_labels, point_labels, group_labels
