"""Vessels crossing a fissure (from vescan/crossings.py): what ONE crossing is, of which type,
how long. Used by the anatomy stage (and later by the statistics).

Grouping. The crossing label propagates to the whole subtree, so counting labelled groups
inflates the number (75 groups for 4 real crossings on one patient): a crossing is a connected
component of the group graph restricted to the labelled groups (not by geometry: cells share no
points, and stitching them by position split those 4 into 11). Sorted longest first, so
CrossingId agrees between stages.

Type, measured against the fissure sheets (stage surfaces), 13 vessels on 3 patients:
  TRANSLOBAR    crosses and stays on the other side: steep piercings;
  INTERLOBAR    runs along the fissure going in and out: >= 2 piercings, median angle
                < 30 degrees (interlobar 15.0 / 4.1 / 1.6, translobar 39.4 and above: 30 sits
                in the middle of a wide gap, it is not tuned);
  UNCLASSIFIED  labelled but never touches a sheet (periphery, hilum).
A single piercing is translobar by construction. The angle is to the fissure PLANE, 0-90,
unsigned (the sheets' normals are not consistently oriented), from the local plane (radius
5 mm, RMS 0.06-0.31 mm; the global plane leaves 1.4-5.8) and the tangent over +-5 mm. Below
~10 degrees the angle is ill-conditioned ("grazing"), hence the median.
The fraction of length close to the fissure was tried and dropped: it does not separate the
two types.

Three lengths, three questions:
  length_mm                 all the labelled centerline, whole subtree summed;
  length_to_end_mm          the longest single path from the start of the label to a leaf;
  length_after_piercing_mm  the same but from the piercing (usually much shorter: most
                            piercings do not fall on the root group).
Lengths are geometric (the point spacing is not constant) and measured on one cell per group
(the longest), so a branch traced several times is not counted several times.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from vescan_thalweg.lobes import LOBE_ORDER
from vescan_thalweg.surface import representative_cells

from . import labels as L


@dataclass(frozen=True)
class Rules:
    """The crossing thresholds (VeScan's, measured: see the module docstring); in the config
    they are pipeline.anatomy.*."""
    min_piercings_for_interlobar: int = 2
    interlobar_max_median_angle_deg: float = 30.0
    grazing_angle_deg: float = 10.0          # below: "grazing" angle, imprecise
    tangent_window_mm: float = 5.0           # half-window for the centerline tangent
    fissure_patch_radius_mm: float = 5.0     # radius of the local fissure plane
    fissure_patch_min_points: int = 12       # minimum points to estimate it
    duplicate_piercing_mm: float = 0.1       # the same piercing seen by two segments


def lobe_name(value, lobe_order=None):
    """LobeLabelPoint 1..N -> lobe name (on the order of the active lobes); None otherwise."""
    lobe_order = lobe_order or LOBE_ORDER
    value = int(value)
    return lobe_order[value - 1] if 1 <= value <= len(lobe_order) else None


class Centerline:
    """The labelled centerline as arrays: points, cells, group and generation per cell."""

    def __init__(self, xyz, cells, group_of_cell, generation_of_cell, point_segments, point_lobes, radius):
        self.xyz, self.cells = xyz, cells
        self.group_of_cell = [int(g) for g in group_of_cell]
        self.point_segments, self.point_lobes, self.radius = point_segments, point_lobes, radius
        self.representative = representative_cells(self.group_of_cell, cells)
        self.generation = {}
        if generation_of_cell is not None:
            for g, gen in zip(self.group_of_cell, generation_of_cell):
                gen = int(gen)
                if gen >= 0 and (g not in self.generation or gen < self.generation[g]):
                    self.generation[g] = gen


def crossing_geometry(cl: Centerline, crossing_values, lobe_order=None):
    """Group -> {cell_id, length, points, lobes, origin} for the groups with at least one point
    labelled crossing (on the representative cell). A segment counts if both its ends are
    labelled; origin is the lobe of the last point before the label."""
    geometry = {}
    for g, cell in cl.representative.items():
        ids = cl.cells[cell]
        flags = [int(cl.point_segments[i]) in crossing_values for i in ids]
        if not any(flags):
            continue
        lobes = [lobe_name(cl.point_lobes[i], lobe_order) if cl.point_lobes is not None else None for i in ids]
        length = sum(math.dist(cl.xyz[ids[k]], cl.xyz[ids[k + 1]])
                     for k in range(len(ids) - 1) if flags[k] and flags[k + 1])
        first = flags.index(True)
        geometry[g] = dict(cell_id=cell, length=length, points=sum(flags),
                           lobes={lobes[k] for k, f in enumerate(flags) if f and lobes[k]},
                           origin=lobes[first - 1] if first > 0 else None)
    return geometry


def crossing_runs(geometry, edges):
    """Connected components of the labelled groups in the group graph, longest first."""
    neighbours = {}
    for e in edges:
        neighbours.setdefault(e["parent"], set()).add(e["child"])
        neighbours.setdefault(e["child"], set()).add(e["parent"])
    runs, visited = [], set()
    for g in geometry:
        if g in visited:
            continue
        stack, run = [g], set()
        while stack:
            cur = stack.pop()
            if cur in run:
                continue
            run.add(cur)
            visited.add(cur)
            stack.extend(n for n in neighbours.get(cur, ()) if n in geometry and n not in run)
        runs.append(run)
    runs.sort(key=lambda r: -sum(geometry[g]["length"] for g in r))
    return runs


class FissureProbe:
    """Where a segment pierces a fissure sheet, and the sheet's local plane there."""

    def __init__(self, fissures):
        self.points = vtk_to_numpy(fissures.GetPoints().GetData())
        self.obb = vtk.vtkOBBTree()
        self.obb.SetDataSet(fissures)
        self.obb.BuildLocator()
        self.kd = vtk.vtkKdTree()
        self.kd.BuildLocatorFromPoints(fissures.GetPoints())
        contact, fissure = (fissures.GetCellData().GetArray(n) for n in ("FissureContact", "Fissure"))
        self.contact = vtk_to_numpy(contact) if contact else None
        self.fissure = vtk_to_numpy(fissure) if fissure else None

    def intersect(self, a, b):
        pts, cells = vtk.vtkPoints(), vtk.vtkIdList()
        if self.obb.IntersectWithLine(a, b, pts, cells) == 0:
            return []
        return [(np.array(pts.GetPoint(i)), cells.GetId(i)) for i in range(pts.GetNumberOfPoints())]

    def local_plane(self, position, radius, min_points):
        """(normal, RMS, n points) of the plane of the vertices within radius; (None, None, n) if too few."""
        ids = vtk.vtkIdList()
        self.kd.FindPointsWithinRadius(radius, position, ids)
        n = ids.GetNumberOfIds()
        if n < min_points:
            return None, None, n
        patch = self.points[[ids.GetId(i) for i in range(n)]]
        c = patch.mean(axis=0)
        normal = np.linalg.svd(patch - c, full_matrices=False)[2][2]
        return normal, float(np.sqrt((((patch - c) @ normal) ** 2).mean())), n


def _tangent(polyline, index, window):
    def walk(step):
        i, travelled = index, 0.0
        while 0 <= i + step < len(polyline) and travelled < window:
            travelled += float(np.linalg.norm(polyline[i + step] - polyline[i]))
            i += step
        return i
    d = polyline[walk(+1)] - polyline[walk(-1)]
    norm = float(np.linalg.norm(d))
    return d / norm if norm > 1e-9 else None


def _angle_to_plane_deg(tangent, normal):
    return 90.0 - math.degrees(math.acos(min(1.0, abs(float(np.dot(tangent, normal))))))


def longest_downstream(geometry, run, children):
    """Group -> longest labelled path from its start to a leaf, within the run."""
    deepest = {}

    def walk(g, seen):
        if g not in deepest:
            best = max((walk(c, seen | {c}) for c in children.get(g, ()) if c in run and c not in seen),
                       default=0.0)
            deepest[g] = geometry[g]["length"] + best
        return deepest[g]

    for g in run:
        walk(g, {g})
    return deepest


def find_piercings(cl: Centerline, geometry, run, probe, crossing_values, rules: Rules):
    """Every point where a crossing's centerline pierces a fissure (one cell per group)."""
    piercings = []
    for g in sorted(run):
        ids = cl.cells[geometry[g]["cell_id"]]
        line = cl.xyz[ids]
        labelled = [cl.point_segments is None or int(cl.point_segments[i]) in crossing_values for i in ids]
        for k in range(len(line) - 1):
            for pos, fcell in probe.intersect(line[k], line[k + 1]):
                if any(float(np.linalg.norm(pos - p["position"])) < rules.duplicate_piercing_mm for p in piercings):
                    continue
                tangent = _tangent(line, k, rules.tangent_window_mm)
                normal, rms, _ = probe.local_plane(pos, rules.fissure_patch_radius_mm,
                                                   rules.fissure_patch_min_points)
                if tangent is None or normal is None:
                    continue
                # labelled length left in the cell after the piercing (the subtree is added later)
                remaining = float(np.linalg.norm(line[k + 1] - pos))
                for j in range(k + 1, len(line) - 1):
                    if labelled[j] and labelled[j + 1]:
                        remaining += float(np.linalg.norm(line[j + 1] - line[j]))
                angle = _angle_to_plane_deg(tangent, normal)
                radius = float(cl.radius[ids[k]]) if cl.radius is not None else None
                piercings.append({
                    "position": pos, "group": g, "distal_in_cell_mm": remaining,
                    "angle_deg": round(angle, 2), "patch_rms_mm": round(rms, 3),
                    "grazing": angle < rules.grazing_angle_deg,
                    "fissure_contact": int(probe.contact[fcell]) if probe.contact is not None else None,
                    "fissure": int(probe.fissure[fcell]) if probe.fissure is not None else None,
                    "radius_mm": round(radius, 3) if radius is not None else None,
                })
    return piercings


def crossing_type(piercings, rules: Rules):
    if not piercings:
        return L.CROSSING_UNCLASSIFIED
    if len(piercings) < rules.min_piercings_for_interlobar:
        return L.CROSSING_TRANSLOBAR
    median = float(np.median([p["angle_deg"] for p in piercings]))
    return L.CROSSING_INTERLOBAR if median < rules.interlobar_max_median_angle_deg else L.CROSSING_TRANSLOBAR


def analyse(cl: Centerline, edges, crossing_values, fissures=None, rules: Rules = Rules(), lobe_order=None):
    """Every crossing of a structure, longest first. Without `fissures` they are all
    'unclassified' (counts and lengths do not change)."""
    geometry = crossing_geometry(cl, crossing_values, lobe_order)
    if not geometry:
        return []
    probe = FissureProbe(fissures) if fissures is not None else None
    children, parent = {}, {}
    for e in edges:
        children.setdefault(e["parent"], []).append(e["child"])
        parent[e["child"]] = e["parent"]

    vessels = []
    for index, run in enumerate(crossing_runs(geometry, edges), start=1):
        # topological root (parent outside the run), not the minimum generation: two groups of
        # the run can share a generation; (generation, id) only decides between real roots
        key = lambda g: (cl.generation.get(g, 1 << 30), g)
        roots = [g for g in run if parent.get(g) not in run]
        ordered = sorted(roots or run, key=key) + sorted(run - set(roots), key=key)
        from_lobe = next((geometry[g]["origin"] for g in ordered if geometry[g]["origin"]), None)
        to_lobes = set().union(*(geometry[g]["lobes"] for g in run))
        entered = sorted(to_lobes - {from_lobe}) if from_lobe else sorted(to_lobes)

        piercings = find_piercings(cl, geometry, run, probe, crossing_values, rules) if probe else []
        kind = crossing_type(piercings, rules) if probe else L.CROSSING_UNCLASSIFIED
        angles = [p["angle_deg"] for p in piercings]
        deepest = longest_downstream(geometry, run, children)
        to_end = deepest.get(ordered[0]) if ordered else None
        for p in piercings:
            below = max((deepest[c] for c in children.get(p["group"], ()) if c in run), default=0.0)
            p["distal_length_mm"] = round(p.pop("distal_in_cell_mm") + below, 3)
        after = max((p["distal_length_mm"] for p in piercings), default=None)
        radii = [p["radius_mm"] for p in piercings if p["radius_mm"] is not None]

        vessels.append({
            "crossing": index,
            "crossing_type": L.CROSSING_TYPE_NAME[kind],
            "crossing_type_value": kind,
            "from_lobe": from_lobe,
            "to_lobe": "+".join(entered or sorted(to_lobes)) or None,
            "length_mm": round(sum(geometry[g]["length"] for g in run), 3),
            "length_to_end_mm": round(to_end, 3) if to_end is not None else None,
            "length_after_piercing_mm": round(after, 3) if after is not None else None,
            "points": sum(geometry[g]["points"] for g in run),
            "groups": sorted(run),
            "n_groups": len(run),
            "start_generation": cl.generation.get(ordered[0]) if ordered else None,
            "n_piercings": len(piercings),
            "angle_median_deg": round(float(np.median(angles)), 2) if angles else None,
            "angle_min_deg": round(min(angles), 2) if angles else None,
            "angle_max_deg": round(max(angles), 2) if angles else None,
            "fissures": sorted({p["fissure"] for p in piercings if p["fissure"] is not None}),
            "patch_rms_max_mm": round(max(p["patch_rms_mm"] for p in piercings), 3) if piercings else None,
            "radius_mm": round(float(np.median(radii)), 3) if radii else None,
            "piercings": [{k: v for k, v in p.items() if k != "position"} for p in piercings],
        })
    return vessels


def crossing_arrays(cl: Centerline, vessels, crossing_values):
    """CrossingId/CrossingType per cell (every cell of the group, for visualisation) and per
    point (only the points labelled crossing). 0 = no crossing."""
    id_of, type_of = {}, {}
    for v in vessels or []:
        for g in v["groups"]:
            id_of[g], type_of[g] = v["crossing"], v["crossing_type_value"]
    n_cells, n_points = len(cl.cells), len(cl.xyz)
    cell_id, cell_type = np.zeros(n_cells, np.int32), np.zeros(n_cells, np.int32)
    point_id, point_type = np.zeros(n_points, np.int32), np.zeros(n_points, np.int32)
    for cell, g in enumerate(cl.group_of_cell):
        if g not in id_of:
            continue
        cell_id[cell], cell_type[cell] = id_of[g], type_of[g]
        ids = cl.cells[cell]
        if cl.point_segments is not None:
            ids = [i for i in ids if int(cl.point_segments[i]) in crossing_values]
        point_id[ids], point_type[ids] = id_of[g], type_of[g]
    return {"CrossingId": cell_id, "CrossingType": cell_type}, {"CrossingIdPoint": point_id,
                                                               "CrossingTypePoint": point_type}
