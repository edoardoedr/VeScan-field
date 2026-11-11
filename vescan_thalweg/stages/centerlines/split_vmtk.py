"""bifurcations = "vmtk": the branch tree from vmtk's split, as vescan_vmtk builds it, with thalweg's
numpy port of vmtk in place of the compiled one.

thalweg.branching.vmtk_branching runs vmtk's chain on the traced graph (source-to-tip paths,
attributes, vmtkbranchextractor): its `split` holds GroupIds / CenterlineIds / TractIds / Blanking
per cell, as vtkvmtkCenterlineBranchExtractor's output in vescan_vmtk. From there the code is
vescan_vmtk/stages/centerlines/tree.py: groups (the longest unique cell of each group, shorter than
min_branch_length dropped), adjacency, orphan-root repair, generations, the topology JSON and
05_branch_tree.vtk. Only SplitCenterline changes: vtkvmtkCenterlineUtilities -> thalweg.vmtk.utilities.
"""
from __future__ import annotations

import json
import logging
from collections import Counter

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from vescan_thalweg.surface import add_cell_array, extract_lines, save_surface

from .export_tree import bifurcation_markups, build_graph_data, compute_generations, curve_markups

logger = logging.getLogger(__name__)

RADIUS, BLANKING, GROUP_IDS, CENTERLINE_IDS, TRACT_IDS = "Radius", "Blanking", "GroupIds", "CenterlineIds", "TractIds"


def vmtk_split(graph, name, vcfg):
    """thalweg's split centerlines (thalweg.vmtk.Centerlines) of structure `name`: the first steps of
    thalweg.branching.vmtk_branching (to_vmtk -> centerline_attributes -> extract_branches), the
    same calls and flags. Its later steps (bifurcation frames, offsets, vectors, branch geometry) are
    not needed for the groups, and bifurcation_vectors refuses a 1-point cell, which the split
    can hold (build_groups skips such cells, as vescan_vmtk does)."""
    from thalweg.adapters import to_vmtk
    from thalweg.vmtk import VMTK_FLAGS, centerline_attributes, extract_branches

    def flags(step):
        return {f: vcfg.vmtk_compatible for f in VMTK_FLAGS.get(step, ())}

    paths = to_vmtk(graph, name, step=vcfg.resampling_step_mm)
    cl = centerline_attributes(paths.centerlines, **flags("centerline_attributes"))
    return extract_branches(cl, **flags("extract_branches"))


def to_polydata(cl):
    """thalweg Centerlines -> vtkPolyData as vtkvmtkCenterlineBranchExtractor's output in vescan_vmtk:
    polyline cells, point array Radius, cell arrays GroupIds/CenterlineIds/TractIds/Blanking."""
    from thalweg.vmtk.centerlines import RADIUS as THALWEG_RADIUS

    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(cl.points, np.float64), deep=True))
    poly.SetPoints(points)
    lines = vtk.vtkCellArray()
    for c in cl.cells:
        lines.InsertNextCell(len(c))
        for i in c:
            lines.InsertCellPoint(int(i))
    poly.SetLines(lines)
    r = numpy_to_vtk(np.ascontiguousarray(cl.point_data[THALWEG_RADIUS], np.float64), deep=True)
    r.SetName(RADIUS)
    poly.GetPointData().AddArray(r)
    for name in (GROUP_IDS, CENTERLINE_IDS, TRACT_IDS, BLANKING):
        add_cell_array(poly, name, np.asarray(cl.cell_data[name]).astype(np.int32))
    return poly


class SplitCenterline:
    """The split centerline with vmtk's group queries (vescan_vmtk's class, on thalweg's port)."""

    def __init__(self, cl):
        from thalweg.vmtk import utilities as U
        self.cl = cl
        self.U = U
        self.poly = to_polydata(cl)
        self.radius = np.asarray(cl.point_data[_thalweg_radius()], float)
        self.group_ids = U.get_groups_id_list(cl, GROUP_IDS)

    def n_points(self, cell):
        return len(self.cl.cells[cell])

    def cells_of(self, gid):
        """The geometrically distinct cells of a group."""
        return self.U.get_group_unique_cell_ids(self.cl, gid, GROUP_IDS)

    def geometry(self, cell):
        ids = self.cl.cells[cell]
        return [tuple(float(v) for v in p) for p in self.cl.points[ids]], [float(r) for r in self.radius[ids]]

    def blanked(self, gid):
        return bool(self.U.is_group_blanked(self.cl, gid, GROUP_IDS, BLANKING))

    def adjacency(self):
        """thalweg.vmtk.utilities.find_adjacent_centerline_group_ids for every group, with the cells
        indexed by (centerline, tract) once: the same lists in the same order, without its
        all-pairs loop (cells^2 per group)."""
        g = np.asarray(self.cl.cell_data[GROUP_IDS]).astype(int)
        c = np.asarray(self.cl.cell_data[CENTERLINE_IDS]).astype(int)
        t = np.asarray(self.cl.cell_data[TRACT_IDS]).astype(int)
        by_tract, cells_of_group = {}, {}
        for i in range(len(g)):
            by_tract.setdefault((c[i], t[i]), []).append(i)
            cells_of_group.setdefault(g[i], []).append(i)
        out = {}
        for gid in self.group_ids:
            up, down = [], []
            for i in cells_of_group.get(gid, []):
                for lst, dt in ((up, -1), (down, +1)):
                    for j in by_tract.get((c[i], t[i] + dt), []):
                        if g[j] != gid and int(g[j]) not in lst:
                            lst.append(int(g[j]))
            out[gid] = {"upstream": up, "downstream": down}
        return out


def _thalweg_radius():
    from thalweg.vmtk.centerlines import RADIUS as THALWEG_RADIUS
    return THALWEG_RADIUS


def _length(positions):
    return sum(sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5 for a, b in zip(positions[:-1], positions[1:]))


# ---------------------------------------------------------------------- from vescan_vmtk tree.py
def build_groups(split, min_length):
    """build_graph's groups: the longest candidate cell of each group, if >= min_length."""
    groups = {}
    for gid in split.group_ids:
        cells = split.cells_of(gid)
        sizes = [split.n_points(c) for c in cells]
        if not cells or max(sizes) < 2:
            continue
        best = cells[sizes.index(max(sizes))]
        positions, radii = split.geometry(best)
        length = _length(positions)
        if length < min_length:
            continue
        groups[gid] = {"group_id": gid, "cell_id": best, "n_candidate_cells": len(cells),
                       "blanked": split.blanked(gid), "positions": positions, "radii": radii,
                       "length": length, "avg_radius": (sum(radii) / len(radii)) if radii else None,
                       "generation": -1}
    return groups


def repair_orphan_roots(groups, adjacency, roots, tolerance):
    """Every orphan root but the longest is re-attached to the group whose last point is nearest to
    its first point (within tolerance). Returns [main root] + the orphans still unattached."""
    if len(roots) <= 1:
        return roots
    main = max(roots, key=lambda r: groups[r]["length"] if r in groups else -1)
    ends = [(gid, g["positions"][-1]) for gid, g in groups.items()]
    still, repaired = [], 0
    for orphan in roots:
        if orphan == main:
            continue
        if orphan not in groups:
            still.append(orphan)
            continue
        start = groups[orphan]["positions"][0]
        parent, best = None, tolerance
        for gid, end in ends:
            if gid == orphan:
                continue
            d = sum((start[i] - end[i]) ** 2 for i in range(3)) ** 0.5
            if d <= best:
                parent, best = gid, d
        if parent is None:
            still.append(orphan)
            continue
        adjacency[orphan]["upstream"].append(parent)
        adjacency[parent]["downstream"].append(orphan)
        repaired += 1
    if repaired:
        logger.info(f"{repaired} orphan groups re-attached by end-point distance")
    return [main] + still


def instances(split, groups, min_length):
    """What is exported: every candidate cell of every kept group (min_length applied per cell)."""
    out = []
    for gid in split.group_ids:
        if gid not in groups:
            continue
        cells = split.cells_of(gid)
        for i, cell in enumerate(cells):
            positions, radii = split.geometry(cell)
            if len(positions) < 2 or _length(positions) < min_length:
                continue
            out.append({"group_id": gid, "instance_index": i, "n_instances": len(cells),
                        "blanked": groups[gid]["blanked"], "generation": groups[gid]["generation"],
                        "positions": positions, "radii": radii})
    return out


def save_models(models, out_dir):
    """05_branch_models/: one polyline model per instance (Branch_ / Bifurcation_Model_...)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in models:
        points = vtk.vtkPoints()                 # float32, as VeScan
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(m["positions"]))
        radius = vtk.vtkDoubleArray()
        radius.SetName(RADIUS)
        for i, p in enumerate(m["positions"]):
            line.GetPointIds().SetId(i, points.InsertNextPoint(p))
            radius.InsertNextValue(m["radii"][i] if m["radii"] else 0.0)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(line)
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)
        poly.GetPointData().AddArray(radius)
        kind = "Bifurcation_Model" if m["blanked"] else "Branch_Model"
        suffix = "" if m["n_instances"] <= 1 else f"_cell{m['instance_index']}"
        save_surface(poly, out_dir / f"{kind}_group{m['group_id']}_gen{m['generation']}{suffix}.vtk")


def branch_tree(split, groups):
    """05_branch_tree.vtk: the cells of the kept groups (in group order, all candidate cells), with
    the split's arrays plus GroupId, IsBifurcation, Generation, Length, AverageRadius."""
    poly = split.poly
    gid = vtk_to_numpy(poly.GetCellData().GetArray(GROUP_IDS)).astype(int)
    group = [groups.get(int(g)) for g in gid]
    add_cell_array(poly, "GroupId", gid.astype(np.int32))
    add_cell_array(poly, "IsBifurcation", vtk_to_numpy(poly.GetCellData().GetArray(BLANKING)).astype(np.int32))
    add_cell_array(poly, "Generation", np.array([g["generation"] if g else -1 for g in group], np.int32))
    add_cell_array(poly, "Length", np.array([g["length"] if g else 0.0 for g in group]), vtk.VTK_DOUBLE)
    add_cell_array(poly, "AverageRadius", np.array([(g["avg_radius"] or 0.0) if g else 0.0 for g in group]),
                   vtk.VTK_DOUBLE)
    kept = [c for g in split.group_ids if g in groups for c in split.cells_of(g)]
    out = extract_lines(poly, kept)
    out.GetPoints().SetData(numpy_to_vtk(vtk_to_numpy(out.GetPoints().GetData()).astype(np.float32), deep=True))
    return out


def write_tree_files(paths, split, vcfg, add_virtual_root, virtual_root_min_length_fraction):
    """The 05_ files from the split (models and markups with paths.debug); returns a summary dict.
    As vescan_vmtk tree.run, with the split in memory instead of read back from 05_branch_tree.vtk."""
    paths.make_dirs()
    groups = build_groups(split, vcfg.min_branch_length)
    adjacency = split.adjacency()
    roots = [g for g, links in adjacency.items() if not links["upstream"]]
    if not roots:
        raise ValueError("no group without upstream: the tree root cannot be found")
    roots = repair_orphan_roots(groups, adjacency, roots, vcfg.orphan_root_repair_tolerance)
    compute_generations(roots, adjacency, groups)

    branches = [g for g in groups.values() if not g["blanked"]]
    unreached = sum(1 for g in groups.values() if g["generation"] < 0)
    children = Counter(len(adjacency[g["group_id"]]["downstream"]) for g in groups.values() if g["blanked"])
    logger.info(f"{len(split.group_ids)} groups in {split.poly.GetNumberOfCells()} split cells; kept {len(groups)} "
                f"({len(branches)} branches, {len(groups) - len(branches)} bifurcations), roots {roots[:5]}"
                f"{'...' if len(roots) > 5 else ''}, max generation "
                f"{max((g['generation'] for g in branches), default=0)}, children per bifurcation {dict(children)}")
    if unreached:
        logger.warning(f"{unreached} groups not reached from the roots")

    models = []
    if paths.debug:
        models = instances(split, groups, vcfg.min_branch_length)
        save_models(models, paths.branch_models_dir)
        curves = [m for m in models if not m["blanked"]]
        with open(paths.branch_tree_curves, "w", encoding="utf-8") as f:
            json.dump(curve_markups([m["positions"] for m in curves],
                                    [f"group{m['group_id']}_gen{m['generation']}" for m in curves]), f, indent=2)
        with open(paths.bifurcation_points, "w", encoding="utf-8") as f:
            json.dump(bifurcation_markups(groups, adjacency), f, indent=2)
    data = build_graph_data(groups, adjacency, roots, add_virtual_root=add_virtual_root,
                            virtual_root_min_length_fraction=virtual_root_min_length_fraction)
    with open(paths.branch_tree_topology, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tree = branch_tree(split, groups)
    save_surface(tree, paths.branch_tree)
    return dict(groups=len(groups), branches=len(branches), bifurcations=len(groups) - len(branches),
                children_per_bifurcation={str(k): v for k, v in sorted(children.items())},
                roots=data["roots"], models=len(models) if paths.debug else None,
                branch_tree_cells=tree.GetNumberOfCells(),
                max_generation=max((g["generation"] for g in branches), default=-1))
