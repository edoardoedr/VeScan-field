"""[E] The tree files in VeScan's format (stage 5, build_graph), read by stages 6-11.

Writes into <results>/final_export_<structure>/supporting_files/ (vescan_field.paths.PipelinePaths):
  05_branch_tree.vtk              one cell per branch and per bifurcation child; cell data
                                  GroupIds, GroupId, Blanking, IsBifurcation, Generation,
                                  Length, AverageRadius; point data Radius
  05_branch_tree_topology.json    {"roots", "nodes", "edges"} (virtual root for the veins)
  05_bifurcation_points.mrk.json  one Slicer point per bifurcation
  05_branch_tree_curves.mrk.json  one Slicer curve per branch
(01_preprocessed.vtk is written by the surfaces stage.)

Our own code, with no VeScan dependency: compute_generations, build_graph_data (with
find_substantial_roots for the virtual root; the walks are vescan_field.group_tree's) and the
bifurcation and curve markups reproduce vescan/stages/build_graph.py and network.py, with
byte-identical output. The groups (`groups`, `adjacency`, `roots`) come from
tree.build_groups, in build_graph's format. CenterlineIds/TractIds are not written: only
VeScan's stages 4-5 use them.
"""
from __future__ import annotations

import json
import logging
import numpy as np

from vescan_field.group_tree import GroupTree
from vescan_field.io import MARKUPS_SCHEMA, control_point, fiducial_markups
from vescan_field.paths import PipelinePaths
from vescan_field.surface import save_surface

logger = logging.getLogger(__name__)

VIRTUAL_ROOT_GROUP_ID = -1           # never a real GroupId (>= 0)


# --------------------------------------------------------------------------- generations and topology
def compute_generations(roots, adjacency, groups):
    """Breadth-first from the roots: generation = blanked groups (bifurcations) crossed; the
    trunk is 0. Writes groups[g]["generation"] and returns {group: generation}."""
    tree = GroupTree.from_groups(groups, adjacency, roots)
    generation = {r: 0 for r in roots}
    for parent, child in tree.bfs_edges():
        generation[child] = generation[parent] + (1 if groups.get(parent, {}).get("blanked") else 0)
    for g, gen in generation.items():
        if g in groups:
            groups[g]["generation"] = gen
    return generation


def find_substantial_roots(roots, tree: GroupTree, min_length_fraction=0.1):
    """The roots heading a real trunk: downstream tree >= min_length_fraction of the longest."""
    lengths = {r: tree.length(tree.subtree(r)) for r in roots}
    longest = max(lengths.values(), default=0.0)
    if longest <= 0.0:
        return list(roots)
    return [r for r in roots if lengths[r] >= longest * min_length_fraction]


def build_graph_data(groups, adjacency, roots, add_virtual_root=False,
                     virtual_root_min_length_fraction=0.1):
    """{"roots", "nodes", "edges"} as VeScan's build_graph_data: one node per group (key =
    GroupId as a string), one edge per parent -> child pair. With add_virtual_root and more
    than one substantial root (veins, confluence), a virtual node (-1, generation -1, not
    blanked) is the parent of the substantial roots; the others stay separate roots."""
    nodes = {str(gid): {"group_id": gid, "blanked": g["blanked"], "generation": g["generation"],
                        "length": g["length"], "avg_radius": g["avg_radius"],
                        "n_candidate_cells": g["n_candidate_cells"]}
             for gid, g in groups.items()}
    edges = [{"parent": gid, "child": child}
             for gid, links in adjacency.items() for child in links["downstream"]]

    if add_virtual_root and len(roots) > 1:
        tree = GroupTree.from_groups(groups, adjacency, roots)
        substantial = find_substantial_roots(roots, tree, virtual_root_min_length_fraction)
        if len(substantial) > 1:
            # where each trunk starts: its first point or that of the first descendant with geometry
            firsts = (tree.first_below(r, lambda g: g in groups) for r in substantial)
            starts = [groups[g]["positions"][0] for g in firsts if g is not None]
            centroid = ([sum(p[i] for p in starts) / len(starts) for i in range(3)]
                        if starts else [0.0, 0.0, 0.0])
            nodes[str(VIRTUAL_ROOT_GROUP_ID)] = {
                "group_id": VIRTUAL_ROOT_GROUP_ID, "virtual": True, "blanked": False,
                "generation": -1, "length": 0.0, "avg_radius": None, "n_candidate_cells": 0,
                "position": centroid}
            edges = edges + [{"parent": VIRTUAL_ROOT_GROUP_ID, "child": r} for r in substantial]
            stubs = [r for r in roots if r not in set(substantial)]
            roots = [VIRTUAL_ROOT_GROUP_ID] + stubs
            logger.info(f"virtual root above {len(substantial)} roots; {len(stubs)} roots stay separate")
    return {"roots": roots, "nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------- Slicer markups
def bifurcation_markups(groups, adjacency, coordinate_space="LPS"):
    """One point per blanked group, at its first point, labelled with the number of children."""
    bif = [g for g in groups.values() if g["blanked"]]
    labels = [f"bifurcation_group{g['group_id']}_{len(adjacency[g['group_id']]['downstream'])}children"
              for g in bif]
    return fiducial_markups("bifurcations", labels, [g["positions"][0] for g in bif], coordinate_space)


def curve_markups(branches, names, coordinate_space="LPS", base_name="branch"):
    """One open curve per branch (list of positions), named "<base_name> (<name>)"."""
    markups = [{"type": "Curve", "coordinateSystem": coordinate_space, "coordinateUnits": "mm",
                "locked": False, "labelFormat": "%N-%d", "name": f"{base_name} ({name})",
                "controlPoints": [control_point(i + 1, "", p) for i, p in enumerate(positions)]}
               for name, positions in zip(names, branches)]
    return {"@schema": MARKUPS_SCHEMA, "markups": markups}


# --------------------------------------------------------------------------- 05_branch_tree.vtk
def branch_tree_polydata(groups):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    pts, rad, lines = [], [], vtk.vtkCellArray()
    cell_vals = {k: [] for k in ("GroupIds", "Blanking", "Generation", "Length", "AverageRadius")}
    off = 0
    for gid, gr in groups.items():
        for p, r in gr["cells"]:
            lines.InsertNextCell(len(p))
            for i in range(off, off + len(p)):
                lines.InsertCellPoint(i)
            off += len(p)
            pts.append(p)
            rad.append(r)
            cell_vals["GroupIds"].append(gid)
            cell_vals["Blanking"].append(int(gr["blanked"]))
            cell_vals["Generation"].append(gr["generation"])
            cell_vals["Length"].append(gr["length"])
            cell_vals["AverageRadius"].append(gr["avg_radius"] or 0.0)
    pd = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.concatenate(pts).astype(np.float64), deep=True))
    pd.SetPoints(points)
    pd.SetLines(lines)

    def add(data, name, values, vtk_type):
        arr = numpy_to_vtk(np.ascontiguousarray(values), deep=True, array_type=vtk_type)
        arr.SetName(name)
        data.AddArray(arr)

    add(pd.GetPointData(), "Radius", np.concatenate(rad).astype(np.float64), vtk.VTK_DOUBLE)
    cd = pd.GetCellData()
    for name, key in [("Blanking", "Blanking"), ("GroupIds", "GroupIds"), ("GroupId", "GroupIds"),
                      ("IsBifurcation", "Blanking"), ("Generation", "Generation")]:
        add(cd, name, np.array(cell_vals[key], np.int32), vtk.VTK_INT)
    for name in ("Length", "AverageRadius"):
        add(cd, name, np.array(cell_vals[name], np.float64), vtk.VTK_DOUBLE)
    return pd


# --------------------------------------------------------------------------- all together
def write_tree_files(paths: PipelinePaths, groups, adjacency, roots, add_virtual_root=False,
                     virtual_root_min_length_fraction=0.1):
    paths.make_dirs()

    compute_generations(roots, adjacency, groups)
    save_surface(branch_tree_polydata(groups), paths.branch_tree)
    data = build_graph_data(groups, adjacency, roots, add_virtual_root=add_virtual_root,
                            virtual_root_min_length_fraction=virtual_root_min_length_fraction)
    paths.branch_tree_topology.write_text(json.dumps(data, indent=2))
    branches = [g for g in groups.values() if not g["blanked"]]
    if paths.debug:
        paths.bifurcation_points.write_text(
            json.dumps(bifurcation_markups(groups, adjacency), indent=2))
        curves = curve_markups([g["positions"] for g in branches],
                               [f"group{g['group_id']}_gen{g['generation']}" for g in branches])
        paths.branch_tree_curves.write_text(json.dumps(curves, indent=2))

    gens = [g["generation"] for g in branches]
    logger.info(f"VeScan: {len(branches)} branches, {len(groups) - len(branches)} bifurcations, "
                f"max generation {max(gens, default=-1)}, roots {data['roots'][:5]}"
                f"{'...' if len(data['roots']) > 5 else ''} -> {paths.supporting_files_dir}/")
    return dict(branches=len(branches), bifurcations=len(groups) - len(branches),
                max_generation=max(gens, default=-1), roots=data["roots"],
                topology_nodes=len(data["nodes"]), topology_edges=len(data["edges"]))
