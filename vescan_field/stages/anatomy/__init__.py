"""Stage anatomy: reached lobes, anatomical segments and crossing statistics on the centerlines
(VeScan's stages 9, 10 and 11).

    reachability  which lobes each group leads to; LobeLabel per group and per point
    segments      anatomical segment (trachea/trunk, main right/left, lobar, crossing)
    crossings     vessels crossing a fissure: interlobar / translobar, lengths
    labels        the value scale and the names
    statistics    statistics.json / .csv: the crossing vessels, for reporting

Input: 05_branch_tree.vtk + 05_branch_tree_topology.json (stage centerlines), the lobe surfaces
and lobe_fissures.vtk in the surfaces folder (stage surfaces). Outputs, in VeScan's layout
(vescan_field.paths.PipelinePaths), in <results>/final_export_<structure>/:
    anatomical_segments.vtk     the stage-9 arrays (LobeMask, NumReachableLobes, LobeLabel,
                                Reaches_<LOBE>, LobeLabelPoint) + AnatomicalSegment,
                                AnatomicalSegmentPoint, CrossingId, CrossingType (cell and point)
    supporting_files/debug/09_lobe_reachability.vtk / .json, 10_anatomical_segments.json
                                with pipeline.debug_files only: stage 9 alone + per-group
                                legend; label names and crossings
    statistics.json / .csv      crossing vessels: counts, type, lobes, lengths, angles
                                (pipeline.anatomy.statistics)
Centerlines only: transferring the labels onto the surfaces is the surface_labels stage.
"""
from __future__ import annotations

import dataclasses
import logging
import time
from collections import Counter

import numpy as np
import vtk

from vescan_field import io
from vescan_field.group_tree import GroupTree
from vescan_field.surface import (add_cell_array, add_point_array, cell_array, cell_point_ids, close_if_open,
                             load_surface, point_array, points, save_surface)

from . import crossings as X
from . import labels as L
from . import statistics
from .reachability import label_reachability
from .segments import classify

logger = logging.getLogger(__name__)

VESSEL_TYPE = {"artery": "artery", "veins": "vein", "airways": "airway"}


def _copy_with(poly, cell_arrays=None, point_arrays=None):
    out = vtk.vtkPolyData()
    out.DeepCopy(poly)
    for name, values in (cell_arrays or {}).items():
        add_cell_array(out, name, values)
    for name, values in (point_arrays or {}).items():
        add_point_array(out, name, values)
    return out


def run(ctx, cfg) -> dict:
    """cfg: vescan_field.config.AnatomyConfig."""
    p = ctx.paths
    tree_path, topo_path = p.branch_tree, p.branch_tree_topology
    lobe_paths = {lobe: ctx.surfaces.surface(name) for lobe, name in ctx.lobes.items()}
    missing = [p.name for p in [tree_path, topo_path, *lobe_paths.values()] if not p.exists()]
    if missing:
        logger.warning(f"anatomy skipped: missing {missing} (stages centerlines and surfaces are needed)")
        return dict(skipped=missing)
    vessel_type = VESSEL_TYPE[ctx.structure]
    t0 = time.time()

    poly, tree = load_surface(tree_path), GroupTree.load(topo_path)
    lobes = {lobe: close_if_open(load_surface(p), f"lobe {lobe}") for lobe, p in lobe_paths.items()}
    fissures_path = ctx.surfaces.fissures
    fissures = load_surface(fissures_path) if fissures_path.exists() else None
    if fissures is None:
        logger.warning("lobe_fissures.vtk missing, crossings not classified")

    # [9] reached lobes
    reach_arrays, lobe_label_point, legend = label_reachability(
        poly, tree, lobes, tolerance=cfg.containment_tolerance, threshold=cfg.containment_threshold,
        min_region_mm=cfg.no_lobe_region_min_length_mm)
    poly9 = _copy_with(poly, reach_arrays, {"LobeLabelPoint": lobe_label_point})
    if p.debug:
        p.make_dirs()
        save_surface(poly9, p.lobe_reachability)
        io.write_json(p.lobe_reachability_legend, legend)

    # [10] anatomical segments
    group_of_cell = cell_array(poly, "GroupId").astype(int)
    cells, xyz = cell_point_ids(poly), points(poly)
    cell_seg, point_seg, group_labels = classify(legend, group_of_cell, cells, xyz, lobe_label_point, vessel_type)

    # crossings, on the not yet refined label
    crossing_vals = L.crossing_values(vessel_type)
    cl = X.Centerline(xyz, cells, group_of_cell, cell_array(poly, "Generation"), point_seg,
                      lobe_label_point, point_array(poly, "Radius"))
    rules = X.Rules(**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(X.Rules)})
    vessels = X.analyse(cl, tree.edges, crossing_vals, fissures, rules, lobe_order=legend["lobe_order"])
    crossing_cell, crossing_point = X.crossing_arrays(cl, vessels, crossing_vals)

    # 9/18/28 -> 73-78 where the type is known (cells and points already labelled as crossing)
    refined = {g: L.refined_crossing_value(vessel_type, v["crossing_type_value"])
               for v in vessels for g in v["groups"]}
    refined = {g: r for g, r in refined.items() if r is not None}
    for cell, g in enumerate(group_of_cell):
        r = refined.get(int(g))
        if r is None:
            continue
        if cell_seg[cell] in crossing_vals:
            cell_seg[cell] = r
            group_labels[int(g)] = r
        ids = cells[cell]
        point_seg[ids[np.isin(point_seg[ids], list(crossing_vals))]] = r

    poly10 = _copy_with(poly9, {"AnatomicalSegment": cell_seg, **crossing_cell},
                        {"AnatomicalSegmentPoint": point_seg, **crossing_point})
    save_surface(poly10, p.anatomical_segments)
    keys = ("crossing", "crossing_type", "from_lobe", "to_lobe", "length_mm", "n_groups", "groups",
            "n_piercings", "angle_median_deg", "fissures")
    if p.debug:
        io.write_json(p.anatomical_segments_legend, {
            "vessel_type": vessel_type,
            "block_start": L.BLOCK_START[vessel_type],
            "lobe_order": legend["lobe_order"],
            "label_names": {str(v): L.label_name(v, vessel_type) for v in L.all_labels(vessel_type)},
            "crossing_type_names": {str(v): n for v, n in L.CROSSING_TYPE_NAME.items()},
            "groups": {str(g): {"label": v, "name": L.label_name(v, vessel_type)} for g, v in group_labels.items()},
            "crossings": [{k: v[k] for k in keys} for v in vessels],
        })
    if cfg.statistics:
        doc = statistics.write(p, ctx.patient, ctx.structure, vessel_type, vessels)
        logger.info(f"statistics: {doc['n_crossings']} crossing vessels, {doc['total_length_mm']:.1f} mm "
                    f"-> {p.statistics.name}, {p.statistics_csv.name}")
    ctx.timings["anatomy"] = time.time() - t0

    counts = Counter(L.label_name(v, vessel_type) for v in group_labels.values())
    kinds = Counter(v["crossing_type"] for v in vessels)
    logger.info(f"anatomy: {len(vessels)} crossings ({kinds['interlobar']} interlobar, "
                f"{kinds['translobar']} translobar, {kinds['unclassified']} unclassified); "
                f"{counts.get(L.UNCLASSIFIED_NAME, 0)} of {len(group_labels)} groups unclassified")
    return dict(groups_per_label=dict(counts.most_common()),
                crossings=[{k: v[k] for k in ("crossing", "crossing_type", "from_lobe", "to_lobe",
                                              "length_mm", "n_piercings")} for v in vessels])
