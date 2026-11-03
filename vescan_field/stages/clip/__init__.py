"""Stage clip: cut the centerline after N bifurcations, then the surface (VeScan's stages 6-7).
The last stage: it cuts the surface already labelled by surface_labels, whose labels it keeps.

    cut           cut centerline and cut points
    clip_surface  surface clipped on the border between kept and dropped centerline, with caps

Inputs in <results>/final_export_<structure>/: supporting_files/05_branch_tree.vtk and topology
(stage centerlines); the surface, branch_labeled_surface.vtk (stage surface_labels) or, without
it, supporting_files/01_preprocessed.vtk (stage surfaces). Outputs (vescan_field.paths.PipelinePaths):
    cut_centerline.vtk                      the cells of the kept groups, with every array
    supporting_files/06_cut_points.mrk.json one Slicer point per dropped branch
    clipped_surface.vtk                     the clipped surface, with the labels of its source
No VMTK: the cap is rewritten (vescan_field.surface.cap_holes), no flow extensions.
"""
from __future__ import annotations

import json
import logging
import time

from vescan_field import io
from vescan_field.group_tree import GroupTree
from vescan_field.surface import extract_lines, load_surface, save_surface

from .clip_surface import clip_surface
from .cut import cut_centerline, groups_outside

logger = logging.getLogger(__name__)


def run(ctx, cfg) -> dict:
    """cfg: vescan_field.config.ClipConfig."""
    p = ctx.paths
    source = p.branch_labeled_surface if p.branch_labeled_surface.exists() else p.preprocessed
    inputs = [p.branch_tree, p.branch_tree_topology, source]
    restricted = cfg.generation_lobes is not None and set(cfg.generation_lobes) != set(ctx.lobes)
    if restricted:
        inputs.append(p.anatomical_segments)               # its Reaches_<LOBE> arrays say where each group leads
    missing = [p.name for p in inputs if not p.exists()]
    if missing:
        logger.warning(f"clip skipped: missing {missing} (stages centerlines and surfaces are needed)")
        return dict(skipped=missing)
    if source == p.preprocessed:
        logger.warning(f"clip: no {p.branch_labeled_surface.name} (stage surface_labels), "
                       f"clipping the unlabelled {p.preprocessed.name}")
    t0 = time.time()
    branch_tree, tree = load_surface(inputs[0]), GroupTree.load(inputs[1])

    exempt = groups_outside(load_surface(p.anatomical_segments), ctx.lobes, cfg.generation_lobes) if restricted else set()
    cut, kept_cells, dropped_cells, cuts = cut_centerline(branch_tree, tree, cfg.max_generations, exempt)
    save_surface(cut, p.cut_centerline)
    markups = io.fiducial_markups("cut_points",
                                  [f"cut_group{c['parent_group_id']}_dropped_group{c['dropped_group_id']}"
                                   for c in cuts], [c["position"] for c in cuts])
    p.cut_points.write_text(json.dumps(markups, indent=2))

    surface = load_surface(inputs[2])
    clipped = clip_surface(surface, cut, extract_lines(branch_tree, dropped_cells), cap=cfg.cap)
    save_surface(clipped, p.clipped_surface)
    ctx.timings["clip"] = time.time() - t0
    logger.info(f"clip: generation <= {cfg.max_generations}"
                f"{f' in lobes {cfg.generation_lobes}' if restricted else ''}, {len(kept_cells)}/{len(kept_cells) + len(dropped_cells)} "
                f"cells kept, {len(cuts)} cuts, surface {surface.GetNumberOfPoints()} -> "
                f"{clipped.GetNumberOfPoints()} points")
    return dict(max_generations=cfg.max_generations, generation_lobes=cfg.generation_lobes, surface=source.name, kept_cells=int(len(kept_cells)),
                dropped_cells=int(len(dropped_cells)), cut_points=len(cuts),
                clipped_points=clipped.GetNumberOfPoints(), clipped_cells=clipped.GetNumberOfCells())
