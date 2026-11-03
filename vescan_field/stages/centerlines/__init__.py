"""Stage centerlines: from a structure's margin to the centerlines, the graph and the VeScan files.

    extract     [B]-[C] boundary, distance, field graph, TEASAR, ridge, radius +-delta
    topology    graph oriented from the root (root at the inflow), field_graph.json
    tree        [D] VeScan-style groups: branches + minimal bifurcation
    export      field_centerlines.vtk / .mrk.json, field_segments.csv, field_graph*.json (debug files)
    export_tree [E] the tree's 05_* files (VeScan format) in final_export_<structure>/supporting_files/

Outputs in ctx.paths (final_export_<structure>/); the field method's own files only with
pipeline.debug_files, in supporting_files/debug/.
"""
from __future__ import annotations

import logging
import time

from vescan_field import io

from . import export, topology
from .extract import extract

logger = logging.getLogger(__name__)

EXTRACT_KEYS = ("delta", "min_branch_mm", "teasar_scale", "teasar_const", "min_voxels", "step_mm",
                "crop_pad_voxels", "boundary_neighbours", "graph_epsilon", "smooth_iterations_before",
                "smooth_iterations_after", "ridge_iterations", "ridge_grid")


def run(ctx, cfg) -> dict:
    """cfg: vescan_field.config.CenterlinesConfig. Returns the summary's 'centerlines' block."""
    structure, p = ctx.structure, ctx.paths
    params = {k: getattr(cfg, k) for k in EXTRACT_KEYS}
    # the same components as the surface: surfaces.preprocess.keep_largest_component
    res = extract(ctx.margin, ctx.geom, largest_component_only=ctx.keep_largest_component, **params)
    ctx.timings.update(res["timings"])
    graph = topology.orient(res, inflow=structure in cfg.inflow_root_structures,
                            inflow_min_radius_fraction=cfg.inflow_min_radius_fraction,
                            inflow_fork_radii=cfg.inflow_fork_radii)

    if p.debug:
        t0 = time.time()
        p.make_dirs()
        export.write_vtk_centerlines(p.centerlines, res, structure)
        export.write_markups_curves(p.centerlines_curves, res, structure)
        export.write_segments_csv(p.segments_csv, res, structure)
        io.write_json(p.graph, dict(structure=structure, **graph))
        export.write_graph_points_markups(p.graph_nodes, graph, structure)
        ctx.timings["export"] = time.time() - t0

    vescan_info = None
    if cfg.write_vescan:
        from . import export_tree, tree
        t0 = time.time()
        groups, adjacency, roots = tree.build_groups(
            res, graph, halfwidth_mm=cfg.bifurcation_halfwidth_mm or cfg.step_mm,
            short_segment_divisor=cfg.short_segment_divisor)
        vescan_info = export_tree.write_tree_files(
            p, groups, adjacency, roots,
            add_virtual_root=(structure == "veins" and cfg.add_virtual_root_for_veins),
            virtual_root_min_length_fraction=cfg.virtual_root_min_length_fraction)
        ctx.timings["export_tree"] = time.time() - t0

    logger.info(f"{res['counts']['segments']} segments, {res['counts']['total_length_mm'] / 10:.0f} cm")
    return dict(params=params, largest_component_only=ctx.keep_largest_component, counts=res["counts"], vescan=vescan_info,
                roots_lps=res["geom"].mm_to_lps(res["roots_mm"]))
