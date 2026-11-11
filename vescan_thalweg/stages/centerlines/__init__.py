"""Stage centerlines: thalweg's centerline tree, then VeScan's groups and the 05_ files.

    trace           thalweg.centerlines.centerline_graph on the union margin (seed-free tracing,
                    pruning, ridge radius, inlet root) + radius_interval (RadiusLow/High)
    bifurcations    pipeline.centerlines.bifurcations:
      "vmtk"        split_vmtk: vmtk's branch extractor (thalweg's numpy port) -> blanked
                    bifurcation groups -> vescan_vmtk's tree (groups, adjacency, roots)
      "point"       segments_point -> vescan_field's tree: the bifurcation is the junction
    export_tree     05_branch_tree.vtk, 05_branch_tree_topology.json (+ markups with debug)

Outputs in ctx.paths (final_export_<structure>/), in the same format as VeScan vmtk and field, read
by the later stages. With pipeline.debug_files, thalweg's own graph (thalweg_graph.thalweg.json.gz)
and, for "vmtk", the split (thalweg_split.vtp) in supporting_files/debug/.
"""
from __future__ import annotations

import dataclasses
import logging
import time

from .trace import trace

logger = logging.getLogger(__name__)


def run(ctx, cfg) -> dict:
    """cfg: vescan_thalweg.config.CenterlinesConfig. Returns the summary's 'centerlines' block."""
    p = ctx.paths
    t0 = time.time()
    graph, name = trace(ctx, cfg.trace)
    ctx.timings["trace"] = time.time() - t0
    st = graph.structure(name)
    summary = dict(bifurcations=cfg.bifurcations, thalweg_name=name, trace=dataclasses.asdict(cfg.trace),
                   thalweg_parameters=st.parameters, thalweg_statistics=st.statistics,
                   graph=dict(nodes=len(graph.nodes), edges=len(graph.edges),
                              total_length_mm=round(sum(e.length_mm for e in graph.edges), 2)))
    logger.info(f"thalweg: {len(graph.edges)} edges, {summary['graph']['total_length_mm'] / 10:.0f} cm")
    if p.debug:
        p.make_dirs()
        graph.write(p.thalweg_graph)

    if not cfg.write_vescan:
        return summary
    add_virtual_root = ctx.structure == "veins" and cfg.add_virtual_root_for_veins
    t0 = time.time()
    if cfg.bifurcations == "vmtk":
        from . import split_vmtk
        cl = split_vmtk.vmtk_split(graph, name, cfg.vmtk)
        ctx.timings["vmtk_branching"] = time.time() - t0
        if p.debug:
            from thalweg.export import write_vtp_centerlines
            write_vtp_centerlines(cl, p.thalweg_split)
        t0 = time.time()
        summary["vmtk"] = dataclasses.asdict(cfg.vmtk)
        summary["vescan"] = split_vmtk.write_tree_files(
            p, split_vmtk.SplitCenterline(cl), cfg.vmtk, add_virtual_root, cfg.virtual_root_min_length_fraction)
    else:
        from . import export_tree, segments_point, tree
        res, gdict = segments_point.to_segments(graph, name, cfg.point.step_mm)
        groups, adjacency, roots = tree.build_groups(
            res, gdict, halfwidth_mm=cfg.point.bifurcation_halfwidth_mm or cfg.point.step_mm,
            short_segment_divisor=cfg.point.short_segment_divisor)
        summary["point"] = dataclasses.asdict(cfg.point)
        summary["vescan"] = export_tree.write_tree_files(
            p, groups, adjacency, roots, add_virtual_root=add_virtual_root,
            virtual_root_min_length_fraction=cfg.virtual_root_min_length_fraction)
    ctx.timings["export_tree"] = time.time() - t0
    return summary
