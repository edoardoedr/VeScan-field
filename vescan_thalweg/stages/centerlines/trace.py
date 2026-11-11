"""The structure's centerline tree from thalweg: tracing, pruning, ridge radius, inlet root and the
model's radius interval. Only calls into thalweg (thalweg.centerlines), on the union margin the
orchestrator decoded (vescan_thalweg.margin)."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def trace(ctx, tcfg):
    """thalweg.TubeGraph of ctx's structure. tcfg: config.TraceConfig. The structure is named after
    its first class (a store name thalweg knows: e.g. "lung_airways" picks the flat-tube settings
    the same way thalweg would)."""
    from thalweg.centerlines import centerline_graph, radius_interval

    name = ctx.classes[0]
    graph = centerline_graph(ctx.store, name, margin=(ctx.margin, ctx.geometry, ctx.ref),
                             connectivity=tcfg.connectivity, scale=tcfg.cover_scale,
                             const=tcfg.cover_constant_mm, eps=tcfg.cost_epsilon_mm,
                             ridge_passes=tcfg.ridge_passes, prune=tcfg.prune, recenter=tcfg.recenter,
                             root=tcfg.root, log=logger.info)
    graph = radius_interval(graph, name, ctx.margin, ctx.geometry, logits=tcfg.radius_interval_logits)
    return graph, name
