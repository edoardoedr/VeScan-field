"""Stage surface_labels: the centerline labels onto the surfaces (VeScan's stage 8 and the
passes after stages 9 and 10).

Surface, in <results>/final_export_<structure>/ (vescan_thalweg.paths.PipelinePaths):
    supporting_files/01_preprocessed.vtk -> branch_labeled_surface.vtk   (whole surface)
Stage clip, after this one, cuts the labelled surface into clipped_surface.vtk.
Three passes, each one if its source file exists:
    branches  from 05_branch_tree.vtk: GroupId, IsBifurcation, Generation, CellId
    lobes     from anatomical_segments.vtk (stage 9's arrays): NumReachableLobes, LobeMask, LobeLabel,
              LobeLabelPoint, Reaches_<LOBE>        (unlabelled cells dropped, islands filled)
    anatomy   from anatomical_segments.vtk: AnatomicalSegment(Point), CrossingId(Point),
              CrossingType(Point)                    (same; crossings protected)
"""
from __future__ import annotations

import logging
import time

from vescan_thalweg.stages.anatomy import labels as L
from vescan_thalweg.stages.anatomy import VESSEL_TYPE
from vescan_thalweg.surface import add_cell_id_array, load_surface, save_surface

from .transfer import transfer

logger = logging.getLogger(__name__)

ANATOMY_ARRAYS = ["AnatomicalSegment", "AnatomicalSegmentPoint", "CrossingId", "CrossingIdPoint",
                  "CrossingType", "CrossingTypePoint"]


def passes(ctx, cfg):
    """(name, source file, arrays, drop+islands, protected values) of the three passes."""
    vessel_type = VESSEL_TYPE[ctx.structure]
    crossing_lobe_label = len(ctx.lobes) + 2           # LobeLabel N+2 on the active lobes
    crossing = L.crossing_values(vessel_type)
    kinds = {L.CROSSING_INTERLOBAR, L.CROSSING_TRANSLOBAR, L.CROSSING_UNCLASSIFIED}
    ids = set(range(1, cfg.max_protected_crossing_id + 1))
    p = ctx.paths
    return [
        ("branches", p.branch_tree, list(cfg.branch_arrays), False, {}),
        ("lobes", p.anatomical_segments,
         ["NumReachableLobes", "LobeMask", "LobeLabel", "LobeLabelPoint"] + [f"Reaches_{l}" for l in ctx.lobes],
         True, {"LobeLabel": {crossing_lobe_label}, "LobeLabelPoint": {crossing_lobe_label}}),
        ("anatomy", p.anatomical_segments, ANATOMY_ARRAYS, True,
         {"AnatomicalSegment": crossing, "AnatomicalSegmentPoint": crossing, "CrossingType": kinds,
          "CrossingTypePoint": kinds, "CrossingId": ids, "CrossingIdPoint": ids}),
    ]


def run(ctx, cfg) -> dict:
    """cfg: vescan_thalweg.config.SurfaceLabelsConfig."""
    p = ctx.paths
    if not p.preprocessed.exists():
        logger.warning(f"surface_labels skipped: missing {p.preprocessed.name} (stage surfaces is needed)")
        return dict(skipped=True)
    targets = [(p.preprocessed, p.branch_labeled_surface)]
    t0 = time.time()
    centerlines = {}
    done = {}
    for name, path, arrays, cleanup, protected in passes(ctx, cfg):
        if not path.exists():
            continue
        centerlines[name] = (load_surface(path), arrays, cleanup, protected)
        if "CellId" in arrays and centerlines[name][0].GetCellData().GetArray("CellId") is None:
            add_cell_id_array(centerlines[name][0])
    for src, dst in targets:
        surface = load_surface(src)
        for name, (centerline, arrays, cleanup, protected) in centerlines.items():
            surface = transfer(surface, centerline, arrays, drop_unlabeled=cleanup, fill=cleanup,
                               max_island_area_fraction=cfg.max_island_area_fraction,
                               max_island_passes=cfg.max_island_passes, protected=protected)
        save_surface(surface, dst)
        done[dst.name] = list(centerlines)
    ctx.timings["surface_labels"] = time.time() - t0
    summary = ", ".join(f"{k} ({' + '.join(v)})" for k, v in done.items())
    logger.info(f"labels onto the surfaces: {summary}")
    return dict(surfaces=done)
