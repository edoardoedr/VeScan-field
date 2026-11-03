"""Stage surfaces: surfaces from the masks, for visualisation and for the later stages.

    convert        mask -> closed surface (as Slicer)
    preprocess     largest component, decimation to N points, outward normals
    lobe_segments  one lobe mesh labelled per vessel type, fissures, legend

run_patient (once per patient) -> the surfaces folder (batch.surfaces_root_dir[/<patient>],
as VeScan's converted surfaces):
    lung_*_lobe_*.vtk, lobe_segments.vtk, lobe_segments.json, lobe_fissures.vtk
run (per structure):
    <surfaces>/<first class>.vtk                    (e.g. lung_arteries.vtk, lung_airways.vtk)
    <results>/final_export_<s>/supporting_files/01_preprocessed.vtk
Paths from vescan_field.paths (SurfacePaths, PipelinePaths).
The lobes are not decimated: fissures are found from the coincident triangles of adjacent lobes.
"""
from __future__ import annotations

import logging
import time

from vescan_field import io
from vescan_field.field import load_reference
from vescan_field.surface import save_surface

from . import lobe_segments
from .convert import convert_mask
from .preprocess import preprocess

logger = logging.getLogger(__name__)


def _convert(mask, geom, ccfg):
    return convert_mask(mask, geom, method=ccfg.method, smoothing_factor=ccfg.smoothing_factor,
                        decimation_factor=ccfg.decimation_factor, compute_normals=ccfg.compute_normals)


def run_patient(pctx, cfg) -> dict:
    """Lobes: surfaces and (if enabled) the labelled mesh with the fissures."""
    sp = pctx.surfaces
    sp.surfaces_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lobes, missing = {}, []
    for lobe, mask_name in pctx.lobes.items():
        path = pctx.patient_dir / f"{mask_name}.nii.gz"
        if not path.exists():
            missing.append(mask_name)
            continue
        geom, mask = load_reference(path)
        poly = _convert(mask, geom, cfg.convert)
        if poly is None:
            missing.append(mask_name)
            continue
        save_surface(poly, sp.surface(mask_name))
        lobes[lobe] = poly
    if missing:
        logger.warning(f"missing or empty lobes: {missing}")
    pctx.timings["surfaces_convert_lobes"] = time.time() - t0
    summary = dict(lobes={lobe: str(sp.surface(pctx.lobes[lobe])) for lobe in lobes}, missing=missing)

    lcfg = cfg.lobe_segments
    if lcfg.enabled and lobes:
        t0 = time.time()
        combined, fissures = lobe_segments.build_combined_lobes(
            lobes, extract_fissures_too=lcfg.extract_fissures, tolerance=lcfg.fissure_contact_tolerance,
            min_component_area=lcfg.min_fissure_component_area, min_fissure_area=lcfg.min_fissure_area)
        save_surface(combined, sp.segments)
        sheets = lobe_segments.build_fissure_surface(fissures)
        if sheets is not None:
            save_surface(sheets, sp.fissures)
        legend = lobe_segments.legend(fissures)
        io.write_json(sp.segments_legend, legend)
        pctx.timings["surfaces_lobe_segments"] = time.time() - t0
        summary["fissures"] = {f["contact_name"]: f["area_mm2"] for f in fissures}
    logger.info(f"lobe surfaces: {len(lobes)}, fissures: {len(summary.get('fissures', {}))} -> {sp.surfaces_dir}/")
    return summary


def run(ctx, cfg) -> dict:
    """Vessel: surface from the structure's mask and 01_preprocessed.vtk."""
    t0 = time.time()
    poly = _convert(ctx.mask, ctx.ct_geom, cfg.convert)
    if poly is None:
        raise ValueError(f"empty mask for {ctx.structure} ({ctx.classes})")
    raw_path = ctx.surfaces.surface(ctx.classes[0])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    save_surface(poly, raw_path)
    ctx.timings["surfaces_convert"] = time.time() - t0
    summary = dict(surface=str(raw_path), surface_points=poly.GetNumberOfPoints())

    pcfg = cfg.preprocess
    if pcfg.enabled:
        t0 = time.time()
        pre, info = preprocess(poly, target_points=pcfg.target_points, tolerance=pcfg.tolerance,
                               decimate=pcfg.decimate, keep_largest_component=pcfg.keep_largest_component)
        save_surface(pre, ctx.paths.preprocessed)
        ctx.timings["surfaces_preprocess"] = time.time() - t0
        summary.update(preprocessed_points=pre.GetNumberOfPoints(), decimation=info)
    logger.info(f"surface: {summary['surface_points']} points"
                + (f", 01_preprocessed {summary['preprocessed_points']} points" if pcfg.enabled else ""))
    return summary
