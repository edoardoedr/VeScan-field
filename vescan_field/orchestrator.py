"""One patient: patient-level stages, then structures -> field (generic) -> enabled stages.

The loop over patients (one or batch) is in vescan_field/main.py (launched by main_field.py). Each stage in STAGES with
stages.<name> = true can have:
    run_patient(pctx, cfg)   once per patient, before the structures (e.g. the lobes)
    run(ctx, cfg)            for each structure, after the margin from the store
with cfg its pipeline.<name> section. Every output path comes from vescan_field.paths. The
summaries go to <results>/patient_summary.json and <results>/final_export_<structure>/run_summary.json.
"""
from __future__ import annotations

import dataclasses
import importlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vescan_field import io
from vescan_field import paths as pathsmod
from vescan_field.config import RootConfig
from vescan_field.field import Geometry, load_reference, margin_from_store
from vescan_field.lobes import lobe_mask_names
from vescan_field.paths import PipelinePaths, SurfacePaths
from vescan_field.store import Store

logger = logging.getLogger(__name__)

# run order; each name is vescan_field.stages.<name>, stages.<name> and pipeline.<name>
STAGES = ["surfaces", "centerlines", "anatomy", "surface_labels", "clip"]


@dataclass
class PatientContext:
    """What the stages receive at patient level."""
    patient: str
    patient_dir: Path
    results_dir: Path
    surfaces: SurfacePaths               # batch.surfaces_root_dir[/<patient>]
    store: Store
    lobes: dict                          # enabled lobe -> mask name
    timings: dict = field(default_factory=dict)


@dataclass
class StructureContext:
    """What the stages receive for one structure of a patient."""
    patient: str
    structure: str
    classes: list
    store: Store
    patient_dir: Path
    results_dir: Path                    # <results>/
    surfaces: SurfacePaths               # patient-level surfaces (lobes, fissures, raw surfaces)
    paths: PipelinePaths                 # <results>/final_export_<structure>/
    mask: np.ndarray                     # union of the structure's masks, full CT (z,y,x)
    ct_geom: Geometry                    # CT geometry
    lobes: dict                          # enabled lobe -> mask name
    margin: np.ndarray                   # margin on the CT crop (float32, z,y,x)
    geom: Geometry                       # crop geometry
    keep_largest_component: bool = True  # surfaces.preprocess.keep_largest_component (also the centerlines)
    qa: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


def _enabled_stages(cfg: RootConfig):
    for name in STAGES:
        if getattr(cfg.pipeline.stages, name):
            yield name, importlib.import_module(f"vescan_field.stages.{name}"), getattr(cfg.pipeline, name)


def run_patient(cfg: RootConfig, patient: str, patient_dir: Path, surfaces_dir: Path, results_dir: Path,
                structures: dict):
    """Patient-level stages, then every structure (`structures`: structure -> classes)."""
    store = Store(patient_dir / cfg.batch.store)
    try:
        lcfg = cfg.pipeline.lobes
        surfaces = SurfacePaths(surfaces_dir)
        pctx = PatientContext(patient=patient, patient_dir=patient_dir, results_dir=results_dir,
                              surfaces=surfaces, store=store, lobes=lobe_mask_names(lcfg.enabled, lcfg.names))
        results_dir.mkdir(parents=True, exist_ok=True)
        summary = dict(pipeline="field", patient=patient, patient_dir=str(patient_dir),
                       surfaces_dir=str(surfaces_dir))
        for name, stage, scfg in _enabled_stages(cfg):
            if hasattr(stage, "run_patient"):
                logger.info(f"[{patient}] {name}")
                summary[name] = stage.run_patient(pctx, scfg)
        summary["timings_s"] = pctx.timings
        io.write_json(pathsmod.patient_summary(results_dir), summary)

        for structure, classes in structures.items():
            run_structure(cfg, patient, patient_dir, surfaces, results_dir, store, structure, classes)
    finally:
        store.close()


def run_structure(cfg: RootConfig, patient: str, patient_dir: Path, surfaces: SurfacePaths,
                  results_dir: Path, store: Store, structure: str, classes: list):
    """Field, then the enabled stages, then the summary."""
    logger.info(f"[{patient} / {structure}] classes {[store.class_index[c] for c in classes]} "
                f"({' + '.join(classes)})")
    t_all = time.time()
    ctx = build_context(cfg, patient, patient_dir, surfaces, results_dir, store, structure, classes)

    summary = dict(pipeline="field", patient=patient, structure=structure, store_classes=classes,
                   store=str(store.path), field=dataclasses.asdict(cfg.pipeline.field),
                   qa_mask=ctx.qa)
    for name, stage, scfg in _enabled_stages(cfg):
        if hasattr(stage, "run"):
            summary[name] = stage.run(ctx, scfg)

    ctx.timings["total"] = time.time() - t_all
    summary["timings_s"] = ctx.timings
    io.write_json(ctx.paths.summary, summary)
    logger.info(f"{ctx.timings['total']:.1f} s -> {ctx.paths.output_dir}/")


def build_context(cfg: RootConfig, patient: str, patient_dir: Path, surfaces: SurfacePaths,
                  results_dir: Path, store: Store, structure: str, classes: list) -> StructureContext:
    """Generic part: masks (CT geometry and QA), margin on the CT crop (and, with
    pipeline.debug_files, field_margin.nii.gz)."""
    paths = PipelinePaths(pathsmod.final_export_dir(results_dir, structure), debug=cfg.pipeline.debug_files)
    paths.make_dirs()
    timings = {}

    t0 = time.time()
    ct_geom, mask = load_reference(patient_dir / f"{classes[0]}.nii.gz")
    for c in classes[1:]:
        mask = mask | load_reference(patient_dir / f"{c}.nii.gz")[1]
    timings["load_reference"] = time.time() - t0

    fcfg = cfg.pipeline.field
    margin, g, qa, t = margin_from_store(store, [store.class_index[c] for c in classes], ct_geom,
                                         mask.shape, mask=mask, pad_voxels=fcfg.pad_voxels)
    timings.update(t)
    if paths.debug:
        io.write_margin_image(paths.margin, margin, g)

    return StructureContext(patient=patient, structure=structure, classes=classes, store=store,
                            patient_dir=patient_dir, results_dir=results_dir, surfaces=surfaces,
                            paths=paths, mask=mask, ct_geom=ct_geom,
                            lobes=lobe_mask_names(cfg.pipeline.lobes.enabled, cfg.pipeline.lobes.names),
                            margin=margin, geom=g, qa=qa, timings=timings,
                            keep_largest_component=cfg.pipeline.surfaces.preprocess.keep_largest_component)
