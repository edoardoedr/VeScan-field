"""Typed JSON config, on the scheme of vescan/config.py: nested dataclasses, one section per
stage under `pipeline`, a default for every field, `null` = default. An unknown key is an
error (not a warning as in VeScan). Relative paths are resolved against the config file's
folder.

    batch      patients, surfaces, results, store, structures (store classes per structure)
    pipeline   stages (switches) + the generic sections (lobes, field) + one section per
               stage (surfaces, centerlines, ...)
    logging    level and per-patient log file

Adding a stage: a field in StagesToggle, a <Name>Config dataclass and the matching field in
PipelineConfig.
"""
from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from vescan_field.lobes import LOBE_ORDER
from vescan_field.log import LEVELS

IGNORED_KEYS = {"experiment_name", "notes", "description"}


class ConfigError(SystemExit):
    """Malformed config value: a SystemExit, so it exits with a one-line message."""


# ---------------------------------------------------------------------------
# pipeline.* - one section for the generic part and for each stage in vescan_field/stages/
# ---------------------------------------------------------------------------

@dataclass
class StagesToggle:
    surfaces: bool = True
    centerlines: bool = True
    anatomy: bool = True
    surface_labels: bool = True
    clip: bool = True


@dataclass
class LobesConfig:
    """Generic part: active lobes and their mask names (as VeScan's pipeline.lobes)."""
    enabled: List[str] = field(default_factory=lambda: list(LOBE_ORDER))
    names: Dict[str, str] = field(default_factory=dict)   # lobe -> mask name (override)


@dataclass
class ConvertConfig:
    """surfaces.convert: mask -> closed surface, as Slicer (stages.surfaces.convert)."""
    method: str = "flying-edges"         # or "surface-nets"
    smoothing_factor: float = 0.5        # vtkWindowedSincPolyDataFilter, Slicer's formulas
    decimation_factor: float = 0.0       # vtkDecimatePro during conversion (0 = off, Slicer's default)
    compute_normals: bool = True


@dataclass
class PreprocessConfig:
    """surfaces.preprocess: vessel surface -> 01_preprocessed.vtk (stages.surfaces.preprocess)."""
    enabled: bool = True
    decimate: bool = True
    target_points: int = 100000          # final number of points (the bisection finds the aggressiveness); standard of the dataset runs
    tolerance: float = 0.05              # points obtained <= target_points * (1 + tolerance)
    keep_largest_component: bool = True  # also the field centerlines: TEASAR on the largest component only


@dataclass
class LobeSegmentsConfig:
    """surfaces.lobe_segments: labelled lobe mesh and fissures (stages.surfaces.lobe_segments)."""
    enabled: bool = True
    extract_fissures: bool = True
    fissure_contact_tolerance: float = 1.0
    min_fissure_component_area: float = 50.0
    min_fissure_area: float = 100.0


@dataclass
class SurfacesConfig:
    """Stage surfaces (vescan_field.stages.surfaces)."""
    convert: ConvertConfig = field(default_factory=ConvertConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    lobe_segments: LobeSegmentsConfig = field(default_factory=LobeSegmentsConfig)


@dataclass
class FieldConfig:
    """Generic part: margin from the store on the CT grid (vescan_field.field.margin_from_store)."""
    pad_voxels: int = 3


@dataclass
class CenterlinesConfig:
    """Stage centerlines (vescan_field.stages.centerlines): centerlines, graph, VeScan groups."""
    # [B]-[C] extraction (stages.centerlines.extract)
    delta: float = 2.0                   # half-width of the radius interval (logits)
    min_branch_mm: float = 3.0           # TEASAR: shortest terminal branch
    teasar_scale: float = 1.5            # TEASAR: covered tube = scale * D + const
    teasar_const: float = 0.5
    min_voxels: int = 30                 # smaller components are dropped
    step_mm: float = 0.5                 # resampling step of the polylines
    # extraction details (from the draft; defaults are the ones validated on the phantom)
    crop_pad_voxels: int = 4             # crop padding around the structure
    boundary_neighbours: int = 8         # boundary crossings per distance (median over k)
    graph_epsilon: float = 1e-3          # edge weight = L / (D^2 + epsilon)
    smooth_iterations_before: int = 2    # polyline smoothing before the ridge step
    smooth_iterations_after: int = 1     # and after
    ridge_iterations: int = 3            # ridge refinement: search passes
    ridge_grid: int = 5                  # n x n grid in the orthogonal plane
    # graph (stages.centerlines.topology): root at the inflow end
    inflow_root_structures: List[str] = field(default_factory=lambda: ["artery", "airways"])
    inflow_min_radius_fraction: float = 0.5
    inflow_fork_radii: float = 2.0       # fork at the inflow: leaves within this many radii of the root
    # 05_ files for VeScan (stages.centerlines.tree / export_tree)
    write_vescan: bool = True
    bifurcation_halfwidth_mm: Optional[float] = None   # point before/after the junction; None = step_mm
    short_segment_divisor: float = 3.0   # on a very short segment the bifurcation part is at most L/3
    add_virtual_root_for_veins: bool = True
    virtual_root_min_length_fraction: float = 0.1


@dataclass
class AnatomyConfig:
    """Stage anatomy (vescan_field.stages.anatomy): VeScan's stages 9, 10 and 11, same defaults."""
    containment_threshold: float = 0.1           # fraction of a branch's points inside a lobe
    containment_tolerance: float = 1e-4          # vtkSelectEnclosedPoints
    no_lobe_region_min_length_mm: float = 25.0   # "substantial" no-lobe regions
    min_piercings_for_interlobar: int = 2
    interlobar_max_median_angle_deg: float = 30.0
    # piercing geometry (stages.anatomy.crossings.Rules)
    grazing_angle_deg: float = 10.0              # below: "grazing" angle, imprecise
    tangent_window_mm: float = 5.0               # half-window for the tangent
    fissure_patch_radius_mm: float = 5.0         # radius of the local fissure plane
    fissure_patch_min_points: int = 12
    duplicate_piercing_mm: float = 0.1
    # VeScan stage 11 (stages.anatomy.statistics)
    statistics: bool = True                      # statistics.json / .csv in final_export_<structure>/


@dataclass
class ClipConfig:
    """Stage clip (vescan_field.stages.clip): VeScan's stages 6-7, without flow extensions."""
    max_generations: int = 8                      # 0 = trunk only; standard of the dataset runs
    cap: bool = True                              # caps on the holes left by the cut


@dataclass
class SurfaceLabelsConfig:
    """Stage surface_labels (vescan_field.stages.surface_labels): labels onto the surfaces."""
    branch_arrays: List[str] = field(default_factory=lambda: ["GroupId", "IsBifurcation", "Generation", "CellId",
                                                              "Length", "AverageRadius"])  # standard of the dataset runs
    max_island_area_fraction: float = 0.0025      # islands below this fraction of the area are filled
    max_island_passes: int = 5                    # fill passes (while anything changes)
    max_protected_crossing_id: int = 500          # protected CrossingIds 1..N (one per crossing)


@dataclass
class PipelineConfig:
    stages: StagesToggle = field(default_factory=StagesToggle)
    debug_files: bool = False            # also the files no stage reads, in supporting_files/debug/
    lobes: LobesConfig = field(default_factory=LobesConfig)
    surfaces: SurfacesConfig = field(default_factory=SurfacesConfig)
    centerlines: CenterlinesConfig = field(default_factory=CenterlinesConfig)
    anatomy: AnatomyConfig = field(default_factory=AnatomyConfig)
    surface_labels: SurfaceLabelsConfig = field(default_factory=SurfaceLabelsConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    # last: in the class body the name `field` shadows dataclasses.field from here on
    field: FieldConfig = field(default_factory=FieldConfig)


# ---------------------------------------------------------------------------
# batch.* - as VeScan: enabled=false -> patients_root_dir/surfaces_root_dir/results_root_dir
# are the folders of ONE patient; enabled=true -> one subfolder per patient in all three.
# surfaces_root_dir holds the patient-level surfaces (lobes, fissures, raw vessel surfaces),
# results_root_dir the per-structure final_export_<structure>/ and <structure>/ folders.
# ---------------------------------------------------------------------------

@dataclass
class StructureConfig:
    enabled: bool = True
    classes: List[str] = field(default_factory=list)   # store classes (union)


@dataclass
class BatchStructuresConfig:
    artery: StructureConfig = field(default_factory=lambda: StructureConfig(True, ["lung_arteries"]))
    veins: StructureConfig = field(default_factory=lambda: StructureConfig(True, ["lung_veins"]))
    airways: StructureConfig = field(
        default_factory=lambda: StructureConfig(True, ["lung_airways", "lung_airways_wall"]))


@dataclass
class BatchConfig:
    enabled: bool = False
    patients_root_dir: str = ""
    surfaces_root_dir: str = ""
    results_root_dir: str = ""
    store: str = "lung_vessels.duckn.zip"
    structures: BatchStructuresConfig = field(default_factory=BatchStructuresConfig)


@dataclass
class LoggingConfig:
    """Pipeline messages (vescan_field.log): console and, if file_name, one file per patient."""
    level: str = "INFO"                           # DEBUG, INFO, WARNING, ERROR
    file_name: Optional[str] = "pipeline.log"     # in <results>/<patient>/; null = console only


@dataclass
class RootConfig:
    batch: BatchConfig = field(default_factory=BatchConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def enabled_structures(self) -> dict:
        """structure -> store classes, in the order artery, veins, airways."""
        out = {}
        for f in dataclasses.fields(self.batch.structures):
            s = getattr(self.batch.structures, f.name)
            if s.enabled:
                out[f.name] = list(s.classes)
        return out


STRUCTURE_NAMES = [f.name for f in dataclasses.fields(BatchStructuresConfig)]


# ---------------------------------------------------------------------------
# Generic dataclass <- JSON loader (from vescan/config.py)
# ---------------------------------------------------------------------------

def _unwrap_optional(tp):
    """Optional[X] -> (X, True); otherwise (tp, False)."""
    if typing.get_origin(tp) is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _known_paths(cls, prefix: str = "") -> set:
    paths = set()
    for f in dataclasses.fields(cls):
        path = f"{prefix}.{f.name}" if prefix else f.name
        real_type, _optional = _unwrap_optional(typing.get_type_hints(cls)[f.name])
        if dataclasses.is_dataclass(real_type):
            paths |= _known_paths(real_type, path)
        else:
            paths.add(path)
    return paths


def _leaf_paths(data, prefix: str = "") -> set:
    paths = set()
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                paths |= _leaf_paths(value, path)
            else:
                paths.add(path)
    return paths


def _build(cls, data, path: str = ""):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"Config error: '{path or cls.__name__}' must be an object, got {data!r}")

    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data or data[f.name] is None:
            continue  # missing or null -> the field's default
        value = data[f.name]
        field_path = f"{path}.{f.name}" if path else f.name
        real_type, _optional = _unwrap_optional(hints[f.name])

        if dataclasses.is_dataclass(real_type):
            kwargs[f.name] = _build(real_type, value, field_path)
        elif real_type is bool:
            if not isinstance(value, bool):
                raise ConfigError(f"Config error: '{field_path}' must be true or false, got {value!r}")
            kwargs[f.name] = value
        elif real_type in (int, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"Config error: '{field_path}' must be a number, got {value!r}")
            kwargs[f.name] = real_type(value)
        elif real_type is str:
            if not isinstance(value, str):
                raise ConfigError(f"Config error: '{field_path}' must be a string, got {value!r}")
            kwargs[f.name] = value
        elif typing.get_origin(real_type) is list:
            (elem_type,) = typing.get_args(real_type) or (str,)
            if not isinstance(value, list) or not all(isinstance(v, elem_type) for v in value):
                raise ConfigError(f"Config error: '{field_path}' must be a list of "
                                  f"{elem_type.__name__}, got {value!r}")
            kwargs[f.name] = list(value)
        elif typing.get_origin(real_type) is dict:
            key_type, value_type = typing.get_args(real_type) or (str, str)
            if not isinstance(value, dict) or not all(isinstance(k, key_type) and isinstance(v, value_type)
                                                      for k, v in value.items()):
                raise ConfigError(f"Config error: '{field_path}' must be an object mapping "
                                  f"{key_type.__name__} to {value_type.__name__}, got {value!r}")
            kwargs[f.name] = dict(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def build(data: dict, base_dir=None) -> RootConfig:
    """RootConfig from an already-read dict; relative paths resolved against base_dir."""
    known = _known_paths(RootConfig)
    # a dict field (pipeline.lobes.names) is a leaf: its keys are data, not config keys
    unknown = sorted(p for p in _leaf_paths(data) - known - IGNORED_KEYS
                     if not any(p.startswith(f"{leaf}.") for leaf in known))
    if unknown:
        raise ConfigError(f"Config error: unknown key(s) {unknown}")
    cfg = _build(RootConfig, data)
    bad = set(cfg.pipeline.centerlines.inflow_root_structures) - set(STRUCTURE_NAMES)
    if bad:
        raise ConfigError(f"Config error: 'pipeline.centerlines.inflow_root_structures' has unknown "
                          f"structure(s) {sorted(bad)} (known: {STRUCTURE_NAMES})")
    if cfg.logging.level not in LEVELS:
        raise ConfigError(f"Config error: 'logging.level' must be DEBUG, INFO, WARNING or ERROR, "
                          f"got {cfg.logging.level!r}")
    lobes = cfg.pipeline.lobes
    bad = sorted((set(lobes.enabled) | set(lobes.names)) - set(LOBE_ORDER))
    if bad:
        raise ConfigError(f"Config error: 'pipeline.lobes' has unknown lobe(s) {bad} (known: {LOBE_ORDER})")
    if cfg.pipeline.surfaces.convert.method not in ("flying-edges", "surface-nets"):
        raise ConfigError(f"Config error: 'pipeline.surfaces.convert.method' must be 'flying-edges' or "
                          f"'surface-nets', got {cfg.pipeline.surfaces.convert.method!r}")
    if cfg.pipeline.surfaces.preprocess.target_points < 4:
        raise ConfigError("Config error: 'pipeline.surfaces.preprocess.target_points' must be >= 4")
    for name, classes in cfg.enabled_structures().items():
        if not classes:
            raise ConfigError(f"Config error: 'batch.structures.{name}.classes' is empty")
    for key in ("patients_root_dir", "surfaces_root_dir", "results_root_dir"):
        value = getattr(cfg.batch, key)
        if not value:
            raise ConfigError(f"Config error: 'batch.{key}' is required")
        p = Path(value)
        if not p.is_absolute() and base_dir is not None:
            p = (Path(base_dir) / p).resolve()
        setattr(cfg.batch, key, str(p))
    return cfg


def load(path) -> RootConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return build(data, base_dir=path.parent)
