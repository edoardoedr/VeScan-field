"""Every output path of a run, defined once (as vescan/paths.py, with the same names and layout).

    SurfacePaths(surfaces_dir)    patient level, in batch.surfaces_root_dir[/<patient>]: the
                                  surfaces converted from the masks (lobes, raw vessel surfaces)
                                  and the lobe segments / fissures (VeScan's LobeOverviewPaths)
    PipelinePaths(output_dir, debug)
                                  per structure, <results>/final_export_<structure>/: the files
                                  important for visualisation directly in output_dir without a
                                  prefix, the files the later stages read in supporting_files/
                                  with their stage number (VeScan's PipelinePaths), the run
                                  summary; with debug (pipeline.debug_files), the files nothing
                                  reads (markups, legends, the field method's own files) in
                                  supporting_files/debug/

The stages read paths from these objects only, so a name or folder changes here alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SurfacePaths:
    surfaces_dir: Path

    segments: Path = field(init=False)
    segments_legend: Path = field(init=False)
    fissures: Path = field(init=False)           # the fissure sheets, one copy each

    def __post_init__(self):
        d = self.surfaces_dir = Path(self.surfaces_dir)
        self.segments = d / "lobe_segments.vtk"
        self.segments_legend = d / "lobe_segments.json"
        self.fissures = d / "lobe_fissures.vtk"

    def surface(self, mask_name: str) -> Path:
        """The surface converted from one mask (a lobe, or a structure's first class)."""
        return self.surfaces_dir / f"{mask_name}.vtk"


@dataclass
class PipelinePaths:
    output_dir: Path
    debug: bool = False                          # write the files of debug_dir

    supporting_files_dir: Path = field(init=False)
    debug_dir: Path = field(init=False)
    preprocessed: Path = field(init=False)
    branch_tree: Path = field(init=False)
    branch_tree_topology: Path = field(init=False)
    cut_centerline: Path = field(init=False)
    clipped_surface: Path = field(init=False)
    cut_points: Path = field(init=False)
    branch_labeled_surface: Path = field(init=False)
    anatomical_segments: Path = field(init=False)
    statistics: Path = field(init=False)
    statistics_csv: Path = field(init=False)
    summary: Path = field(init=False)
    # debug only (pipeline.debug_files): read by no stage
    margin: Path = field(init=False)
    centerlines: Path = field(init=False)
    centerlines_curves: Path = field(init=False)
    segments_csv: Path = field(init=False)
    graph: Path = field(init=False)
    graph_nodes: Path = field(init=False)
    branch_tree_curves: Path = field(init=False)
    bifurcation_points: Path = field(init=False)
    lobe_reachability: Path = field(init=False)
    lobe_reachability_legend: Path = field(init=False)
    anatomical_segments_legend: Path = field(init=False)

    def __post_init__(self):
        d = self.output_dir = Path(self.output_dir)
        s = self.supporting_files_dir = d / "supporting_files"
        g = self.debug_dir = s / "debug"
        self.preprocessed = s / "01_preprocessed.vtk"
        self.branch_tree = s / "05_branch_tree.vtk"
        self.branch_tree_topology = s / "05_branch_tree_topology.json"
        self.cut_centerline = d / "cut_centerline.vtk"
        self.clipped_surface = d / "clipped_surface.vtk"
        self.cut_points = s / "06_cut_points.mrk.json"
        self.branch_labeled_surface = d / "branch_labeled_surface.vtk"
        self.anatomical_segments = d / "anatomical_segments.vtk"
        self.statistics = d / "statistics.json"
        self.statistics_csv = d / "statistics.csv"
        self.summary = d / "run_summary.json"          # not in VeScan: parameters, counts, timings
        self.margin = g / "field_margin.nii.gz"
        self.centerlines = g / "field_centerlines.vtk"
        self.centerlines_curves = g / "field_centerlines.mrk.json"
        self.segments_csv = g / "field_segments.csv"
        self.graph = g / "field_graph.json"
        self.graph_nodes = g / "field_graph_nodes.mrk.json"
        self.branch_tree_curves = g / "05_branch_tree_curves.mrk.json"
        self.bifurcation_points = g / "05_bifurcation_points.mrk.json"
        self.lobe_reachability = g / "09_lobe_reachability.vtk"
        self.lobe_reachability_legend = g / "09_lobe_reachability.json"
        self.anatomical_segments_legend = g / "10_anatomical_segments.json"

    def make_dirs(self):
        self.supporting_files_dir.mkdir(parents=True, exist_ok=True)
        if self.debug:
            self.debug_dir.mkdir(exist_ok=True)


def final_export_dir(results_dir, structure) -> Path:
    """<results>/final_export_<structure>/, as VeScan's batch."""
    return Path(results_dir) / f"final_export_{structure}"


def patient_summary(results_dir) -> Path:
    return Path(results_dir) / "patient_summary.json"
