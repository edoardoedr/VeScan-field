"""Stage anatomy on a synthetic tree with 5 box lobes and a RUL|RML fissure.

LPS: right = negative x. Trunk (0,0,0)->(0,0,20) outside the lobes; bifurcation; main right
towards RUL, which splits into a branch staying in RUL and one going down into RML across the
fissure z = 10; main left towards LUL."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import vtk

from vescan_thalweg import surface
from vescan_thalweg.config import AnatomyConfig
from vescan_thalweg.lobes import lobe_mask_names
from vescan_thalweg.paths import PipelinePaths, SurfacePaths
from vescan_thalweg.stages import anatomy
from vescan_thalweg.stages.anatomy import statistics
from vescan_thalweg.stages.centerlines import export_tree

BOXES = {"RUL": (-60, -10, 10, 60), "RML": (-60, -10, -40, 10), "RLL": (-60, -10, -90, -40),
         "LUL": (10, 60, 0, 60), "LLL": (10, 60, -90, 0)}     # (x0, x1, z0, z1), y in [-30, 30]


def box(x0, x1, z0, z1):
    src = vtk.vtkCubeSource()
    src.SetBounds(x0, x1, -30, 30, z0, z1)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(src.GetOutputPort())
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.Update()
    return clean.GetOutput()


def fissure_sheet():
    """RUL|RML sheet at z = 10, dense (a local plane needs >= 12 points within 5 mm)."""
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(-60, -30, 10)
    plane.SetPoint1(-10, -30, 10)
    plane.SetPoint2(-60, 30, 10)
    plane.SetResolution(50, 60)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(plane.GetOutputPort())
    tri.Update()
    sheet = tri.GetOutput()
    n = sheet.GetNumberOfCells()
    surface.add_cell_array(sheet, "FissureContact", np.full(n, 60, np.int32))
    surface.add_cell_array(sheet, "Fissure", np.full(n, 70, np.int32))
    return sheet


def segment(a, b, step=1.0):
    a, b = np.array(a, float), np.array(b, float)
    n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1) + 1
    return np.linspace(a, b, n)


def group(gid, blanked, cells):
    cells = [(p, np.full(len(p), 2.0)) for p in cells]
    lengths = [float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p, _ in cells]
    p = cells[int(np.argmax(lengths))][0]
    return dict(group_id=gid, cell_id=-1, n_candidate_cells=len(cells), blanked=blanked,
                positions=[tuple(x) for x in p], radii=[2.0] * len(p), length=max(lengths),
                avg_radius=2.0, generation=-1, cells=cells)


def synthetic_tree():
    j0, jr = (0, 0, 20), (-30, 0, 20)
    groups = {
        0: group(0, False, [segment((0, 0, 0), j0)]),                          # trunk
        1: group(1, True, [segment(j0, (-1, 0, 20)), segment(j0, (1, 0, 20))]),
        2: group(2, False, [segment((-1, 0, 20), jr)]),                        # main right -> RUL
        3: group(3, True, [segment(jr, (-31, 0, 21)), segment(jr, (-30, 0, 19))]),
        4: group(4, False, [segment((-31, 0, 21), (-50, 0, 40))]),             # stays in RUL
        5: group(5, False, [segment((-30, 0, 19), (-30, 0, -20))]),            # RUL -> RML
        7: group(7, False, [segment((1, 0, 20), (40, 0, 20))]),                # main left -> LUL
    }
    adjacency = {0: dict(upstream=[], downstream=[1]), 1: dict(upstream=[0], downstream=[2, 7]),
                 2: dict(upstream=[1], downstream=[3]), 3: dict(upstream=[2], downstream=[4, 5]),
                 4: dict(upstream=[3], downstream=[]), 5: dict(upstream=[3], downstream=[]),
                 7: dict(upstream=[1], downstream=[])}
    return groups, adjacency, [0]


def run_anatomy(tmp_path, lobes=None, debug=True):
    """Stage anatomy on the synthetic tree; with debug (the default here) the 09_/10_ legends,
    which the tests read, are written too."""
    p = PipelinePaths(tmp_path / "final_export_artery", debug=debug)
    export_tree.write_tree_files(p, *synthetic_tree())
    sp = SurfacePaths(tmp_path / "surfaces")
    sp.surfaces_dir.mkdir(exist_ok=True)
    names = lobe_mask_names(lobes)
    for lobe, name in names.items():
        surface.save_surface(box(*BOXES[lobe]), sp.surface(name))
    surface.save_surface(fissure_sheet(), sp.fissures)
    ctx = SimpleNamespace(paths=p, surfaces=sp, results_dir=tmp_path, lobes=names, patient="P1",
                          structure="artery", timings={})
    anatomy.run(ctx, AnatomyConfig())
    if not debug:
        return None, None, p
    return (json.loads(p.anatomical_segments_legend.read_text()),
            json.loads(p.lobe_reachability_legend.read_text()), p)


def test_debug_files_off(tmp_path):
    """Without pipeline.debug_files: the files the later stages read, no debug folder."""
    _, _, p = run_anatomy(tmp_path, debug=False)
    assert p.anatomical_segments.exists() and p.statistics.exists() and p.branch_tree_topology.exists()
    assert not p.debug_dir.exists()


def test_full_lobes(tmp_path):
    seg, reach, p = run_anatomy(tmp_path)
    labels = {int(g): v["label"] for g, v in seg["groups"].items()}
    assert labels[0] == 10                      # pulmonary trunk
    assert labels[2] == 13 and labels[4] == 13  # inside RUL: right upper lobar artery
    assert labels[7] == 17                      # left upper lobar artery
    assert labels[5] == 76                      # translobar arterial crossing
    assert reach["groups"]["0"]["reachable_lobes"] == ["LUL", "RML", "RUL"]
    (c,) = seg["crossings"]
    assert (c["crossing_type"], c["from_lobe"], c["to_lobe"], c["n_piercings"]) == ("translobar", "RUL", "RML", 1)
    # per point: the main right is "right pulmonary artery" until it enters RUL
    poly = surface.load_surface(p.anatomical_segments)
    point_seg = surface.point_array(poly, "AnatomicalSegmentPoint")
    first_cell_of_2 = list(surface.cell_array(poly, "GroupId")).index(2)
    ids = surface.cell_point_ids(poly)[first_cell_of_2]
    assert point_seg[ids[0]] == 11 and point_seg[ids[-1]] == 13
    # stage 11: one row per crossing vessel, the same vessel as in 10_
    stats = json.loads(p.statistics.read_text())
    assert (stats["patient"], stats["n_crossings"], stats["n_translobar"]) == ("P1", 1, 1)
    rows = list(csv.DictReader(p.statistics_csv.open()))
    assert len(rows) == 1 and rows[0]["crossing_type"] == "translobar" and rows[0]["from_lobe"] == "RUL"


def test_statistics_without_crossings(tmp_path):
    """No crossing: still one CSV row, with n_crossings = 0 and empty crossing columns."""
    doc = statistics.document("P1", "veins", "vein", [])
    statistics.save_csv(doc, tmp_path / "s.csv")
    (row,) = csv.DictReader((tmp_path / "s.csv").open())
    assert row["n_crossings"] == "0" and row["crossing"] == "" and doc["total_length_mm"] == 0.0


def test_left_lung_left_out(tmp_path):
    """Without the left lobes the main left reaches no lobe: it is a no-lobe region (30 mm,
    substantial) placed on the left, and the trunk stays trunk."""
    seg, reach, _ = run_anatomy(tmp_path, lobes=["RUL", "RML", "RLL"])
    labels = {int(g): v["label"] for g, v in seg["groups"].items()}
    assert labels[7] == 12 and labels[0] == 10
    assert reach["lobe_order"] == ["RUL", "RML", "RLL"]
