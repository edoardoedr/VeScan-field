"""Exports of the centerlines stage: VTK centerlines (LPS), Slicer markups, CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from vescan_field.io import MARKUPS_SCHEMA

from .refine import resample_polyline


def write_vtk_centerlines(path, res, class_name):
    """Legacy polydata in LPS (Slicer header `SPACE=LPS`). Point data: Radius, RadiusLow,
    RadiusHigh; cell data: SegmentId, Length, AverageRadius (names as in 05_branch_tree.vtk),
    and, if the graph was oriented (topology.orient): StartNodeId, EndNodeId,
    ParentSegmentId, Generation, ComponentId."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    g, segs = res["geom"], res["segments"]
    pts = np.concatenate([g.mm_to_lps(s["points_mm"]) for s in segs])
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(pts.astype(np.float64), deep=True))
    lines = vtk.vtkCellArray()
    off = 0
    for s in segs:
        n = len(s["points_mm"])
        lines.InsertNextCell(n)
        for i in range(off, off + n):
            lines.InsertCellPoint(i)
        off += n
    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetLines(lines)

    def add(data, name, values, vtk_type):
        arr = numpy_to_vtk(np.ascontiguousarray(values), deep=True, array_type=vtk_type)
        arr.SetName(name)
        data.AddArray(arr)

    for name, key in [("Radius", "radius"), ("RadiusLow", "radius_lo"), ("RadiusHigh", "radius_hi")]:
        add(pd.GetPointData(), name, np.concatenate([s[key] for s in segs]), vtk.VTK_DOUBLE)
    add(pd.GetCellData(), "SegmentId", np.arange(len(segs), dtype=np.int32), vtk.VTK_INT)
    add(pd.GetCellData(), "Length", np.array([s["length_mm"] for s in segs]), vtk.VTK_DOUBLE)
    add(pd.GetCellData(), "AverageRadius", np.array([np.mean(s["radius"]) for s in segs]),
        vtk.VTK_DOUBLE)
    if segs and "generation" in segs[0]:
        for name, key in [("StartNodeId", "start"), ("EndNodeId", "end"),
                          ("ParentSegmentId", "parent"), ("Generation", "generation"),
                          ("ComponentId", "component")]:
            add(pd.GetCellData(), name, np.array([s[key] for s in segs], np.int32), vtk.VTK_INT)

    w = vtk.vtkPolyDataWriter()
    w.SetFileName(str(path))
    w.SetInputData(pd)
    w.SetHeader(f"field centerlines {class_name}. SPACE=LPS")
    w.Write()


def write_markups_curves(path, res, class_name, step_mm=2.0):
    g = res["geom"]
    markups = []
    for i, s in enumerate(res["segments"]):
        p = g.mm_to_lps(resample_polyline(s["points_mm"], step_mm)
                        if s["length_mm"] > step_mm else s["points_mm"])
        name = f"{class_name}_{i}"
        markups.append({"type": "Curve", "coordinateSystem": "LPS", "coordinateUnits": "mm",
                        "name": name,
                        "controlPoints": [{"id": str(j + 1), "label": f"{name}-{j + 1}",
                                           "position": [float(x) for x in q]}
                                          for j, q in enumerate(p)]})
    doc = {"@schema": MARKUPS_SCHEMA,
           "markups": markups}
    Path(path).write_text(json.dumps(doc, indent=1))


def write_segments_csv(path, res, class_name):
    g = res["geom"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "segment", "n_points", "length_mm", "radius_median_mm",
                    "radius_lo_median_mm", "radius_hi_median_mm",
                    "start_L", "start_P", "start_S", "end_L", "end_P", "end_S",
                    "start_node", "end_node", "parent", "generation", "component"])
        for i, s in enumerate(res["segments"]):
            ends = g.mm_to_lps(s["points_mm"][[0, -1]])
            w.writerow([class_name, i, len(s["points_mm"]), f"{s['length_mm']:.2f}",
                        f"{np.median(s['radius']):.3f}", f"{np.median(s['radius_lo']):.3f}",
                        f"{np.median(s['radius_hi']):.3f}", *np.round(ends.ravel(), 2),
                        *[s.get(k, "") for k in ("start", "end", "parent", "generation", "component")]])


def write_graph_points_markups(path, graph, class_name):
    """Roots and junctions as Slicer points (label: type, id, degree)."""
    pts = [p for p in graph["points"] if p["type"] != "end"]
    doc = {"@schema": MARKUPS_SCHEMA,
           "markups": [{"type": "Fiducial", "coordinateSystem": "LPS", "coordinateUnits": "mm",
                        "name": f"{class_name}_graph_nodes",
                        "controlPoints": [{"id": str(p["id"]),
                                           "label": f"{p['type']}_{p['id']}_d{p['degree']}",
                                           "position": p["position_lps"]} for p in pts]}]}
    Path(path).write_text(json.dumps(doc, indent=1))
