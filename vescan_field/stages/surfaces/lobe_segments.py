"""One lobe mesh, labelled per vessel type, and the interlobar fissures
(from vescan/stages/build_lobe_segments.py: same values and legend; the names are
in English, as ColorTableLobesFissures_EN.csv).

Cell arrays on lobe_segments.vtk:
  LobeSegment_Vein   30 RUL, 31 RML, 32 RLL, 33 LLL, 34 LUL
  LobeSegment_Artery 40-44, LobeSegment_Airway 50-54 (same order)
  FissureContact     60 RUL|RML, 61 RUL|RLL, 62 RML|RLL, 63 LUL|LLL (0 = not a fissure)
  Fissure            70 right horizontal, 71 right oblique (61+62), 72 left oblique
In Slicer a lobe is hidden with a threshold (Models > Display > Scalars > Threshold).

Fissures: adjacent lobes of the same label map give COINCIDENT triangles on their border, so
the fissure is the set of a lobe's cells closer than a tolerance to the neighbouring lobe
(selection on a plateau: the tolerance is a guard, not a parameter to tune). The sheet is
curved: the least-squares plane in the legend is only a descriptor (RMS 1.4-5.8 mm).
It is the segmentation's fissure: where the anatomical one is incomplete, TotalSegmentator
closes it anyway. lobe_fissures.vtk has each sheet once (side A).
"""
from __future__ import annotations

import logging

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from vescan_field import threads
from vescan_field.lobes import LOBE_ORDER
from vescan_field.surface import add_cell_array, cell_areas, close_if_open, extract_cells, triangles

logger = logging.getLogger(__name__)

ARRAY_NAME = {"vein": "LobeSegment_Vein", "artery": "LobeSegment_Artery", "airway": "LobeSegment_Airway"}
BLOCK_START = {"vein": 30, "artery": 40, "airway": 50}
# names as in ColorTableLobesFissures_EN.csv
VESSEL_TYPE_LABEL = {"vein": "vein", "artery": "artery", "airway": "bronchus"}
LOBE_LABEL = {
    "RUL": "Right upper lobe", "RML": "Right middle lobe", "RLL": "Right lower lobe",
    "LLL": "Left lower lobe", "LUL": "Left upper lobe",
}

FISSURE_CONTACT_ARRAY = "FissureContact"
FISSURE_ARRAY = "Fissure"
# the 4 lobe pairs that touch (the others measure 0 mm2: the two lungs do not touch)
# (lobe A, lobe B, contact value, contact name, fissure value)
FISSURE_CONTACTS = [
    ("RUL", "RML", 60, "Right horizontal fissure", 70),
    ("RUL", "RLL", 61, "Right oblique fissure - upper part", 71),
    ("RML", "RLL", 62, "Right oblique fissure - lower part", 71),
    ("LUL", "LLL", 63, "Left oblique fissure", 72),
]
FISSURE_LABEL = {70: "Right horizontal fissure", 71: "Right oblique fissure",
                 72: "Left oblique fissure"}


def label_value(vessel_type, lobe):
    return BLOCK_START[vessel_type] + LOBE_ORDER.index(lobe)


def label_name(vessel_type, lobe):
    return f"{LOBE_LABEL[lobe]} {VESSEL_TYPE_LABEL[vessel_type]} segment"


def contact_cell_mask(surface_a, surface_b, tolerance):
    """Cells of A with all 3 vertices within `tolerance` of B's vertices.

    VeScan uses vtkDistancePolyDataFilter's point-to-surface distance, which in the vmtk-build
    env (VTK 9.5.2) segfaults intermittently, also when re-reading the lobes from file as
    VeScan does. Here: distance to the nearest vertex (cKDTree). On the fissure the vertices of
    the two lobes coincide (median 0.0003 mm on PD-1-Lung-00043), and at 1 mm the selection
    differs from VTK's by 0.05-0.1%, only at the rim of the sheet."""
    from scipy.spatial import cKDTree
    pa = vtk_to_numpy(surface_a.GetPoints().GetData())
    pb = vtk_to_numpy(surface_b.GetPoints().GetData())
    d = cKDTree(pb).query(pa, workers=threads.KDTREE_WORKERS)[0]
    return (d < tolerance)[triangles(surface_a)].all(axis=1)


def drop_small_components(patch, min_area):
    """Drop the connected components below min_area. Returns (patch, kept, dropped)."""
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(patch)
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()
    coloured = conn.GetOutput()
    # RegionId is on the points: each cell takes the region of its first vertex
    region = vtk_to_numpy(coloured.GetPointData().GetArray("RegionId"))[triangles(coloured)[:, 0]]
    area = np.bincount(region, weights=cell_areas(coloured), minlength=conn.GetNumberOfExtractedRegions())
    kept = np.flatnonzero(area >= min_area)
    return extract_cells(coloured, np.isin(region, kept)), kept.size, int(area.size - kept.size)


def fit_plane(poly):
    """Least-squares plane: (centroid, normal, RMS, maximum deviation), in mm."""
    pts = vtk_to_numpy(poly.GetPoints().GetData())
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    dev = (pts - c) @ vt[2]
    return c, vt[2], float(np.sqrt((dev ** 2).mean())), float(np.abs(dev).max())


def extract_fissures(lobes, tolerance=1.0, min_component_area=50.0, min_fissure_area=100.0):
    """Fissures between the lobes {name: polydata}. Returns (fissures, cell_labels): one entry
    per contact above min_fissure_area, and for each lobe the FissureContact/Fissure cell arrays
    (both sides of the contact; where two fissures meet the first in the list wins)."""
    cell_labels = {name: {FISSURE_CONTACT_ARRAY: np.zeros(p.GetNumberOfCells(), np.int32),
                          FISSURE_ARRAY: np.zeros(p.GetNumberOfCells(), np.int32)}
                   for name, p in lobes.items()}
    fissures = []
    for a, b, contact_value, contact_name, fissure_value in FISSURE_CONTACTS:
        if a not in lobes or b not in lobes:
            continue
        mask_a = contact_cell_mask(lobes[a], lobes[b], tolerance)
        mask_b = contact_cell_mask(lobes[b], lobes[a], tolerance)
        if not mask_a.any() or not mask_b.any():
            logger.info(f"{contact_name} ({a}|{b}): no contact")
            continue
        # the sheet from side A only: B's coincides with it
        patch, n_kept, n_dropped = drop_small_components(extract_cells(lobes[a], mask_a), min_component_area)
        area = float(cell_areas(patch).sum()) if patch.GetNumberOfCells() else 0.0
        if area < min_fissure_area:
            logger.info(f"{contact_name} ({a}|{b}): {area:.1f} mm2 < {min_fissure_area} mm2, absent")
            continue
        for lobe, mask in ((a, mask_a), (b, mask_b)):
            free = mask & (cell_labels[lobe][FISSURE_CONTACT_ARRAY] == 0)
            cell_labels[lobe][FISSURE_CONTACT_ARRAY][free] = contact_value
            cell_labels[lobe][FISSURE_ARRAY][free] = fissure_value
        centroid, normal, rms, max_dev = fit_plane(patch)
        add_cell_array(patch, FISSURE_CONTACT_ARRAY, np.full(patch.GetNumberOfCells(), contact_value, np.int32))
        add_cell_array(patch, FISSURE_ARRAY, np.full(patch.GetNumberOfCells(), fissure_value, np.int32))
        fissures.append(dict(
            contact_value=contact_value, contact_name=contact_name, fissure_value=fissure_value,
            fissure_name=FISSURE_LABEL[fissure_value], lobe_a=a, lobe_b=b, polydata=patch,
            area_mm2=round(area, 1), n_components=int(n_kept),
            centroid=[round(float(x), 3) for x in centroid],
            plane_normal=[round(float(x), 4) for x in normal],
            plane_rms_mm=round(rms, 3), plane_max_deviation_mm=round(max_dev, 3),
            n_points=patch.GetNumberOfPoints(), n_cells=patch.GetNumberOfCells()))
        logger.info(f"{contact_name:45s} {a}|{b}  {area:8.1f} mm2, plane RMS {rms:.2f} mm"
                    f"{f', {n_dropped} small components dropped' if n_dropped else ''}")
    return fissures, cell_labels


def build_combined_lobes(lobes, extract_fissures_too=True, tolerance=1.0, min_component_area=50.0,
                         min_fissure_area=100.0):
    """lobes: {name: polydata} (subset of LOBE_ORDER). Returns (combined mesh, fissures)."""
    unknown = [n for n in lobes if n not in LOBE_ORDER]
    if unknown or not lobes:
        raise ValueError(f"lobes {list(lobes)}: expected a non-empty subset of {LOBE_ORDER}")
    lobes = {n: close_if_open(p, f"lobe {n}") for n, p in lobes.items()}
    fissures, cell_labels = (extract_fissures(lobes, tolerance, min_component_area, min_fissure_area)
                             if extract_fissures_too else ([], {}))
    append = vtk.vtkAppendPolyData()
    for lobe in LOBE_ORDER:
        if lobe not in lobes:
            continue
        poly = lobes[lobe]
        n = poly.GetNumberOfCells()
        for vessel_type, name in ARRAY_NAME.items():
            add_cell_array(poly, name, np.full(n, label_value(vessel_type, lobe), np.int32))
        for name, values in cell_labels.get(lobe, {}).items():
            add_cell_array(poly, name, values)
        append.AddInputData(poly)
    append.Update()
    return append.GetOutput(), fissures


def build_fissure_surface(fissures):
    """The fissure sheets in one polydata (one copy each), or None."""
    if not fissures:
        return None
    append = vtk.vtkAppendPolyData()
    for f in fissures:
        append.AddInputData(f["polydata"])
    append.Update()
    return append.GetOutput()


def legend(fissures=None):
    """JSON legend: labels of each array and, if any, the fissures with area and plane."""
    data = {"lobe_order": LOBE_ORDER,
            "arrays": {ARRAY_NAME[t]: {"vessel_type": t,
                                       "labels": {str(label_value(t, lobe)): label_name(t, lobe)
                                                  for lobe in LOBE_ORDER}}
                       for t in ARRAY_NAME}}
    if not fissures:
        return data
    data["arrays"][FISSURE_CONTACT_ARRAY] = {
        "description": "Contact surface between two lobes (4 contacts)",
        "labels": {str(f["contact_value"]): f"{f['contact_name']} ({f['lobe_a']}|{f['lobe_b']})"
                   for f in fissures}}
    data["arrays"][FISSURE_ARRAY] = {
        "description": "Anatomical fissure (3 fissures; the right oblique one joins contacts 61 and 62)",
        "labels": {str(v): n for v, n in FISSURE_LABEL.items()
                   if any(f["fissure_value"] == v for f in fissures)}}
    data["fissures"] = [{k: v for k, v in f.items() if k != "polydata"} for f in fissures]
    return data
