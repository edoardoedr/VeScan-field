"""[VeScan stage 11] Crossing-vessel statistics: statistics.json and statistics.csv
(from vescan/stages/statistics.py: same document, same columns).

Reporting only. The crossing vessels are the ones crossings.analyse found in this anatomy run, so
the `crossing` number here is the CrossingId on anatomical_segments.vtk. This module only writes
them out:
  statistics.json  one document per structure: counts (interlobar / translobar / unclassified),
                   total length and every crossing with its three lengths, lobes, angles and
                   per-piercing detail;
  statistics.csv   one row per crossing vessel, so concatenating the files across patients and
                   structures is the whole aggregation; a structure with no crossing still writes
                   one row with n_crossings = 0, so "none found" stays distinct from "not run".
statistics_scripts/aggregate_statistics.py collects the JSON of many patients into one .xlsx.

Difference from VeScan: patient and structure are passed in. VeScan guesses them from the
path, two levels above final_export_<structure>, which in the results_root_dir/<patient>/ layout
gives the dataset folder's name instead of the patient's.
"""
from __future__ import annotations

import csv
import json

import numpy as np

from vescan_field.stages.surfaces.lobe_segments import FISSURE_LABEL

CSV_COLUMNS = ["patient", "structure", "vessel_type", "n_crossings",
               "n_interlobar", "n_translobar",
               "crossing", "crossing_type", "from_lobe", "to_lobe",
               "length_mm", "length_to_end_mm", "length_after_piercing_mm", "points",
               "n_piercings", "angle_median_deg", "angle_min_deg", "angle_max_deg",
               "fissures", "radius_mm", "patch_rms_max_mm"]

# columns copied from a crossing as they are, None written as an empty cell
_PER_CROSSING = ["crossing", "crossing_type", "length_mm", "length_to_end_mm", "length_after_piercing_mm",
                 "points", "n_piercings", "angle_median_deg", "angle_min_deg", "angle_max_deg",
                 "radius_mm", "patch_rms_max_mm"]


def _count(vessels, kind):
    return sum(1 for v in vessels if v["crossing_type"] == kind)


def document(patient, structure, vessel_type, vessels):
    """The statistics.json document (same keys and order as VeScan's)."""
    return {
        "patient": patient,
        "structure": structure,
        "vessel_type": vessel_type,
        "n_crossings": len(vessels),
        "n_interlobar": _count(vessels, "interlobar"),
        "n_translobar": _count(vessels, "translobar"),
        "n_unclassified": _count(vessels, "unclassified"),
        "total_length_mm": round(sum(v["length_mm"] for v in vessels), 3) if vessels else 0.0,
        "crossings": vessels,
    }


def fissure_names(values):
    """The `fissures` column: the names of the fissures a crossing meets, joined with '+'."""
    return "+".join(FISSURE_LABEL.get(v, str(v)) for v in values or [])


def save_json(doc, path):
    def plain(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(type(o))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False, default=plain)


def save_csv(doc, path):
    """One row per crossing vessel; one row with empty crossing columns when there is none.
    The per-piercing detail stays in the JSON: an interlobar vessel with five piercings must
    stay one row, or counting rows would count it five times."""
    common = {k: doc[k] for k in ("patient", "structure", "vessel_type", "n_crossings",
                                  "n_interlobar", "n_translobar")}
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        if not doc["crossings"]:
            writer.writerow(dict(common, **{c: "" for c in CSV_COLUMNS if c not in common}))
            return
        for c in doc["crossings"]:
            row = {k: "" if c[k] is None else c[k] for k in _PER_CROSSING}
            writer.writerow(dict(common, **row, from_lobe=c["from_lobe"] or "", to_lobe=c["to_lobe"] or "",
                                 fissures=fissure_names(c["fissures"])))


def write(paths, patient, structure, vessel_type, vessels):
    """statistics.json and statistics.csv in final_export_<structure>/; returns the document."""
    doc = document(patient, structure, vessel_type, vessels)
    save_json(doc, paths.statistics)
    save_csv(doc, paths.statistics_csv)
    return doc
