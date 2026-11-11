"""Lung lobes: canonical order, mask names, sides (from vescan/lobes.py).

LOBE_ORDER is the order used everywhere (labels, legends); an active subset is chosen with
pipeline.lobes.enabled, and the mask names can be overridden with pipeline.lobes.names.
"""
from __future__ import annotations

LOBE_ORDER = ["RUL", "RML", "RLL", "LLL", "LUL"]

RIGHT_LOBES = {"RUL", "RML", "RLL"}
LEFT_LOBES = {"LUL", "LLL"}
SIDE_RIGHT, SIDE_LEFT = "right", "left"
BOTH_SIDES = {SIDE_RIGHT, SIDE_LEFT}

# TotalSegmentator mask names (<name>.nii.gz in the patient folder)
LOBE_SEGMENTATION_NAME = {
    "RUL": "lung_upper_lobe_right",
    "RML": "lung_middle_lobe_right",
    "RLL": "lung_lower_lobe_right",
    "LLL": "lung_lower_lobe_left",
    "LUL": "lung_upper_lobe_left",
}

assert list(LOBE_SEGMENTATION_NAME) == LOBE_ORDER
assert RIGHT_LOBES | LEFT_LOBES == set(LOBE_ORDER)


def side_of(lobe: str) -> str:
    if lobe in RIGHT_LOBES:
        return SIDE_RIGHT
    if lobe in LEFT_LOBES:
        return SIDE_LEFT
    raise ValueError(f"unknown lobe '{lobe}' (expected {LOBE_ORDER})")


def sides_of(lobes) -> set:
    """The lungs touched by a set of lobes: empty, one side or BOTH_SIDES."""
    return {side_of(lobe) for lobe in lobes}


def lobe_mask_names(enabled=None, names=None) -> dict:
    """lobe -> mask name (no extension), in LOBE_ORDER, enabled lobes only."""
    enabled = LOBE_ORDER if enabled is None else enabled
    unknown = [lobe for lobe in list(enabled) + list(names or {}) if lobe not in LOBE_ORDER]
    if unknown:
        raise ValueError(f"unknown lobes {unknown} (expected {LOBE_ORDER})")
    return {lobe: (names or {}).get(lobe, LOBE_SEGMENTATION_NAME[lobe])
            for lobe in LOBE_ORDER if lobe in enabled}
