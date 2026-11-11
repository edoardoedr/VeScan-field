"""The single scale of anatomical labels (AnatomicalSegment), the same as VeScan's.
The names (written to the outputs) are those of the colour table ColorTableVessels_EN.csv.

Each vessel type has a block of 9 values, always on the 5 canonical lobes (LOBE_ORDER), so a
run on a subset of lobes stays comparable with a full one:

    offset 0 trunk, 1 right, 2 left, 3-7 lobar (RUL RML RLL LLL LUL), 8 crossing
    airway 1-9 (1 trachea), artery 10-18 (10 pulmonary trunk), vein 20-28 (20 never used: the
    veins have no trunk, they converge on the atrium through the right/left venous trunks)

Refined crossings, in a separate block so as not to renumber the others nor collide with
LobeSegment_* (30-54) and fissures (60-72): 73/74 airway interlobar/translobar, 75/76 artery,
77/78 vein. 9/18/28 stays "<Multilobar ...> - unclassified": a crossing of undetermined type
(no fissures or no piercing found).
0 = unclassified.
"""
from __future__ import annotations

from vescan_thalweg.lobes import LOBE_ORDER, SIDE_LEFT, SIDE_RIGHT

OFFSET_TRUNK, OFFSET_RIGHT, OFFSET_LEFT, OFFSET_LOBE_BASE, OFFSET_CROSSING = 0, 1, 2, 3, 8
SIDE_OFFSET = {SIDE_RIGHT: OFFSET_RIGHT, SIDE_LEFT: OFFSET_LEFT}
BLOCK_START = {"airway": 1, "artery": 10, "vein": 20}

# crossing type (CrossingType)
CROSSING_NONE, CROSSING_INTERLOBAR, CROSSING_TRANSLOBAR, CROSSING_UNCLASSIFIED = 0, 1, 2, 3
CROSSING_TYPE_NAME = {CROSSING_NONE: "none", CROSSING_INTERLOBAR: "interlobar",
                      CROSSING_TRANSLOBAR: "translobar", CROSSING_UNCLASSIFIED: "unclassified"}
REFINED_CROSSING_START = 73
REFINED_CROSSING_ORDER = ["airway", "artery", "vein"]

UNCLASSIFIED_NAME = "Unclassified"
NAMES = {
    "airway": {
        "trunk": "Trachea", "right": "Right main bronchus", "left": "Left main bronchus",
        "crossing": "Multilobar bronchus",
        "RUL": "Right upper lobe bronchus", "RML": "Right middle lobe bronchus",
        "RLL": "Right lower lobe bronchus", "LLL": "Left lower lobe bronchus",
        "LUL": "Left upper lobe bronchus",
    },
    "artery": {
        "trunk": "Pulmonary trunk", "right": "Right pulmonary artery", "left": "Left pulmonary artery",
        "crossing": "Multilobar artery",
        "RUL": "Right upper lobe artery", "RML": "Right middle lobe artery",
        "RLL": "Right lower lobe artery", "LLL": "Left lower lobe artery",
        "LUL": "Left upper lobe artery",
    },
    "vein": {   # no "trunk": the veins have no single trunk
        "right": "Right pulmonary venous trunk", "left": "Left pulmonary venous trunk",
        "crossing": "Multilobar vein",
        "RUL": "Right upper lobe vein", "RML": "Right middle lobe vein",
        "RLL": "Right lower lobe vein", "LLL": "Left lower lobe vein",
        "LUL": "Left upper lobe vein",
    },
}


def lobe_value(lobe, vessel_type):
    return BLOCK_START[vessel_type] + OFFSET_LOBE_BASE + LOBE_ORDER.index(lobe)


def side_value(side, vessel_type):
    return BLOCK_START[vessel_type] + SIDE_OFFSET[side]


def trunk_value(vessel_type):
    return BLOCK_START[vessel_type] + OFFSET_TRUNK


def crossing_value(vessel_type):
    """Crossing of undetermined type (9/18/28)."""
    return BLOCK_START[vessel_type] + OFFSET_CROSSING


def refined_crossing_value(vessel_type, crossing_type):
    """73-78 for interlobar/translobar, None for the other types (they stay 9/18/28)."""
    kind = {CROSSING_INTERLOBAR: 0, CROSSING_TRANSLOBAR: 1}.get(crossing_type)
    if kind is None:
        return None
    return REFINED_CROSSING_START + 2 * REFINED_CROSSING_ORDER.index(vessel_type) + kind


def crossing_values(vessel_type):
    """Every value meaning "crossing" for a vessel type: whoever re-reads a labelled file must
    use these, not just 9/18/28."""
    return {crossing_value(vessel_type), refined_crossing_value(vessel_type, CROSSING_INTERLOBAR),
            refined_crossing_value(vessel_type, CROSSING_TRANSLOBAR)}


def label_name(label, vessel_type):
    if label == 0:
        return UNCLASSIFIED_NAME
    names = NAMES[vessel_type]
    if label >= REFINED_CROSSING_START:
        kind = "interlobar" if label == refined_crossing_value(vessel_type, CROSSING_INTERLOBAR) \
            else "translobar"
        return f"{names['crossing']} - {kind}"
    offset = label - BLOCK_START[vessel_type]
    if offset == OFFSET_TRUNK:
        return names.get("trunk", f"<{vessel_type} has no trunk segment>")
    return {OFFSET_RIGHT: names["right"], OFFSET_LEFT: names["left"],
            OFFSET_CROSSING: f"{names['crossing']} - unclassified"}.get(offset) or names[LOBE_ORDER[offset - OFFSET_LOBE_BASE]]


def all_labels(vessel_type):
    """Every value a vessel type can have (veins without the trunk)."""
    start = BLOCK_START[vessel_type]
    trunk = [] if vessel_type == "vein" else [start + OFFSET_TRUNK]
    return ([0] + trunk + [start + OFFSET_RIGHT, start + OFFSET_LEFT]
            + [start + OFFSET_LOBE_BASE + i for i in range(len(LOBE_ORDER))]
            + [start + OFFSET_CROSSING, refined_crossing_value(vessel_type, CROSSING_INTERLOBAR),
               refined_crossing_value(vessel_type, CROSSING_TRANSLOBAR)])
