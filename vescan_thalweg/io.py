"""Generic I/O shared by the stages: JSON (with numpy), images on the CT grid, Slicer markups."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MARKUPS_SCHEMA = ("https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/"
                  "Markups/Resources/Schema/markups-schema-v1.0.3.json#")


SLICER_ORIENTATION = [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]


def control_point(cid, label, position):
    """One Slicer control point (same fields as VeScan's files)."""
    return {"id": str(cid), "label": label, "description": "", "associatedNodeID": "",
            "position": list(position), "orientation": SLICER_ORIENTATION, "selected": True,
            "locked": False, "visibility": True, "positionStatus": "defined"}


def fiducial_markups(name, labels, positions, coordinate_space="LPS"):
    """A Slicer point markup (Fiducial) with the given labels."""
    return {"@schema": MARKUPS_SCHEMA,
            "markups": [{"type": "Fiducial", "coordinateSystem": coordinate_space, "coordinateUnits": "mm",
                         "locked": False, "labelFormat": "%N-%d", "name": name,
                         "controlPoints": [control_point(i + 1, lab, pos)
                                           for i, (lab, pos) in enumerate(zip(labels, positions))]}]}


def write_json(path, data):
    """Indented JSON; numpy arrays and scalars converted."""
    def conv(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(data, indent=2, default=conv))


def write_margin_image(path, margin, geom):
    """The margin on the CT crop as an image with the CT geometry (to open in Slicer)."""
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(margin.astype(np.float32))
    img.SetSpacing(tuple(float(s) for s in geom.spacing[::-1]))
    img.SetOrigin(tuple(float(x) for x in geom.crop_origin_lps()))
    img.SetDirection(tuple(float(x) for x in geom.direction.ravel()))
    sitk.WriteImage(img, str(path), useCompression=True)
