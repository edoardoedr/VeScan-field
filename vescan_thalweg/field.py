"""CT grid geometry of the masks (read by the surfaces stage).

The margin is not here: thalweg decodes it from the store on the model's own grid
(vescan_thalweg.margin). Internal coordinates: mm in array order (z,y,x). Exports in LPS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Geometry:
    spacing: np.ndarray                                                   # mm, order (z,y,x)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))       # LPS
    direction: np.ndarray = field(default_factory=lambda: np.eye(3))     # as SimpleITK
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, int))  # crop index

    def mm_to_lps(self, p_mm: np.ndarray) -> np.ndarray:
        idx = p_mm / self.spacing + self.offset          # indices (z,y,x) in the full volume
        ijk_mm = (idx * self.spacing)[:, ::-1]           # (x,y,z) in mm
        return self.origin + ijk_mm @ self.direction.T

    def index_to_lps(self, idx_zyx: np.ndarray) -> np.ndarray:
        """Crop indices (z,y,x) -> LPS."""
        return self.mm_to_lps(np.asarray(idx_zyx, float) * self.spacing)


def load_reference(path):
    """Geometry and array (z,y,x) of an image (CT or segmentation) via SimpleITK."""
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    geom = Geometry(np.array(img.GetSpacing(), float)[::-1],
                    np.array(img.GetOrigin(), float),
                    np.array(img.GetDirection(), float).reshape(3, 3))
    return geom, sitk.GetArrayFromImage(img)
