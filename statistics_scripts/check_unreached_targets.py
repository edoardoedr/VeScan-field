#!/usr/bin/env python3
"""One-off check for NEXT.md 7.4: are the thalweg tips that VMTK does not reach on a detached piece
of 01_preprocessed.vtk?

    python statistics_scripts/check_unreached_targets.py <vmtk final_export_<s>> <thalweg final_export_<s>>

Runs VMTK seeded at the thalweg tips (statistics_scripts/vmtk_seeded.py, same functions and
criterion: a target is reached when a centerline end lies within 1 mm + 2 x its radius), then
labels the connected regions of 01_preprocessed.vtk and reports, for reached and unreached
targets, whether they lie on the source's region, their thalweg radius and how far the nearest
centerline end is. Run on PD-1-Lung-00043 arteries on 2026-11-13 (REPORT.md, section 7.4).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "statistics_scripts"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import vtk  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from vtk.util.numpy_support import vtk_to_numpy  # noqa: E402

import vmtk_seeded as V  # noqa: E402
from vescan_vmtk.config import EndpointsConfig, ExtractionConfig  # noqa: E402
from vescan_vmtk.stages.centerlines import endpoints as E  # noqa: E402


def main(vm, th):
    sp = vm / "supporting_files" / "01_preprocessed.vtk"
    surface = V.close_if_open(V.load_surface(sp), str(sp))
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(surface)
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()
    out = conn.GetOutput()
    reg = vtk_to_numpy(out.GetPointData().GetArray("RegionId"))
    kd = cKDTree(vtk_to_numpy(out.GetPoints().GetData()))
    region = lambda p: reg[kd.query(p)[1]]                                 # noqa: E731
    print("surface regions:", conn.GetNumberOfExtractedRegions(), "sizes:", sorted(np.bincount(reg))[::-1][:5])

    source = E.detect(sp, EndpointsConfig())[0]
    tips, radii = V.thalweg_tips(th)
    seeded = np.asarray(E.near_surface(surface, [list(t) for t in tips], radii,
                                       EndpointsConfig().endpoint_distance_radius_factor))
    line, _, sec = V.run_vmtk(surface, source, seeded, ExtractionConfig())
    xyz, cells, rad = V.lines(line)
    ends = np.array([c[-1] for c in cells] + [c[0] for c in cells])
    d = (np.linalg.norm(seeded[:, None] - xyz[ends][None], axis=2) - 2 * rad[ends][None]).min(1)
    ok = d <= 1.0
    regs = np.array([region(t) for t in seeded])
    tip_r = np.asarray(radii)[cKDTree(tips).query(seeded)[1]]
    src = region(source)
    print(f"targets {len(seeded)}, reached {ok.sum()}, extraction {sec:.0f} s")
    for name, m in (("reached", ok), ("unreached", ~ok)):
        print(f"  {name:9s} n={m.sum():4d}  on the source's region {np.mean(regs[m] == src):.2f}  "
              f"thalweg radius median {np.median(tip_r[m]):.2f} mm  nearest end median {np.median(d[m]):.2f} mm")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
