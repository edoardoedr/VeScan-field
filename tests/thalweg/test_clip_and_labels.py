"""Stages clip and surface_labels on test_anatomy's synthetic tree (trunk, bifurcation, main
right -> RUL with a branch going down into RML, main left -> LUL), with a tube around it."""
import json
from types import SimpleNamespace

import numpy as np
import vtk

from vescan_thalweg import surface
from vescan_thalweg.config import AnatomyConfig, ClipConfig, SurfaceLabelsConfig
from vescan_thalweg.field import Geometry
from vescan_thalweg.lobes import lobe_mask_names
from vescan_thalweg.stages import anatomy, clip, surface_labels
from vescan_thalweg.stages.surfaces.convert import convert_mask
from vescan_thalweg.stages.surface_labels.transfer import fill_islands
from .test_anatomy import run_anatomy, synthetic_tree

SP = np.array([0.5, 0.5, 0.5])
ORIGIN = np.array([-60.0, -10, -30])                       # LPS of voxel (0,0,0)


def tube_surface(groups, radius=2.0):
    """Closed surface around the tree's segments (a mask of capsules)."""
    shape = (110, 40, 220)                                  # z, y, x
    z, y, x = np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, SP)], indexing="ij")
    P = np.stack([x, y, z], -1) + ORIGIN
    mask = np.zeros(shape, bool)
    for g in groups.values():
        for pts, _ in g["cells"]:
            for a, b in zip(pts[:-1], pts[1:]):
                ab = b - a
                t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
                mask |= np.linalg.norm(P - (a + t[..., None] * ab), axis=-1) <= radius
    return convert_mask(mask, Geometry(SP, origin=ORIGIN))


def test_clip_and_labels(tmp_path):
    seg, _, p = run_anatomy(tmp_path)
    groups, _, _ = synthetic_tree()
    surface.save_surface(tube_surface(groups), p.preprocessed)
    ctx = SimpleNamespace(paths=p, results_dir=tmp_path, lobes=lobe_mask_names(), structure="artery",
                          timings={})

    surface_labels.run(ctx, SurfaceLabelsConfig())
    full = surface.load_surface(p.branch_labeled_surface)
    xyz, seg_point = surface.points(full), surface.point_array(full, "AnatomicalSegmentPoint")
    trunk = (np.abs(xyz[:, 0]) < 1) & (xyz[:, 2] < 10)
    left = xyz[:, 0] > 20
    assert (seg_point[trunk] == 10).mean() > 0.95            # pulmonary trunk
    assert (seg_point[left] == 17).mean() > 0.95             # left upper lobar artery

    clip.run(ctx, ClipConfig(max_generations=0))           # last: trunk + first bifurcation
    cut = surface.load_surface(p.cut_centerline)
    assert set(surface.cell_array(cut, "GroupId").tolist()) == {0, 1}
    marks = json.loads(p.cut_points.read_text())["markups"][0]["controlPoints"]
    assert sorted(p["label"] for p in marks) == ["cut_group1_dropped_group2", "cut_group1_dropped_group7"]
    clipped = surface.load_surface(p.clipped_surface)
    assert surface.count_open_edge_points(clipped) == 0      # caps closed
    assert surface.signed_volume(clipped) > 0
    assert np.abs(surface.points(clipped)[:, 0]).max() < 6   # trunk and branch stubs remain
    # the labels of the whole surface, carried through the cut (not interpolated) and onto the caps
    for name in ("GroupId", "AnatomicalSegment", "LobeLabel"):
        values = set(np.unique(surface.cell_array(full, name)).tolist())
        assert set(np.unique(surface.cell_array(clipped, name)).tolist()) <= values, name
    for name in ("AnatomicalSegmentPoint", "LobeLabelPoint"):
        values = set(np.unique(surface.point_array(full, name)).tolist())
        assert set(np.unique(surface.point_array(clipped, name)).tolist()) <= values, name
    assert surface.point_array(clipped, "KeptDroppedSignedDistance") is None

    # generation_lobes: the limit applies only in the listed lobes; a group that reaches another lobe stays
    kept = lambda cfg: (clip.run(ctx, cfg), set(surface.cell_array(surface.load_surface(p.cut_centerline),
                                                                   "GroupId").tolist()))[1]
    everywhere = kept(ClipConfig(max_generations=0))
    assert kept(ClipConfig(max_generations=0, generation_lobes=list(ctx.lobes))) == everywhere
    only_left = kept(ClipConfig(max_generations=0, generation_lobes=["LUL"]))
    assert everywhere < only_left                            # right-side groups are not limited any more
    assert kept(ClipConfig(max_generations=0, generation_lobes=["RUL", "RML", "RLL", "LLL"])) > everywhere


def grid_surface(n=40):
    plane = vtk.vtkPlaneSource()
    plane.SetResolution(n, n)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(plane.GetOutputPort())
    tri.Update()
    return tri.GetOutput()


def test_fill_islands():
    poly = grid_surface()
    xy = surface.points(poly)[:, :2]
    label = np.ones(len(xy), np.int32)
    label[(np.abs(xy[:, 0] - 0.2) < 0.02) & (np.abs(xy[:, 1]) < 0.02)] = 2      # small island
    label[(np.abs(xy[:, 0] + 0.2) < 0.02) & (np.abs(xy[:, 1]) < 0.02)] = 3      # protected island
    label[xy[:, 0] > 0.35] = 4                                                  # large region
    surface.add_point_array(poly, "L", label)
    filled = fill_islands(poly, ["L"], max_area_fraction=0.0025, protected={"L": {3}})
    out = surface.point_array(poly, "L")
    assert filled > 0 and 2 not in out and 3 in out and 4 in out
