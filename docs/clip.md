# Stage `clip`

[← README](../README.md) · [surfaces](surfaces.md) · [centerlines](centerlines.md) · [anatomy](anatomy.md) · [surface_labels](surface_labels.md) · [clip](clip.md)

Cut the tree, then clip the surface. Module [`vescan_field/stages/clip/`](../vescan_field/stages/clip/). It is the **last stage**: it cuts the surface already labelled by [surface_labels](surface_labels.md), and the cut surface keeps the labels.

| Input | Output |
|---|---|
| `supporting_files/05_branch_tree.vtk` + `05_branch_tree_topology.json`; `branch_labeled_surface.vtk` (or, without stage surface_labels, the unlabelled `01_preprocessed.vtk`) | `final_export_<s>/cut_centerline.vtk`, `clipped_surface.vtk`; `supporting_files/06_cut_points.mrk.json` |

**Cutting the tree.**
- The groups with generation ≤ `max_generations` are kept (0 = trunk only).
- There is one cut point per kept-parent → dropped-child edge, at the last point of the parent where the child would have continued. `06_cut_points.mrk.json` has one Slicer point per dropped branch.
- `cut_centerline.vtk` is the cells of the kept groups, with every array.

**Clipping the surface.**
- **Which side**: each surface point is assigned to the nearer of the kept and dropped centerlines (distance to the polyline, not to its points).
- **The cut**: `vtkClipPolyData` cuts inside the triangles, where the two distances are equal, so the border is smooth and does not follow the triangulation.
- **Cleanup**: the largest component is kept, which drops the slivers near bifurcations where kept and dropped are ambiguous. With `cap` on, the holes are closed with one point at each border's edge-length-weighted barycenter and a fan of triangles, as `vtkvmtkCapPolyData` does.
- **Labels**:
  - a new point on a cut edge takes the values of the nearer original point, so labels are never interpolated;
  - a cap triangle takes the values of the triangle across its border edge;
  - a cap centre takes those of its nearest border point;
  - `Normals` is dropped, because it would be wrong on the caps.
- **Why not VMTK's clippers**: see [the comparison below](#comparison-with-vmtk-and-slicers-clip-vessel).

## Comparison with VMTK and Slicer's Clip Vessel

VeScan cuts the tree after N bifurcations, so one patient can need hundreds of cuts at once. The tools in VMTK and in the 3D Slicer extension [SlicerExtension-VMTK](https://github.com/vmtk/SlicerExtension-VMTK) are built for another task. VeScan therefore uses its own clip. It is closest to VMTK's branch clipper, but it uses a single pass.

The comparison is with VMTK `7efa98c` and SlicerExtension-VMTK `23190ba` (upstream as of October 2026).

| | VeScan `clip` | VMTK branch clipper | Slicer Clip Vessel |
|---|---|---|---|
| Code | `vescan_field/stages/clip/` | `vtkvmtkPolyDataCenterlineGroupsClipper` (`vmtkbranchclipper`) | `ClipVessel/ClipVessel.py` (reworked by the Slicer community in July 2026) |
| What is cut | every group of generation > N, chosen from the branch tree | the centerline groups given as input | the vessel ends at the clip points, placed by hand or detected at the centerline ends |
| Where the cut falls | where the distance to the kept centerline equals the distance to the dropped one | where the "power distance" `\|x − c\|² − r²` to a group's centerline equals the one to all the other groups | on a plane at each clip point, normal to the centerline (or set by hand) |
| Uses the vessel radius | no: plain distance to the polyline | yes: radius interpolated along the centerline | yes: it sizes the local patch (2.5 × radius) |
| Shape of the cut border | smooth, follows the surface between the two centerlines; not planar | the same kind of border as VeScan | planar; with `PLANE_PATCH` (the default) only the cells near the point are cut, so the rest of the vessel has no seam |
| Passes over the surface | one, whatever the number of cuts | one per group (with ~700 kept groups: 20+ minutes, not finished) | one per clip point, in sequence, each local |
| Pieces left over | largest component kept | none removed: every group is a separate piece with `GroupIds` | only the region connected to each clip point is removed; then the largest component |
| Caps | barycenter fan (as `vtkvmtkCapPolyData`), on by default | not part of the clipper (`vmtksurfacecapper`) | centre point, simple or smooth, optional remeshing; off by default; skipped if a border is not planar |
| Surface arrays | kept: the labels of [surface_labels](surface_labels.md) carry over, caps included | point `GroupIds` added | cell data carried over by the clip; boundary labels and an optional per-face id (`ModelFaceID`) |
| Flow extensions (CFD) | no | `vmtkflowextensions`, separate | yes, with per-end length |

**Why VeScan does it differently.**
- **Number of cuts**: one patient has hundreds of dropped groups. VMTK's per-group clipper was the original Clip Vessel method, and on real data it did not finish in 20 minutes. Its helper `vtkvmtkCenterlineSplitExtractor` also renumbers the `GroupIds` at every call.
- **Where to cut**: in VeScan the cut follows the anatomy, between the kept and dropped tree, so the border sits where one branch gives way to the other. A plane needs a point placed on the child, past its origin, and a direction. At a bifurcation, a plane through the parent would cut the sibling branches too.
- **Labels**: the clip is the last stage and must keep the labels already on the surface. Planes, caps and flow extensions in Clip Vessel are meant for CFD meshes and rebuild part of the mesh.

**What VeScan gives up.**
- **Planar borders**: they are needed for cross-section measurements or CFD inlets and outlets. For those, Clip Vessel can be run on `clipped_surface.vtk` with points near the cut points in `06_cut_points.mrk.json`. Those points lie at the bifurcation, so move them a little way down the dropped child.
- **The radius**: where a thin branch leaves a thick vessel, plain distance can move the border by a fraction of the parent's radius:
  - into the parent's wall, near the child's origin;
  - or along the child, leaving a short stub.

  VMTK's power distance makes a point on the parent's wall belong to the parent. Using the radius is a possible improvement, not yet measured.

## Configuration (`pipeline.clip`)

| Key | Default | Meaning |
|---|---|---|
| `max_generations` | 8 | keep generations 0..N |
| `cap` | true | close the cut holes |
