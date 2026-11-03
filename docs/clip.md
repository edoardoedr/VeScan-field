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
- **Why not VMTK's clipper**: VMTK's `vtkvmtkPolyDataCenterlineGroupsClipper` (Slicer's Clip Vessel) became unusable with hundreds of kept groups (20+ minutes). VMTK's flow extensions (CFD only) are not ported.

## Configuration (`pipeline.clip`)

| Key | Default | Meaning |
|---|---|---|
| `max_generations` | 8 | keep generations 0..N |
| `cap` | true | close the cut holes |
