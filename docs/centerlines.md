# Stage `centerlines` (field)

[← README](../README.md) · [surfaces](surfaces.md) · [centerlines](centerlines.md) · [anatomy](anatomy.md) · [surface_labels](surface_labels.md) · [clip](clip.md)

The centerline tree without VMTK. Module [`vescan_field/stages/centerlines/`](../vescan_field/stages/centerlines/). The centerline tree comes from the **continuous field of the segmentation network**, not from a surface. The method follows the field-based approach described by Mike Halle (SPL) for his [haversack](https://github.com/mhalle/haversack).

| Input | Output |
|---|---|
| the ranked store `lung_vessels.duckn.zip` and the structure's masks, in the patient folder | `final_export_<s>/supporting_files/05_branch_tree.vtk`, `05_branch_tree_topology.json`: the same files, arrays and format as VeScan-vmtk's, read by every later stage |

## The margin, from the store (before the stage)

The store, written by `batch_segment.py` of [edoardoedr/haversack](https://github.com/edoardoedr/haversack), keeps per voxel the classes in rank order and how many logits each one is behind the winner.
- **The margin**: for the structure's classes S (e.g. airways = lumen + wall) the orchestrator builds `m_S = max_{c∈S} d_c − max_{k∉S} d_k`, interpolated onto the CT grid and cropped around the structure. It is positive inside, zero on the boundary.
- **QA**: `run_summary.json` compares the margin's argmax with the structure's masks (Dice ~0.999 on real data).
- **Why lumen + wall for the airways**: with the lumen alone the radii come out about half of the true ones.

## Steps

1. **extract**:
   - **boundary**: the sub-voxel boundary `{m = 0}` (zero crossings on the grid edges, normals from the gradient);
   - **distance**: the distance of every inside voxel from it, in the tangent plane of the k nearest crossings;
   - **graph**: inside voxels joined to their 26 neighbours when the field at the midpoint is positive too (diagonal contacts through the background do not count); edge weight L / (D² + ε), so paths far from the wall are cheaper;
   - **TEASAR**, seedless: the root is the thickest voxel; the farthest uncovered voxel is joined to the tree by the cheapest path, and a tube of radius `teasar_scale`·D + `teasar_const` around it is covered, until nothing is left. Terminal branches shorter than `min_branch_mm` are dropped;
   - **components**: with `surfaces.preprocess.keep_largest_component` (default) only the largest connected component gets a tree, so centerlines and surface have the same components. Otherwise every component of at least `min_voxels` voxels gets its own;
   - **refinement**: each segment between junctions is smoothed, moved onto the ridge of the distance in the plane orthogonal to it (`ridge_iterations`, `ridge_grid`), and resampled every `step_mm`;
   - **radius**: the distance from the boundary, with an interval from the boundaries at ± `delta` logits.
2. **topology**: the segments are oriented from the root, with parent, children and generation.
   - **Root at the inflow end** (`inflow_root_structures`: artery, airways). TEASAR's root is the widest point, which falls on the pulmonary-trunk bifurcation or mid-trachea. The root moves to the nearest end whose terminal segment has a radius ≥ `inflow_min_radius_fraction` × the maximum, so the first bifurcation is a real junction.
   - **Trachea cut by the field of view**: the fork TEASAR leaves there is dropped (`inflow_fork_radii`).
   - **Veins**: the root stays at the atrial confluence.
3. **tree**: the groups, branches and bifurcations.
   - **The bifurcation is a point**, the junction. Its group is the minimal one around it: one point before on the parent and one after on each child, `bifurcation_halfwidth_mm` (default `step_mm`) from the junction.
   - **No merging**: close junctions are not merged, so bifurcations stay binary, as vessels divide.
   - **Mid-vessel root**: the segments leaving it are joined under the longer subtree, so the tree has one root.
4. **export_tree**: the `05_` files, in the format of VeScan-vmtk's.
   - `05_branch_tree.vtk` has one cell per branch and per bifurcation child. Cell arrays `GroupId`, `IsBifurcation`, `Generation`, `Length`, `AverageRadius` (plus `GroupIds`, `Blanking`); point array `Radius`.
   - `05_branch_tree_topology.json` has roots, nodes and edges. For the veins a virtual root joins the separate main trunks (`add_virtual_root_for_veins`).

With `pipeline.debug_files`, also in `supporting_files/debug/`:
- `05_bifurcation_points.mrk.json`, `05_branch_tree_curves.mrk.json` (Slicer markups);
- the method's own files: `field_margin.nii.gz` (the margin on the CT crop), `field_centerlines.vtk` / `.mrk.json` (the segments before grouping), `field_segments.csv`, `field_graph.json` and `field_graph_nodes.mrk.json` (roots, junctions, ends).

## Compared with the VMTK centerlines

On a test patient, against the VMTK tree from the same masks:
- **The field tree contains the VMTK tree.** 99.8-99.9 % of the VMTK length lies within the field tree, at a median distance of 0.23-0.26 mm, with radii 0.08-0.09 mm smaller.
- **It finds more thin branches**, with the same length/radius ratio as the shared ones.
- **Generations are about twice VMTK's.** A bifurcation is a point here and a zone in VMTK, so a trifurcation counts as two binary ones, and there are more branches. Compare the two pipelines by anatomical segment, not by generation.
- **It is fast**: about 30 s per structure, against 2-8 min for VMTK's branch extraction.

## Configuration (`pipeline.centerlines`, `pipeline.field`)

| Key | Default | Meaning |
|---|---|---|
| `field.pad_voxels` | 3 | padding of the store box when building the margin |
| `delta` | 2.0 | half-width of the radius interval (logits) |
| `min_branch_mm` | 3.0 | shortest terminal branch |
| `teasar_scale`, `teasar_const` | 1.5, 0.5 | covered tube = scale · D + const |
| `min_voxels` | 30 | smaller components get no tree (with `keep_largest_component` off) |
| `step_mm` | 0.5 | resampling step of the centerlines |
| `crop_pad_voxels` | 4 | crop padding around the structure |
| `boundary_neighbours` | 8 | boundary crossings per distance (median over k) |
| `graph_epsilon` | 1e-3 | edge weight L / (D² + ε) |
| `smooth_iterations_before`, `_after` | 2, 1 | smoothing around the ridge step |
| `ridge_iterations`, `ridge_grid` | 3, 5 | ridge refinement: passes, n × n grid in the orthogonal plane |
| `inflow_root_structures` | `["artery", "airways"]` | structures whose root moves to the inflow end |
| `inflow_min_radius_fraction` | 0.5 | |
| `inflow_fork_radii` | 2.0 | fork at the inflow: leaves within this many radii of the root are dropped |
| `write_vescan` | true | write the `05_` files (needed by the later stages) |
| `bifurcation_halfwidth_mm` | null (= `step_mm`) | half-width of the bifurcation group |
| `short_segment_divisor` | 3.0 | on a very short segment the bifurcation part is at most L/3 |
| `add_virtual_root_for_veins` | true | |
| `virtual_root_min_length_fraction` | 0.1 | a vein root goes under the virtual root when its downstream tree is at least this fraction of the longest one |
