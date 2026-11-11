# Stage `centerlines` (thalweg)

[← README](../README.md#vescan-thalweg) · [surfaces](surfaces.md) · [centerlines](centerlines_thalweg.md) · [anatomy](anatomy.md) · [surface_labels](surface_labels.md) · [clip](clip.md)

The centerline tree from [thalweg](https://github.com/mhalle/thalweg) by Mike Halle, used through the fork [edoardoedr/thalweg](https://github.com/edoardoedr/thalweg). Module [`vescan_thalweg/stages/centerlines/`](../vescan_thalweg/stages/centerlines/). This stage only calls thalweg and turns its tree into VeScan's groups. Everything before and after it is the same as in VeScan-field.

| Input | Output |
|---|---|
| the ranked store `lung_vessels.duckn.zip` and the structure's classes | `final_export_<s>/supporting_files/05_branch_tree.vtk`, `05_branch_tree_topology.json`: the same files, arrays and format as VeScan-vmtk's and VeScan-field's, read by every later stage |

## The margin (before the stage)

[`vescan_thalweg/margin.py`](../vescan_thalweg/margin.py) decodes the structure's margin **on the store's own grid**. VeScan-field resamples it onto the CT grid instead.

- **Formula**: for the classes S of `batch.structures.<s>.classes`, it is `m_S = max_{c∈S} d_c − max_{k∉S} d_k`, from rankfield's deficits, clamped to ±clip as rankfield's margin. With one class it equals `rankfield.margin`.
- **Which classes**: the config decides. Airways are `["lung_airways", "lung_airways_wall"]` by default (lumen + wall), as in VeScan-field. With `["lung_airways"]` alone, thalweg traces the lumen, as it does by itself.
- **Grid**: all the classes must be in one store part.

## Steps

1. **trace**: `thalweg.centerlines.centerline_graph` on that margin (`pipeline.centerlines.trace`). It does everything below itself; see thalweg's `docs/vmtk-guide.md` and `docs/validation.md` for the details.
   - **Connectivity**: decided by the field's trilinear interpolant, including face saddles and cell interiors.
   - **Distance**: to the margin's sub-voxel zero crossings.
   - **TEASAR**, seedless: edge cost `L · mean(1/(d+ε)²)`, cover `scale·d + const`.
   - **Pruning**: terminal branches shorter than `2·R(junction) + 1 mm`. Flat tubes get wall pruning and recentering.
   - **Radius**: the inscribed ball, refined coarse to fine in `ridge_passes` passes.
   - **Root**: at the inlet, the widest end that runs off the grid. This replaces VeScan-field's inflow rules.
   - **Components**: thalweg traces the largest connected component only.

   Then `thalweg.centerlines.radius_interval` adds the ±`radius_interval_logits` bounds.
2. **bifurcations**, chosen by `pipeline.centerlines.bifurcations`:
   - **`"vmtk"`** (default): the bifurcation is a zone, as in VeScan-vmtk.
     - Thalweg's numpy port of vmtk splits the tree: `to_vmtk` (source-to-tip paths), `centerline_attributes`, `extract_branches` (= `vmtkbranchextractor`). It writes `GroupIds`, `CenterlineIds`, `TractIds` and `Blanking`.
     - From that split, groups, adjacency, orphan-root repair and generations are built exactly as in VeScan-vmtk's tree step.
     - The only change is that `vtkvmtkCenterlineUtilities` is replaced by `thalweg.vmtk.utilities`. The adjacency is indexed once (same lists, same order).
     - Generations are comparable with VeScan-vmtk's.
   - **`"point"`**: the bifurcation is the junction, as in VeScan-field.
     - Thalweg's edges are resampled every `point.step_mm`.
     - VeScan-field's grouping turns each junction into a minimal blanked group (one point before, one point after).
3. **export_tree**: the `05_` files.
   - Virtual root for the veins (`add_virtual_root_for_veins`).
   - Markups with `pipeline.debug_files`.

With `pipeline.debug_files`, also in `supporting_files/debug/`:
- `thalweg_margin.nii.gz`: the union margin on the store's grid;
- `thalweg_graph.thalweg.json.gz`: thalweg's graph, readable by `thalweg` itself (`thalweg table`, `thalweg export`, ...);
- with `"vmtk"`, `thalweg_split.vtp` (the split centerlines) and `05_branch_models/`;
- `05_bifurcation_points.mrk.json`, `05_branch_tree_curves.mrk.json`.

## Configuration (`pipeline.centerlines`)

| Key | Default | Meaning |
|---|---|---|
| `bifurcations` | `"vmtk"` | `"vmtk"`: blanked bifurcation zones; `"point"`: the junction |
| `trace.connectivity` | `"field"` | `"voxel"`: the 26-connected labelmap (comparison only) |
| `trace.cover_scale`, `trace.cover_constant_mm` | 1.5, 1.0 | TEASAR cover = scale · d + constant |
| `trace.cost_epsilon_mm` | 0.1 | edge cost `L · mean(1/(d+ε)²)` |
| `trace.ridge_passes` | 4 | radius refinement passes (1 = thalweg's research reference) |
| `trace.prune` | `"auto"` | `"length"`, `"wall"`, or by structure (wall for flat tubes) |
| `trace.recenter` | null | null = by structure (flat tubes only); true / false |
| `trace.root` | `"inlet"` | `"deepest"`: the tracer's own root, the deepest point |
| `trace.radius_interval_logits` | 2.0 | RadiusLow / RadiusHigh surfaces |
| `vmtk.resampling_step_mm` | 0.3 | step of the source-to-tip paths given to the branch extractor |
| `vmtk.vmtk_compatible` | false | true: vmtk's numbers, its defects included |
| `vmtk.min_branch_length` | 1.0 | shorter groups are dropped |
| `vmtk.orphan_root_repair_tolerance` | 0.5 | orphan groups re-attached by end-point distance |
| `point.step_mm` | 0.5 | resampling step |
| `point.bifurcation_halfwidth_mm` | null (= `step_mm`) | half-width of the bifurcation group |
| `point.short_segment_divisor` | 3.0 | on a very short segment the bifurcation part is at most L/3 |
| `write_vescan` | true | write the `05_` files (needed by the later stages) |
| `add_virtual_root_for_veins` | true | |
| `virtual_root_min_length_fraction` | 0.1 | |
