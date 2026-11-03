# Stage `surface_labels`

[← README](../README.md) · [surfaces](surfaces.md) · [centerlines](centerlines.md) · [anatomy](anatomy.md) · [surface_labels](surface_labels.md) · [clip](clip.md)

Module [`vescan_field/stages/surface_labels/`](../vescan_field/stages/surface_labels/). It carries the centerline's arrays onto the whole vessel surface, for visualisation.

| Input | Output |
|---|---|
| `supporting_files/01_preprocessed.vtk`; `05_branch_tree.vtk`; `anatomical_segments.vtk` | `final_export_<s>/branch_labeled_surface.vtk` |

Three passes, each one if its source exists:

| Pass | From | Arrays |
|---|---|---|
| branches | `05_branch_tree.vtk` | `branch_arrays` (default `GroupId`, `IsBifurcation`, `Generation`, `CellId`, `Length`, `AverageRadius`) |
| lobes | `anatomical_segments.vtk` | `NumReachableLobes`, `LobeMask`, `LobeLabel`, `LobeLabelPoint`, `Reaches_<LOBE>` |
| anatomy | `anatomical_segments.vtk` | `AnatomicalSegment(Point)`, `CrossingId(Point)`, `CrossingType(Point)` |

How a surface point gets its values:
- **Nearest centerline segment, radius-weighted.** Each surface point takes the values of the nearest centerline segment under VMTK's radius-weighted metric (`vtkvmtkPolyBallLine`'s, reimplemented in numpy in `centerline_distance.py`). A wide vessel therefore wins over a thin one passing close by.
- **Point arrays** (`...Point`) come from the nearest centerline point on that segment. The values are categorical and never interpolated, and cell and point values follow the same choice, so they never disagree.
- **Unlabelled centerline cells** (value 0) are ignored in the lobes and anatomy passes: an unclassified cell with a large radius would otherwise steal surface.
- **Islands** of a label smaller than `max_island_area_fraction` (0.25 %) of the surface area are filled with the value around them. These are wall points nearer to a centerline beyond a fissure. 0.25 % falls in an empty gap: islands reach 206-319 mm², real regions start at 804 mm². The crossings are protected, since on the surface they are exactly small patches.

The cut surface `clipped_surface.vtk` is made afterwards by [clip](clip.md) from this labelled surface, and keeps its labels.

## Configuration (`pipeline.surface_labels`)

| Key | Default | Meaning |
|---|---|---|
| `branch_arrays` | `GroupId`, `IsBifurcation`, `Generation`, `CellId`, `Length`, `AverageRadius` | arrays of the branches pass |
| `max_island_area_fraction` | 0.0025 | islands below this fraction of the area are filled |
| `max_island_passes` | 5 | fill passes, while anything changes |
| `max_protected_crossing_id` | 500 | `CrossingId` 1..N protected from the fill |
