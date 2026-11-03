# Stage `surfaces`

[← README](../README.md) · [surfaces](surfaces.md) · [centerlines](centerlines.md) · [anatomy](anatomy.md) · [surface_labels](surface_labels.md) · [clip](clip.md)

Module [`vescan_field/stages/surfaces/`](../vescan_field/stages/surfaces/). Every surface of the pipeline comes from the masks.

| Runs | Input | Output |
|---|---|---|
| once per patient (`run_patient`) | the lobe masks (`pipeline.lobes`) | in `surfaces_root_dir[/<patient>]/`: `lung_*_lobe_*.vtk` (the lobes, not decimated), `lobe_segments.vtk` + `lobe_segments.json`, `lobe_fissures.vtk` |
| per structure (`run`) | the union of the structure's masks (`batch.structures.<s>.classes`) | `<surfaces>/<first class>.vtk` (the raw surface, e.g. `lung_airways.vtk` = lumen + wall); `final_export_<s>/supporting_files/01_preprocessed.vtk` |

## convert: mask → closed surface

The same surface as 3D Slicer's "Closed surface" representation (`vtkBinaryLabelmapToClosedSurfaceConversionRule`):

1. crop to the mask's bounding box + 1 voxel; a mask touching the volume border is padded, so the surface is closed there too;
2. iso-contour with `vtkDiscreteFlyingEdges3D` (default) or `vtkSurfaceNets3D`;
3. optional `vtkDecimatePro` (`decimation_factor`) and `vtkWindowedSincPolyDataFilter` smoothing, with Slicer's formulas;
4. voxel → world with the mask's geometry, in LPS;
5. consistent normals.

## preprocess: the vessel surface for the centerlines

`01_preprocessed.vtk` is the surface the later stages work on: [surface_labels](surface_labels.md) and [clip](clip.md), and in VeScan-vmtk the [centerlines](centerlines.md), which are extracted from it. VeScan-field computes its centerlines from the ranked store instead.

- **largest connected component** only (`keep_largest_component`): segmentations carry detached debris, which would also attract VeScan-vmtk's automatic start point. In VeScan-field the centerlines follow the same choice: with this key on they are computed on the largest connected component of the field only, so centerlines and surface have the same components;
- **decimation to a number of points**: `target_points` is the final number of points.
  - pyfqmr (FastQuadric, the algorithm of Slicer's decimation) takes a triangle target but stops early when its error threshold, which grows with `aggressiveness`, allows no more collapses. At a fixed aggressiveness of 4 an artery tree stays at ~73k points whatever the target.
  - The stage therefore bisects the aggressiveness for the **minimum** value that reaches `target_points × (1 + tolerance)`: the least distorted mesh with that many points.
  - On an artery tree, 347k → 83k points takes 9 passes and 8 s.
- clean, triangulation, consistent normals, oriented outward.

## lobe_segments: one lobe mesh and the interlobar fissures

The five lobe surfaces are merged into one mesh, `lobe_segments.vtk`, with one cell array per vessel type. Load it once in Slicer and switch the active scalar:

| Array | Values |
|---|---|
| `LobeSegment_Vein` | 30 RUL, 31 RML, 32 RLL, 33 LLL, 34 LUL |
| `LobeSegment_Artery` | 40-44 (same order) |
| `LobeSegment_Airway` | 50-54 (same order) |
| `FissureContact` | 60 RUL\|RML, 61 RUL\|RLL, 62 RML\|RLL, 63 LUL\|LLL (0 = not a fissure) |
| `Fissure` | 70 right horizontal, 71 right oblique (= 61 + 62), 72 left oblique |

The values never collide with the vessel labels of the [anatomy](anatomy.md) stage. Colour table: [`ColorTableLobesFissures_EN.csv`](../ColorTableLobesFissures_EN.csv).

**Fissures.** They need no extra input, because they are already in the lobe surfaces.
- **Why they can be found**: adjacent lobes come from one label map, so their surfaces share coincident triangles on the common border. A fissure is the set of a lobe's cells within `fissure_contact_tolerance` (1 mm) of the neighbouring lobe.
- **The tolerance is a guard against sub-voxel jitter, not a knob**: the selection sits on a plateau, and the area measured from the two sides agrees within 0.1 %.
- **The distance**: it is the nearest-vertex distance (a KD-tree). On the fissure the vertices of the two lobes coincide, and the selection differs from VTK's point-to-surface distance by 0.05-0.1 %, at the rim only.
- **Four contacts make three anatomical fissures**: the right oblique fissure separates RLL from both RUL and RML. The other lobe pairs do not touch. Contact components below `min_fissure_component_area` (50 mm²) and contacts below `min_fissure_area` (100 mm²) are dropped.
- **Two outputs**:
  - `lobe_segments.vtk` carries the fissure arrays next to the `LobeSegment_*` ones, so a fissure cell appears twice there, once per lobe;
  - `lobe_fissures.vtk` has each sheet **once**: use it as the geometric object (the [anatomy](anatomy.md) stage measures the crossings against it).
- **Legend**: `lobe_segments.json` names every value and gives, per fissure, its area and least-squares plane (centroid, normal, RMS).

Two caveats:
- **The fissure is a curved sheet, not a plane.** A plane fit leaves 1.4-5.8 mm RMS and peak deviations of 5-16 mm. The plane in the legend is a descriptor only: use the mesh to decide which side of a fissure something is on.
- **It is the segmentation's fissure, not the CT's.** Where the anatomical fissure is incomplete, the segmentation still closes the lobe boundary, so the extracted sheet is always complete.

## Configuration (`pipeline.surfaces`)

| Key | Default | Meaning |
|---|---|---|
| `convert.method` | `flying-edges` | or `surface-nets` |
| `convert.smoothing_factor` | 0.5 | Slicer's smoothing factor (0 = off) |
| `convert.decimation_factor` | 0.0 | `vtkDecimatePro` during conversion (0 = off, Slicer's default) |
| `convert.compute_normals` | true | |
| `preprocess.enabled` | true | write `01_preprocessed.vtk` (needed by the later stages) |
| `preprocess.decimate` | true | |
| `preprocess.target_points` | 100000 | final number of points |
| `preprocess.tolerance` | 0.05 | accepted excess over `target_points` |
| `preprocess.keep_largest_component` | true | in VeScan-field, also restricts the centerlines to the largest component |
| `lobe_segments.enabled` | true | lobe mesh and fissures (once per patient) |
| `lobe_segments.extract_fissures` | true | without fissures the crossings stay unclassified |
| `lobe_segments.fissure_contact_tolerance` | 1.0 mm | |
| `lobe_segments.min_fissure_component_area` | 50 mm² | |
| `lobe_segments.min_fissure_area` | 100 mm² | |
