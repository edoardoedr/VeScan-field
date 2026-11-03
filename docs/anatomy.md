# Stage `anatomy`

[← README](../README.md) · [surfaces](surfaces.md) · [centerlines](centerlines.md) · [anatomy](anatomy.md) · [surface_labels](surface_labels.md) · [clip](clip.md)

Lobe reachability, anatomical segments and crossing-vessel statistics. Module [`vescan_field/stages/anatomy/`](../vescan_field/stages/anatomy/). It works on the centerline only; [surface_labels](surface_labels.md) then carries the labels onto the surfaces.

| Input | Output (in `final_export_<s>/`) |
|---|---|
| `supporting_files/05_branch_tree.vtk` + `05_branch_tree_topology.json` ([centerlines](centerlines.md)); the lobe surfaces and `lobe_fissures.vtk` ([surfaces](surfaces.md)) | `anatomical_segments.vtk`: the centerline with every array below; `statistics.json` / `.csv`. With `pipeline.debug_files`, also `supporting_files/debug/09_lobe_reachability.vtk` / `.json` and `10_anatomical_segments.json` (legends) |

## Lobe reachability

Which lobes each group (branch or bifurcation) **leads to**, not just the ones it sits in.

1. **Containment, the seed only.** A branch belongs directly to a lobe when at least `containment_threshold` (0.1) of its points are inside the lobe surface (`vtkSelectEnclosedPoints`).
   - The threshold is low on purpose: segmented trees often stop before the periphery, and only the distal part of a branch enters the lobe it feeds.
   - The proximal trunks lie in the mediastinum, outside every lobe.
2. **Reachability, the actual answer.** The group graph is walked bottom-up: a group reaches its own lobes plus everything its children reach. More than one lobe = a shared trunk, which step 1 alone could never detect.

With N = number of active lobes:

| Array | Level | Values |
|---|---|---|
| `LobeMask` | group | bitmask, bit *i* = the subtree reaches lobe *i* |
| `NumReachableLobes` | group | 0 (none), 1 (exclusive), > 1 (shared trunk) |
| `Reaches_<LOBE>` | group | 0 does not reach, 1 reaches only this lobe, 2 reaches it and others |
| `LobeLabel` | group | 0 none/fragment, 1..N inside that lobe only, N+1 outside every lobe (trunk), N+2 touches more than one lobe or is downstream of one that does |
| `LobeLabelPoint` | point | the same scale point by point, without propagation: where the value changes is where the vessel enters or leaves a lobe |

`LobeLabel` is local, not per subtree: a group entirely inside one lobe must not be painted as trunk just because a distant descendant enters another lobe. Colour table: [`ColorTableLobeLabel_EN.csv`](../ColorTableLobeLabel_EN.csv).

**Fragments and excluded lobes.**
- Components detached from the main tree are not classified (0). They would otherwise be assigned to whatever lobe they happen to lie near.
- With a subset of lobes (`pipeline.lobes.enabled`), the maximal subtrees that reach no active lobe are the territory of the excluded lobes. From `no_lobe_region_min_length_mm` (25 mm) on they count as real territory, and the anatomical segments put them on the right side instead of leaving them unclassified.

## Anatomical segments

Every group and every point gets a medical anatomical segment, on **one scale for the three vessel types**, so they never collide when shown together:

| | trunk | main right / left | lobar RUL RML RLL LLL LUL | crossing, unclassified (type undetermined) | interlobar crossing | translobar crossing |
|---|---|---|---|---|---|---|
| airway | 1 trachea | 2 / 3 main bronchi | 4-8 | 9 | 73 | 74 |
| artery | 10 pulmonary trunk | 11 / 12 pulmonary arteries | 13-17 | 18 | 75 | 76 |
| vein | — | 21 / 22 venous trunks | 23-27 | 28 | 77 | 78 |

0 = unclassified. The names are those of [`ColorTableVessels_EN.csv`](../ColorTableVessels_EN.csv) (e.g. "Right upper lobe artery", "Multilobar vein - translobar"). A crossing keeps 9 / 18 / 28 ("Multilobar ... - unclassified") when its type cannot be decided: no fissures, or no piercing of a fissure found. The lobar values always follow the five canonical lobes: a run on a subset of lobes stays comparable with a full one.

**Per group** (`AnatomicalSegment`), three tiers:
1. touching several lobes, or downstream of a group that does → **crossing**;
2. inside one lobe → that **lobar** vessel. This is decided before looking at the subtree: a comb of branches all inside RUL can have a subtree reaching all five lobes;
3. outside every lobe → from the lungs its subtree reaches:
   - one lung → **main right/left**, or lobar if it reaches a single lobe;
   - both lungs → **trunk**.
- **Veins** have no single trunk: they converge on the left atrium through right and left venous trunks.
  - The confluence group, whose subtree feeds both lungs, is put on a side **cell by cell**, by comparing each cell's mean x with the patient's midline.
  - The midline is the mean of the right lobar cells' mean x and the left ones'. The tree's root is not used, because it sits in the off-centre left atrium.

**Per point** (`AnatomicalSegmentPoint`): the boundaries fall where they happen.
- A main vessel becomes lobar at the point where the centerline enters the lobe.
- A crossing starts at the point where the vessel changes lobe.
- A point outside every lobe is never lobar.
- A subtree reaching no lobe stays 0 (e.g. a stub at the trunk bifurcation).

## Crossing vessels

**One crossing vessel is one connected run of crossing-labelled groups** in the group graph.
- The label propagates to the whole downstream subtree, so counting labelled groups over-counts (75 groups for 4 real crossings on one patient).
- Joining by geometry fails too: the cells share no point ids, and stitching them by position split those 4 into 11.
- Crossings are numbered longest first (`CrossingId`).

**Two kinds**, measured against the fissure sheets of `lobe_fissures.vtk`:

| | behaviour | signature |
|---|---|---|
| **translobar** | goes through the fissure and stays on the other side | steep piercings |
| **interlobar** | runs along the fissure, in and out of the neighbouring lobe | ≥ 2 piercings, median angle < 30° |
| **unclassified** | labelled crossing but never touches a sheet (or no fissures) | — |

- **Where 30° comes from**: on 13 crossing vessels of 3 patients, the interlobar ones had median angles of 15.0, 4.1 and 1.6°, and the translobar ones 39.4° and above. 30° sits in the middle of a wide gap; it is not tuned.
- **A single piercing** is translobar by construction.
- **Angle convention**: to the fissure **plane**, 0-90°, unsigned, since the sheets' normals are not consistently oriented.
  - The local plane comes from a `fissure_patch_radius_mm` (5 mm) patch of the mesh, which leaves 0.06-0.31 mm RMS; the global plane would leave 1.4-5.8 mm.
  - The tangent comes from ± `tangent_window_mm` (5 mm) along the centerline.
  - Below `grazing_angle_deg` (10°) the angle is ill-conditioned: it is flagged `grazing`, and the decision uses the median.

**Three lengths**, three questions:

| Column | What it measures |
|---|---|
| `length_mm` | the whole labelled subtree, every branch summed: how much vasculature is downstream |
| `length_to_end_mm` | the longest single path from where the label starts to a leaf: how far the vessel still goes |
| `length_after_piercing_mm` | the same, from the fissure piercing (usually shorter: most piercings are not on the run's first group) |

- **Geometric lengths**: they sum the point-to-point distances, never a point count, since the spacing of the centerline points is not constant.
- **One cell per group**: each group is measured on one cell, the longest, so a branch traced by several centerlines is counted once.

Arrays on `anatomical_segments.vtk`:
- `CrossingId(Point)`: which crossing vessel a cell or point belongs to (0 = none). It matches the `crossing` column of the statistics.
- `CrossingType(Point)`: 1 interlobar, 2 translobar, 3 unclassified. Colour table: [`ColorTableCrossings_EN.csv`](../ColorTableCrossings_EN.csv).

## Statistics

With `statistics` on, the crossing vessels go to two files:
- **`statistics.json`**: counts (interlobar / translobar / unclassified), total length, and every crossing with its lobes (`from_lobe`, `to_lobe`), three lengths, number of piercings, angles and per-piercing detail;
- **`statistics.csv`**: **one row per crossing vessel**, so concatenating the files of many patients and structures is the whole aggregation. A structure with no crossing still writes one row, with `n_crossings` = 0, so "none found" stays distinct from "not run".

```
patient,structure,vessel_type,n_crossings,n_interlobar,n_translobar,crossing,crossing_type,from_lobe,to_lobe,length_mm,length_to_end_mm,length_after_piercing_mm,points,n_piercings,angle_median_deg,...,fissures,...
PD-1-Lung-00043,veins,vein,4,1,3,1,translobar,RML,RLL,79.636,55.522,55.651,179,2,42.38,...,Right oblique fissure,...
PD-1-Lung-00043,veins,vein,4,1,3,2,interlobar,RUL,RML,26.555,23.176,23.198,62,3,2.39,...,Right horizontal fissure,...
```

The `fissures` column is an independent cross-check. It comes from the sheet that was geometrically pierced, while `from_lobe`/`to_lobe` come from the reachability labels.

[`aggregate_statistics.py`](../aggregate_statistics.py) collects the `statistics.json` of many patients into one `.xlsx`:

```bash
python aggregate_statistics.py RESULTS_ROOT [RESULTS_ROOT ...] [-o out.xlsx]
```

## Configuration (`pipeline.anatomy`)

| Key | Default | Meaning |
|---|---|---|
| `containment_threshold` | 0.1 | fraction of a branch's points inside a lobe to belong to it |
| `containment_tolerance` | 1e-4 | `vtkSelectEnclosedPoints` tolerance |
| `no_lobe_region_min_length_mm` | 25 | a no-lobe subtree from this length on is real territory |
| `min_piercings_for_interlobar` | 2 | |
| `interlobar_max_median_angle_deg` | 30 | |
| `grazing_angle_deg` | 10 | below: angle flagged as grazing |
| `tangent_window_mm` | 5 | half-window of the centerline tangent |
| `fissure_patch_radius_mm` | 5 | radius of the local fissure plane |
| `fissure_patch_min_points` | 12 | |
| `duplicate_piercing_mm` | 0.1 | piercings closer than this are one |
| `statistics` | true | write `statistics.json` / `.csv` |
