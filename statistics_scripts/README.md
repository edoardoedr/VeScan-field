# statistics_scripts

The comparison of the centerline pipelines on the dataset runs, mainly **VMTK vs thalweg** (can thalweg replace VMTK?), with field as an optional third column, and the crossing statistics. Published with both packages. The comparison scripts import all three pipelines (`vescan_vmtk`, `vescan_field`, `vescan_thalweg`): install both packages to run them. `aggregate_statistics.py` and `plots/` need only the package that made the runs.

Every script reads the runs from one **datasets file** in [`configs/analysis/`](../configs/analysis/): for each dataset, the results root of each pipeline, the pipeline config (store and classes, for script 3), and the patients to exclude.

| File | Runs |
|---|---|
| `server.json` | vmtk, thalweg (server runs) |
| `server_with_field.json` | vmtk, thalweg, field |
| `sample.json`, `sample_with_field.json` | server patients copied here: runs in `Outputs/<dataset>/<pipeline>/<patient>/`, inputs in `Data/<dataset>/<patient>/` (`patients_root`) |

vmtk is always the first run, the baseline. To compare one more run (VMTK seeded at the thalweg endpoints, a run without pruning, ...), add a `label: root` to every dataset: every table gets a column for it.

## Layout

The data and the results live in `analisys/` (not in git, see its README); the code is all here.

```
statistics_scripts/
├── compare_centerlines.py, crossings_by_radius.py,       extraction: read the pipeline runs (on the server),
│   boundary_vs_reference.py, vmtk_seeded.py,             write analisys/extractions/<date>_<what>/
│   aggregate_statistics.py
├── datasets_file.py, line_distance.py                    shared helpers
├── curate_extraction.py                                  split / clean an extraction (pipelines, excluded patients)
├── check_unreached_targets.py                            one-off check for NEXT.md 7.4 (one patient)
└── plots/                                                read an extraction, write analisys/results/<extraction>/
    ├── report_vmtk_vs_thalweg.py, report_vmtk_vs_field.py    figures and tables.md of a comparison report
    └── fissure_*.py                                          the four fissure-crossing analyses
```

Run from the repo root, in the `vescan` env (plus `pandas openpyxl matplotlib`). No script overwrites an existing output (`--force`), and each writes a `.run.json` with the command, date, git commit and inputs.

## The commands

```bash
D=configs/analysis/server.json                 # server_with_field.json to add field
OUT=analisys/extractions/$(date +%F)_vmtk-vs-thalweg

# 1. the centerlines: agreement, precision against 01_preprocessed.vtk, depth, spurs, time, slice spacing
python statistics_scripts/compare_centerlines.py --datasets $D -o $OUT/centerline_comparison.xlsx

# 2. the crossings by vessel radius and depth past the fissure, with agreement between the pipelines
python statistics_scripts/crossings_by_radius.py --datasets $D -o $OUT/crossings_by_radius.xlsx

# 3. where thalweg's boundary lies against 01_preprocessed.vtk, by vessel radius; also writes it as a surface
python statistics_scripts/boundary_vs_reference.py --datasets $D --write-surfaces $OUT/thalweg_surfaces \
    -o $OUT/boundary_vs_reference

# 4. the precision of 1 again, against thalweg's own boundary instead of 01_preprocessed.vtk
python statistics_scripts/compare_centerlines.py --datasets $D --reference-surfaces $OUT/thalweg_surfaces \
    -o $OUT/centerline_comparison_thalwegref.xlsx

# 5. VMTK seeded at the thalweg tips vs its default seeds (no branch split: the raw centerlines)
python statistics_scripts/vmtk_seeded.py --datasets $D -o $OUT/vmtk_seeded.xlsx

# crossing statistics of one pipeline, for statistics_scripts/plots/fissure_*.py (one workbook per pipeline)
python statistics_scripts/aggregate_statistics.py luna16=<thalweg root> ... -o <extraction>/crossing_statistics.xlsx

# figures and tables of the report, from the extraction above (locally, once it is copied into analisys/)
python statistics_scripts/plots/report_vmtk_vs_thalweg.py --date $(date +%F)
```

- `--workers N` sets the processes (default: the CPU quota); `--structures` / `--patients` restrict 1, 3 and 4.
- **Memory**: about 0.4 GB per process for 1, 1.5-2 GB for 3. Distances to the centerlines use `line_distance.py` (exact plain distance, memory linear in the points): until 2026-11-12 they used `vescan_field.centerline_distance.nearest_segments`, whose memory grows with how far the points are from the line (3.5-5 GB per call on an artery, 64 GB saturated with 16 processes).
- Script 5 appends each finished case (failures included) to `<output>.cases.jsonl`: if it stops (Ctrl+C, crash, a lost session), the same command resumes from there, running only the missing cases (`--retry-failed` also reruns the failed ones); `--force` starts again. Each case runs in its own process, killed after `--timeout` seconds (default 3600) and recorded as failed: VMTK can stay inside one extraction indefinitely at 100 % CPU (seen on the server, 2026-11-12). The other scripts still write only at the end.
- Script 3 reads the store (over NFS on the server) once per structure. Without `--write-surfaces` it writes no surfaces; with it, about 15-25 MB per structure.
- Script 4 is the slowest: the thalweg boundary has ~5× the triangles of `01_preprocessed.vtk`. On a subset first if needed (`--patients`, or a datasets file with fewer datasets).
- Script 3 with `--boundary field` measures field's boundary instead (CT grid; needs the field run and config, `server_with_field.json`).

## What each measure answers

From Mike Halle's review of the 2026-11-03 comparison (NEXT.md, section 7). `<pl>` is a run label (vmtk, thalweg, field).

| Measure | Script | Columns |
|---|---|---|
| Mutual distance, matched length | 1 | `dist_<stat>_<a>_to_<b>`, `matched_<a>_in_<b>`, `matched_f1_<a>_<b>` |
| Computation time | 1 | `time_s_<pl>` (the whole structure; before 2026-11-12 it was doubled, see below) |
| Dice against the masks | 1 | `dice_field` (only field computes it; empty for vmtk and thalweg) |
| Crossings by radius and depth, agreement per group | 2 | `summary`: crossings per patient, prevalence; `agreement`: exact count match, Cohen's kappa |
| Centering, radius error, length outside: column (a), against `01_preprocessed.vtk` | 1 | `offset_rel_median_<pl>`, `abs_radius_err_median_<pl>`, `radius_err_median_<pl>`, `outside_frac_<pl>` |
| The same, column (b), against thalweg's own boundary | 4 | the same columns, in `centerline_comparison_thalwegref.xlsx` |
| Why (a) and (b) differ | 3 | thalweg boundary − reference, mm and % of radius, per radius bin |
| The same, column (c), analytic phantoms | — | not done (NEXT.md 7.6) |
| Curvature | 1 | `curvature_median_<pl>` |
| Terminal spurs, redefined | 1 | `spur_frac_<pl>`: tip protrudes less than its own radius beyond the parent wall. `short_leaf_frac_<pl>` keeps the old definition. Both undercount on pruned trees |
| Generations, plus depth not counted in bifurcations | 1 | `max_generation_<pl>` (thalweg with `bifurcations: "vmtk"` is comparable with VMTK), `path_max_mm_<pl>`, `path_p90_mm_<pl>`, `radius_min_mm_<pl>`, `radius_p05_mm_<pl>` |
| Surface reached by one pipeline only | 1 | `missing_area_pct_<pl>` (against the best of the others), `missing_area_pct_<a>_vs_<b>` |
| The same with VMTK seeded at the thalweg tips, and VMTK's failures | 5 | `missing_area_pct_{default,seeded}`, `missing_area_pct_both`, `reached_frac_*`, `excluded_*` (one raw VMTK centerline per seeding, no branch split) |
| Slice thickness | 1 | `slice_mm` (the masks' slice spacing) on every row; sheet `by_slice` (<=1, 1-2, 2-3, >3 mm) |

## What to expect

One patient, so an indication only.

| | vmtk | thalweg |
|---|---|---|
| max generation | 12 | 14 |
| path root → farthest tip | 255 mm | 272 mm |
| surface not reached | 17 % | 0.03 % |
| matched length (vmtk in thalweg / thalweg in vmtk) | 99.8 % | 67.5 % |
| median distance vmtk → thalweg | 0.11 mm | |
| time (whole structure) | 388 s | 61 s |
| \|radius error\| against `01_preprocessed.vtk` | 3.2 % | 10.0 % |
| \|radius error\| against thalweg's boundary | 10.4 % | 2.9 % |
| centering (offset / radius), same two references | 0.045 / 0.061 | 0.065 / 0.035 |

- thalweg contains VMTK's tree (99.8 %) and adds a third more length, in thin branches.
- **Precision depends on the reference.** thalweg's boundary lies outside `01_preprocessed.vtk` by ~0.54 mm on vessels under 1 mm radius, ~0.13 mm at 1-2 mm, ~0 above 3 mm (script 3): the reference's smoothing shrinks the thin vessels, and VMTK, which is computed from that reference, inherits it. Against each pipeline's own input, each is the more precise.
- **Times before 2026-11-12** (the 2026-11-03 extraction): `time_s` summed every step plus `total`, so every time was about double. The ratios between pipelines are unaffected.
