"""The datasets file the comparison scripts read with --datasets (configs/analysis/*.json).

    {
      "exclude": ["<patient id>", ...],                       # dropped everywhere (optional)
      "datasets": {
        "luna16": {
          "runs": {"vmtk": "<results root>", "thalweg": "<results root>"},
          "configs": {"thalweg": "configs/thalweg/luna16_config.json"},  # store, classes, patients
          "patients_root": "Data/luna16"     # optional: where the patient folders (masks, store) are,
                                             # when not where the runs recorded them (a local copy)
        },
        ...
      }
    }

Each results root holds one folder per patient (final_export_<structure>/ inside). The run
labels are column suffixes (lowercase letters and digits) and must be the same, in the same
order, in every dataset; the first one is the baseline (vmtk). More runs (field, a field run
without pruning, VMTK seeded at the thalweg endpoints, ...) are more labels. `configs` gives the
pipeline config of a run, needed only by boundary_vs_reference.py (thalweg or field). Relative
paths are relative to the repo root (the scripts are run from there).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LABEL_RE = re.compile(r"^[a-z0-9]+$")


def load(path):
    """(datasets {name: {"runs": {label: Path}, "configs": {label: Path}, "patients_root": Path|None}},
    labels, exclude set)."""
    data = json.loads(Path(path).read_text())
    unknown = set(data) - {"exclude", "datasets", "notes"}
    if unknown:
        raise SystemExit(f"{path}: unknown key(s) {sorted(unknown)}")
    datasets = {}
    for name, d in data["datasets"].items():
        bad = set(d) - {"runs", "configs", "patients_root"}
        if bad:
            raise SystemExit(f"{path}: dataset {name}: unknown key(s) {sorted(bad)}")
        runs = {label: Path(root) for label, root in d["runs"].items()}
        for label in runs:
            if not LABEL_RE.match(label):
                raise SystemExit(f"{path}: dataset {name}: run label {label!r} must be lowercase letters/digits")
        if len(runs) < 2:
            raise SystemExit(f"{path}: dataset {name}: at least two runs")
        configs = {label: Path(p) for label, p in d.get("configs", {}).items()}
        datasets[name] = dict(runs=runs, configs=configs,
                              patients_root=Path(d["patients_root"]) if d.get("patients_root") else None)
    labelsets = {tuple(d["runs"]) for d in datasets.values()}
    if len(labelsets) != 1:
        raise SystemExit(f"{path}: every dataset must have the same runs, in the same order: {sorted(labelsets)}")
    return datasets, labelsets.pop(), set(data.get("exclude", []))


def common_cases(runs, structures=None, patients=None, exclude=()):
    """([(patient, structure, {label: final_export folder})] present in every run, sorted;
    {label: cases only in that run})."""
    found = {}
    for label, root in runs.items():
        found[label] = {}
        for p in sorted(x for x in Path(root).iterdir() if x.is_dir()):
            for d in sorted(p.glob("final_export_*")):
                found[label][p.name, d.name[len("final_export_"):]] = d
    common = sorted(set.intersection(*(set(f) for f in found.values())))
    out = []
    for patient, structure in common:
        if structures and structure not in structures or patients and patient not in patients:
            continue
        if patient in exclude:
            continue
        out.append((patient, structure, {label: found[label][patient, structure] for label in runs}))
    return out, {label: len(set(found[label]) - set(common)) for label in runs}
