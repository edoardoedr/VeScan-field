"""Config: defaults, null = default, unknown keys, classes per structure, paths."""
import json
from pathlib import Path

import pytest

from vescan_field import config as cfgmod

MINIMAL = {"batch": {"patients_root_dir": "p", "surfaces_root_dir": "s", "results_root_dir": "r"}}


def test_defaults_and_relative_paths(tmp_path):
    cfg = cfgmod.build(MINIMAL, base_dir=tmp_path)
    assert cfg.batch.patients_root_dir == str((tmp_path / "p").resolve())
    assert cfg.pipeline.stages.centerlines is True
    assert cfg.pipeline.centerlines.step_mm == 0.5
    assert cfg.pipeline.field.pad_voxels == 3
    assert cfg.enabled_structures() == {"artery": ["lung_arteries"], "veins": ["lung_veins"],
                                        "airways": ["lung_airways", "lung_airways_wall"]}


def test_null_means_default_and_override():
    data = {**MINIMAL, "pipeline": {"centerlines": {"step_mm": None, "delta": 1.5},
                                    "field": {"pad_voxels": 5}, "debug_files": True}}
    cfg = cfgmod.build(data)
    assert cfg.pipeline.centerlines.step_mm == 0.5 and cfg.pipeline.centerlines.delta == 1.5
    assert cfg.pipeline.field.pad_voxels == 5 and cfg.pipeline.debug_files is True


def test_structures_enabled_and_classes():
    data = {"batch": {**MINIMAL["batch"], "structures": {
        "veins": {"enabled": False}, "airways": {"classes": ["lung_airways"]}}}}
    cfg = cfgmod.build(data)
    assert cfg.enabled_structures() == {"artery": ["lung_arteries"], "airways": ["lung_airways"]}


@pytest.mark.parametrize("data", [
    {**MINIMAL, "pipeline": {"centerlines": {"stepmm": 0.5}}},          # typo
    {**MINIMAL, "pipeline": {"cut_graph": {"max_generations": 5}}},     # stage that does not exist
    {**MINIMAL, "pipeline": {"centerlines": {"delta": "2"}}},           # wrong type
    {**MINIMAL, "pipeline": {"centerlines": {"inflow_root_structures": ["arteries"]}}},
    {"batch": {"patients_root_dir": "p"}},                              # results missing
    {"batch": {"patients_root_dir": "p", "results_root_dir": "r"}},     # surfaces missing
    {"batch": {**MINIMAL["batch"], "structures": {"artery": {"classes": []}}}},
    {**MINIMAL, "pipeline": {"lobes": {"enabled": ["RUL", "XYZ"]}}},
    {**MINIMAL, "pipeline": {"lobes": {"names": {"XYZ": "a"}}}},
    {**MINIMAL, "pipeline": {"lobes": {"names": {"RUL": 3}}}},
    {**MINIMAL, "pipeline": {"surfaces": {"convert": {"method": "marching"}}}},
    {**MINIMAL, "pipeline": {"surfaces": {"preprocess": {"target_points": 2}}}},
])
def test_errors(data):
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.build(data)


def test_example_config_loads():
    path = Path(__file__).resolve().parents[2] / "configs" / "field" / "example.json"
    cfg = cfgmod.load(path)
    assert cfg.batch.enabled is False
    assert json.loads(path.read_text())["pipeline"]["stages"] == {"surfaces": True, "centerlines": True, "anatomy": True, "clip": True, "surface_labels": True}
    assert cfg.pipeline.surfaces.preprocess.target_points == 100000


def test_lobes_names_and_surfaces_sections():
    data = {**MINIMAL, "pipeline": {"lobes": {"enabled": ["RUL", "RML"], "names": {"RUL": "my_rul"}},
                                    "surfaces": {"preprocess": {"target_points": 50000},
                                                 "lobe_segments": {"enabled": False}}}}
    cfg = cfgmod.build(data)
    assert cfg.pipeline.lobes.names == {"RUL": "my_rul"}
    assert cfg.pipeline.surfaces.preprocess.target_points == 50000
    assert cfg.pipeline.surfaces.lobe_segments.enabled is False
    assert cfg.pipeline.surfaces.convert.method == "flying-edges"
    from vescan_field.lobes import lobe_mask_names
    assert lobe_mask_names(cfg.pipeline.lobes.enabled, cfg.pipeline.lobes.names) == \
        {"RUL": "my_rul", "RML": "lung_middle_lobe_right"}
