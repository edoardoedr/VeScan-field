"""Config: defaults, null = default, unknown keys, the centerlines section, the example config."""
import json
from pathlib import Path

import pytest

from vescan_thalweg import config as cfgmod

MINIMAL = {"batch": {"patients_root_dir": "p", "surfaces_root_dir": "s", "results_root_dir": "r"}}


def test_defaults_and_relative_paths(tmp_path):
    cfg = cfgmod.build(MINIMAL, base_dir=tmp_path)
    assert cfg.batch.patients_root_dir == str((tmp_path / "p").resolve())
    c = cfg.pipeline.centerlines
    assert c.bifurcations == "vmtk"
    assert (c.trace.cover_scale, c.trace.cover_constant_mm, c.trace.cost_epsilon_mm) == (1.5, 1.0, 0.1)
    assert c.trace.ridge_passes == 4 and c.trace.prune == "auto" and c.trace.recenter is None
    assert c.trace.root == "inlet"
    assert c.vmtk.min_branch_length == 1.0 and c.point.step_mm == 0.5
    assert not hasattr(cfg.pipeline, "field")
    assert cfg.enabled_structures() == {"artery": ["lung_arteries"], "veins": ["lung_veins"],
                                        "airways": ["lung_airways", "lung_airways_wall"]}


def test_thalweg_defaults_match_the_kernel():
    from thalweg.kernel import medial
    t = cfgmod.TraceConfig()
    assert (t.cover_scale, t.cover_constant_mm, t.cost_epsilon_mm) == (medial.SCALE, medial.CONST, medial.EPS)


def test_null_means_default_and_override():
    data = {**MINIMAL, "pipeline": {"centerlines": {"bifurcations": "point", "trace": {"recenter": None,
                                                                                      "ridge_passes": 1},
                                                    "point": {"step_mm": None}}}}
    c = cfgmod.build(data).pipeline.centerlines
    assert c.bifurcations == "point" and c.trace.ridge_passes == 1 and c.trace.recenter is None
    assert c.point.step_mm == 0.5


@pytest.mark.parametrize("data", [
    {**MINIMAL, "pipeline": {"centerlines": {"bifurcation": "vmtk"}}},              # typo
    {**MINIMAL, "pipeline": {"centerlines": {"bifurcations": "zone"}}},
    {**MINIMAL, "pipeline": {"centerlines": {"trace": {"prune": "spur"}}}},
    {**MINIMAL, "pipeline": {"centerlines": {"trace": {"root": "top"}}}},
    {**MINIMAL, "pipeline": {"centerlines": {"trace": {"connectivity": "graph"}}}},
    {**MINIMAL, "pipeline": {"centerlines": {"trace": {"ridge_passes": 0}}}},
    {**MINIMAL, "pipeline": {"centerlines": {"delta": 2.0}}},                       # a field key
    {**MINIMAL, "pipeline": {"field": {"pad_voxels": 3}}},                          # no field section
    {"batch": {**MINIMAL["batch"], "structures": {"artery": {"classes": []}}}},
])
def test_errors(data):
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.build(data)


def test_example_config_loads():
    path = Path(__file__).resolve().parents[2] / "configs" / "thalweg" / "example.json"
    cfg = cfgmod.load(path)
    assert cfg.batch.enabled is False
    assert json.loads(path.read_text())["pipeline"]["stages"] == {
        "surfaces": True, "centerlines": True, "anatomy": True, "clip": True, "surface_labels": True}
    assert cfg.pipeline.centerlines.bifurcations == "vmtk"
    assert cfg.enabled_structures()["airways"] == ["lung_airways", "lung_airways_wall"]
