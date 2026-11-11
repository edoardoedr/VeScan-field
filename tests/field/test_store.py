"""Store: deficits of a box decoded with rankfield, on synthetic ranks/support and on the real store."""
from pathlib import Path

import numpy as np
import pytest
import rankfield as rf
from rankfield.code import RankField

from vescan_field.store import Store, box_deficits

META = dict(mode="ranked", version="0.3", classes=5, depth=3, clip=8.0, gap_curve="log", gap_range=64.0,
            gap_origin=0.5)
STORE = Path(__file__).resolve().parents[2] / "Data" / "PD-1-Lung-00043" / "lung_vessels.duckn.zip"


def synthetic(seed=1, shape=(6, 7, 8)):
    """(RankField, true deficits) from random logits: ranks of the top `depth` classes, support
    bytes of their gaps from the winner."""
    rng = np.random.default_rng(seed)
    K, depth = META["classes"], META["depth"]
    logits = rng.normal(0, 4, (K,) + shape).astype(np.float32)
    order = np.argsort(-logits, 0)
    top = np.take_along_axis(logits, order, 0)
    ranks = (order[:depth] + 1).astype(np.uint8)
    support = rf.byte_of_gap(top[0] - top[1:depth], META).astype(np.uint8)
    code = RankField(ranks=ranks, support=support, tail=None, meta=dict(META, shape=list(shape)),
                     labels=list(range(K)))
    return code, logits - logits.max(0)


def test_box_deficits_match_the_logits():
    code, true_def = synthetic()
    d = box_deficits(code, (0, 0, 0), code.support.shape[1:])
    assert d.dtype == np.float32 and d.shape == true_def.shape
    for k in range(META["classes"]):
        named = (code.ranks == k + 1).any(0)
        assert np.all(d[k][~named] == -8.0)                    # not named: the clip
        assert np.all(d[k][code.ranks[0] == k + 1] == 0.0)     # the winner
        g = -true_def[k][named]
        # rankfield.byte_of_gap quantizes toward the larger gap: the decoded deficit is at most one
        # step of the log curve below the true one; a step at gap g: (origin + g) * log1p(range/origin) / 255
        step = (0.5 + g) * np.log1p(64.0 / 0.5) / 255
        err = true_def[k][named] - d[k][named]
        assert np.all(err >= -1e-5) and np.all(err <= 1.01 * step + 1e-5)


def test_a_box_is_the_whole_field_cut():
    """The box's deficits are rankfield's deficits of the whole field, cut to the box (the meta's
    whole-grid shape does not get in the way)."""
    code, _ = synthetic(seed=2, shape=(9, 10, 11))
    lo, hi = (2, 1, 3), (7, 10, 8)
    d = box_deficits(code, lo, hi)
    for k in range(META["classes"]):
        np.testing.assert_array_equal(d[k], rf.deficit(code, k)[2:7, 1:10, 3:8])


@pytest.mark.skipif(not STORE.exists(), reason="PD-1-Lung-00043 store not present")
def test_real_store():
    st = Store(STORE)
    try:
        assert set(st.class_index) >= {"lung_arteries", "lung_veins", "lung_airways", "lung_airways_wall"}
        c = st.class_index["lung_arteries"]
        lo, hi = st.class_box(c, 3)
        d = st.deficits(lo, hi)
        assert d.shape == (st.K,) + tuple(int(v) for v in hi - lo)
        win = st.labelmap(lo, hi) == c
        assert np.all(d[c][win] == 0.0) and np.all(d[c][~win] <= 0)    # 0 behind: an exact tie
        q = np.array([[10.0, 20.0, 30.0], [100.5, 150.25, 200.0]])
        np.testing.assert_allclose(st.lps_to_index(st.index_to_lps(q)), q, atol=1e-9)
    finally:
        st.close()
