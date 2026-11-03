"""Store decoding: the levels() table and deficits from synthetic ranks/support."""
import numpy as np

from vescan_field.store import decode_deficit, levels

META = dict(clip=8.0, gap_curve="log", gap_range=64.0, gap_origin=0.5)


def byte_of_gap(gap, meta):
    """Inverse of levels (as rankfield.byte_of_gap, log curve), rounded to the byte."""
    rng, origin = meta["gap_range"], meta["gap_origin"]
    u = 1.0 - np.log1p(gap / origin) / np.log1p(rng / origin)
    return np.round(np.clip(u, 0, 1) * 255).astype(np.uint8)


def test_levels_ends_and_monotone():
    lut = levels(META)
    assert lut.dtype == np.float32 and lut.shape == (256,)
    assert lut[255] == 0.0                         # tied with the winner
    assert np.isclose(lut[0], 64.0)                # the end of the curve
    assert np.all(np.diff(lut) < 0)
    uni = levels(dict(clip=8.0))                   # format 0.2: uniform over the clip
    assert np.isclose(uni[0], 8.0) and np.isclose(uni[255], 0.0)


def test_decode_deficit_matches_logits():
    rng = np.random.default_rng(1)
    K, shape, depth = 5, (6, 7, 8), 3
    logits = rng.normal(0, 4, (K,) + shape).astype(np.float32)
    order = np.argsort(-logits, 0)                 # classes in rank order
    top = np.take_along_axis(logits, order, 0)
    gaps = top[0] - top                            # gap from the winner, >= 0
    ranks = (order[:depth] + 1).astype(np.uint8)
    support = byte_of_gap(gaps[1:depth], META)
    lut = levels(META)
    true_def = logits - logits.max(0)
    for k in range(K):
        d = decode_deficit(ranks, support, k, lut, META["clip"])
        named = (ranks == k + 1).any(0)
        assert np.all(d[~named] == -8.0)           # not named: the clip
        g = -true_def[k][named]
        err = np.abs(d[named] - true_def[k][named])
        # half a step of the log curve at gap g: (origin + g) * log1p(range/origin) / (2*255)
        half_step = (0.5 + g) * np.log1p(64.0 / 0.5) / (2 * 255)
        assert np.all(err <= 1.01 * half_step + 1e-5)
        assert np.all(d[ranks[0] == k + 1] == 0.0)
