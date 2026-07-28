"""Fix #1: memory-bounded pair scoring must be numerically identical to the
original single-shot dot product. This protects the scientific metrics."""
import numpy as np

from src.metrics import chunked_pair_scores, sample_pair_scores


def _naive_scores(embeds, pair_a, pair_b):
    return (embeds[pair_a] * embeds[pair_b]).sum(axis=1).astype(np.float32)


def test_chunked_equals_naive_for_all_chunk_sizes():
    rng = np.random.default_rng(0)
    embeds = rng.standard_normal((50, 8)).astype(np.float32)
    pair_a = rng.integers(0, 50, size=37)
    pair_b = rng.integers(0, 50, size=37)
    naive = _naive_scores(embeds, pair_a, pair_b)

    for chunk in (1, 7, 37, 100):
        got = chunked_pair_scores(embeds, pair_a, pair_b, chunk=chunk)
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, naive)


def test_chunked_handles_empty_pairs():
    embeds = np.zeros((5, 4), dtype=np.float32)
    empty = np.array([], dtype=int)
    got = chunked_pair_scores(embeds, empty, empty, chunk=8)
    assert got.shape == (0,)
    assert got.dtype == np.float32


def test_sample_pair_scores_returns_dot_products_of_returned_pairs():
    rng = np.random.default_rng(1)
    embeds = rng.standard_normal((40, 16)).astype(np.float32)
    labels = rng.integers(0, 6, size=40)
    out = sample_pair_scores(embeds, labels, n_pairs=200, seed=3, impostor_multiplier=5)
    expected = _naive_scores(embeds, out["pair_a"], out["pair_b"])
    np.testing.assert_array_equal(out["scores"], expected)
