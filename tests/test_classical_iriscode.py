import numpy as np

from experiments.evaluate_classical import iris_code_similarities, iris_code_similarity, log_gabor_iris_code


def test_log_gabor_code_shape_and_type():
    polar = np.tile(np.arange(64, dtype=np.uint8), (32, 1))
    code, mask = log_gabor_iris_code(polar)
    assert code.shape == (2, 32, 64)
    assert mask.shape == code.shape
    assert code.dtype == np.bool_
    assert mask.dtype == np.bool_


def test_rotation_compensation_recovers_shifted_code():
    rng = np.random.default_rng(7)
    polar = rng.integers(0, 256, size=(16, 64), dtype=np.uint8)
    code, mask = log_gabor_iris_code(polar)
    shifted_code = np.roll(code, 4, axis=-1)
    shifted_mask = np.roll(mask, 4, axis=-1)
    assert iris_code_similarity(code, mask, shifted_code, shifted_mask, max_rotation=4) == 1.0


def test_vectorized_scores_match_scalar_scores():
    rng = np.random.default_rng(11)
    encoded = [log_gabor_iris_code(rng.integers(0, 256, size=(8, 32), dtype=np.uint8)) for _ in range(4)]
    codes = np.stack([item[0] for item in encoded])
    masks = np.stack([item[1] for item in encoded])
    pair_a = np.array([0, 0, 1], dtype=np.int32)
    pair_b = np.array([1, 2, 3], dtype=np.int32)
    vectorized = iris_code_similarities(codes, masks, pair_a, pair_b, max_rotation=2, chunk_size=2)
    scalar = np.array(
        [iris_code_similarity(codes[a], masks[a], codes[b], masks[b], max_rotation=2) for a, b in zip(pair_a, pair_b)]
    )
    np.testing.assert_allclose(vectorized, scalar, atol=1e-7)
