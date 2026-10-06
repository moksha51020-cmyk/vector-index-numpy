import numpy as np
import pytest
from p1 import IVFVectorIndex


@pytest.fixture
def sample_data():
    dim = 16
    n_samples = 80
    np.random.seed(42)
    ids = [f"item_{i}" for i in range(n_samples)]
    vectors = np.random.randn(n_samples, dim).astype(np.float32)
    return dim, ids, vectors


def test_query_before_train_raises_error():
    """Ensure search fails safely if index has not been built."""
    index = IVFVectorIndex(dimension=8, n_clusters=2)
    with pytest.raises(RuntimeError):
        index.query(np.random.randn(8))


def test_insufficient_samples_raises_error():
    """K-means must reject clusters larger than dataset size."""
    index = IVFVectorIndex(dimension=4, n_clusters=10)
    small_vecs = np.random.randn(3, 4)
    with pytest.raises(ValueError):
        index.train_and_build(["a", "b", "c"], small_vecs)


def test_exact_match_retrieval(sample_data):
    """Querying an identical vector should return that vector as top match."""
    dim, ids, vectors = sample_data
    index = IVFVectorIndex(dimension=dim, n_clusters=5)
    index.train_and_build(ids, vectors)

    target_idx = 10
    target_vec = vectors[target_idx]
    target_id = ids[target_idx]

    results = index.query(target_vec, top_k=1, n_probe=5)
    assert len(results) == 1
    assert results[0][0] == target_id
    assert np.isclose(results[0][1], 1.0, atol=1e-4)


def test_serialization_persistence(tmp_path, sample_data):
    """Index should save to and load from disk without data corruption."""
    dim, ids, vectors = sample_data
    index = IVFVectorIndex(dimension=dim, n_clusters=4)
    index.train_and_build(ids, vectors)

    query_vec = np.random.randn(dim)
    original_res = index.query(query_vec, top_k=3, n_probe=2)

    file_path = str(tmp_path / "temp_index.bin")
    index.save(file_path)

    loaded_index = IVFVectorIndex.load(file_path)
    loaded_res = loaded_index.query(query_vec, top_k=3, n_probe=2)

    assert original_res == loaded_res