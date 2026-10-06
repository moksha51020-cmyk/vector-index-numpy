"""
Lightweight Vector Index (IVF-Flat) built from scratch using NumPy.
Demonstrates: Vector quantization, Voronoi cells, and cosine similarity.
"""

import pickle
from typing import List, Tuple
import numpy as np


class IVFVectorIndex:

    def __init__(self, dimension: int, n_clusters: int = 4):
        """dimension: Length of each vector (e.g., 128, 512, 1536).

        n_clusters: Number of centroid clusters to partition space into.
        """
        self.dimension = dimension
        self.n_clusters = n_clusters
        self.centroids: np.ndarray = None

        # Hash map storing cluster_id -> list of (item_id, vector)
        self.inverted_index = {i: [] for i in range(n_clusters)}
        self.is_trained = False

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """Normalizes vectors so dot product equals cosine similarity."""
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.clip(norms, a_min=1e-9, a_max=None)

    def train_and_build(self, ids: List[str], vectors: np.ndarray):
        """Performs simple k-means to find cluster centroids,

        then assigns every vector to its closest centroid.
        """
        vectors = self._normalize(np.array(vectors, dtype=np.float32))
        n_samples = len(vectors)

        if n_samples < self.n_clusters:
            raise ValueError("More clusters than data points provided.")

        # 1. Initialize random centroids from the dataset
        rng = np.random.default_rng(42)
        initial_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        self.centroids = vectors[initial_indices].copy()

        # 2. Basic K-Means iterations to stabilize centroids
        for _ in range(10):
            # Compute similarity to all centroids: shape (n_samples, n_clusters)
            similarities = np.dot(vectors, self.centroids.T)
            assignments = np.argmax(similarities, axis=1)

            # Recompute centroids as mean of assigned vectors
            for k in range(self.n_clusters):
                cluster_points = vectors[assignments == k]
                if len(cluster_points) > 0:
                    self.centroids[k] = self._normalize(
                        np.mean(cluster_points, axis=0)
                    )

        # 3. Populate inverted index with vector payloads
        for i in range(self.n_clusters):
            self.inverted_index[i] = []

        similarities = np.dot(vectors, self.centroids.T)
        assignments = np.argmax(similarities, axis=1)

        for idx, (doc_id, vector) in enumerate(zip(ids, vectors)):
            cluster_id = assignments[idx]
            self.inverted_index[cluster_id].append((doc_id, vector))

        self.is_trained = True

    def query(
        self, query_vec: np.ndarray, top_k: int = 3, n_probe: int = 2
    ) -> List[Tuple[str, float]]:
        """Searches only the 'n_probe' nearest clusters instead of the whole database."""
        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching.")

        q = self._normalize(np.array(query_vec, dtype=np.float32)).reshape(1, -1)

        # Step 1: Find closest centroids
        centroid_scores = np.dot(q, self.centroids.T)[0]
        # Pick top 'n_probe' clusters to look into
        best_clusters = np.argsort(centroid_scores)[::-1][:n_probe]

        candidates = []
        for cluster_id in best_clusters:
            candidates.extend(self.inverted_index[cluster_id])

        if not candidates:
            return []

        # Step 2: Search only among candidate vectors
        candidate_ids = [c[0] for c in candidates]
        candidate_matrix = np.array([c[1] for c in candidates])

        scores = np.dot(candidate_matrix, q.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(candidate_ids[i], float(scores[i])) for i in top_indices]

    def save(self, file_path: str):
        """Serializes the index to disk."""
        with open(file_path, "wb") as f:
            pickle.dump((self.dimension, self.centroids, self.inverted_index), f)

    @classmethod
    def load(cls, file_path: str) -> "IVFVectorIndex":
        """Loads an index from disk."""
        with open(file_path, "rb") as f:
            dimension, centroids, inverted_index = pickle.load(f)
        idx = cls(dimension=dimension, n_clusters=len(centroids))
        idx.centroids = centroids
        idx.inverted_index = inverted_index
        idx.is_trained = True
        return idx


# --- Example Test Run ---
if __name__ == "__main__":
    DIM = 8
    np.random.seed(42)

    # 1. Create dummy database vectors
    documents = [f"doc_{i}" for i in range(100)]
    database_vectors = np.random.randn(100, DIM)

    # 2. Build index
    print("Building Vector Index...")
    index = IVFVectorIndex(dimension=DIM, n_clusters=5)
    index.train_and_build(documents, database_vectors)

    # 3. Query the index
    test_query = np.random.randn(DIM)
    results = index.query(test_query, top_k=3, n_probe=2)

    print("\nTop 3 Nearest Matches (ID, Cosine Similarity):")
    for doc_id, score in results:
        print(f"  {doc_id} -> Score: {score:.4f}")

    # 4. Save and verify loading
    index.save("test_index.bin")
    loaded_index = IVFVectorIndex.load("test_index.bin")
    loaded_results = loaded_index.query(test_query, top_k=3, n_probe=2)
    assert results == loaded_results
    print("\nIndex successfully saved, loaded, and verified!")

