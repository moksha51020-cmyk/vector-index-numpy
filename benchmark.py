import time
import numpy as np
from p1 import IVFVectorIndex

DIM = 64
NUM_VECTORS = 5000
N_CLUSTERS = 20

print(f"Generating {NUM_VECTORS} vectors of dimension {DIM}...")
np.random.seed(42)
ids = [f"item_{i}" for i in range(NUM_VECTORS)]
vectors = np.random.randn(NUM_VECTORS, DIM).astype(np.float32)

# 1. Build IVF Index
index = IVFVectorIndex(dimension=DIM, n_clusters=N_CLUSTERS)
index.train_and_build(ids, vectors)

query_vec = np.random.randn(DIM).astype(np.float32)

# Benchmark: IVF Query
start = time.perf_counter()
ivf_results = index.query(query_vec, top_k=5, n_probe=3)
ivf_time = (time.perf_counter() - start) * 1000

# Benchmark: Exhaustive Brute-Force Search
start = time.perf_counter()
q_norm = query_vec / np.linalg.norm(query_vec)
db_norms = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
all_scores = np.dot(db_norms, q_norm)
brute_top_indices = np.argsort(all_scores)[::-1][:5]
brute_time = (time.perf_counter() - start) * 1000

print(f"\n--- Benchmark Results ({NUM_VECTORS} Vectors) ---")
print(f"Brute-Force Search    : {brute_time:.3f} ms")
print(f"IVF Search (n_probe=3): {ivf_time:.3f} ms")
print(f"Speedup Factor        : {brute_time / ivf_time:.2f}x faster")