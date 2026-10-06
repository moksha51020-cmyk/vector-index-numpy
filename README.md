# IVF Vector Index from Scratch

A lightweight, standalone Inverted File (IVF-Flat) Vector Index engine implemented in pure Python using NumPy. Designed to perform fast Approximate Nearest Neighbors (ANN) search over dense vector embeddings without third-party vector database dependencies.

---

## Overview

Exhaustive linear search ($O(N)$) across high-dimensional vector spaces degrades rapidly as dataset sizes grow. This project implements vector quantization via k-means clustering to partition Euclidean space into Voronoi cells:

1. **Training & Quantization:** Computes representative cluster centroids using iterative batch updates.
2. **Inverted File Partitioning:** Maps vector IDs into distinct centroid buckets.
3. **Pruned Search:** Given a query vector, it evaluates distance against only the top $k$ closest centroids (`n_probe`), querying a small fraction of the total dataset.
4. **Serialization:** Provides custom binary serialization and deserialization routines to persist the index structure to disk.

---

## Features

- **Pure NumPy Vectorization:** All distance and cosine similarity routines avoid Python loops for matrix math.
- **Adjustable Accuracy/Latency Trade-off:** Tunable `n_probe` and `n_clusters` parameters to balance recall and query latency.
- **Disk Persistence:** Binary export and loading support (`.save()` / `.load()`).
- **Standardized API:** Straightforward interface patterned after production indexing libraries.

---

## Quickstart

### Prerequisites

- Python 3.10+
- `numpy`

