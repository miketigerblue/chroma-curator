"""
chroma_curator/core.py

Reusable core functions for chroma-curator:
- Cosine similarity for vectors
- Vector search helpers

Author: Mike Harris (michael.harris@tigerblue.tech)
Date: 2025-05-25

"""

import numpy as np

def cosine_similarity(v1, v2):
    """
    Compute cosine similarity between two vectors (lists or numpy arrays).
    Returns a float in [-1, 1].
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    num = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(num) / float(denom)

def top_similar_vectors(query, records, top_k=5):
    """
    Given a query vector and a list of record dicts (each with a 'vector' and 'id'),
    return the top_k most similar records (highest cosine similarity).
    """
    scored = []
    for rec in records:
        sim = cosine_similarity(query, rec['vector'])
        scored.append((sim, rec))
    # Sort by descending similarity
    scored.sort(reverse=True, key=lambda x: x[0])
    return [rec for sim, rec in scored[:top_k]]

# Optionally add more helpers here (e.g., load_vectors, batch_search, etc.)
