"""
Filename:  cli.py

Profile a ChromaDB and export a curated, edge-ready set of vectors and metadata
for use in on-device ML/AI applications (e.g., iOS, Core ML, vector search).

Requirements:

    pip install chromadb pandas numpy

Author: Mike Harris (mike@tigerblue.tech)
Date: 2023-10-01

"""

import chromadb
import numpy as np
import pandas as pd
import json
from collections import Counter

# ==============================
# ChromaDB Connection Utilities
# ==============================

def connect_chromadb(persist_dir="./chroma"):
    """
    Connect to a ChromaDB persistent store at the given directory.
    Returns a chromadb.PersistentClient instance.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    return client

# ==============================
# Collection Profiling
# ==============================

def profile_collection(collection):
    """
    Profiles a ChromaDB collection:
        - Fetches all vectors, metadatas, documents, and ids
        - Calculates basic statistics (counts, completeness, norms)
        - Checks for duplicates, field coverage, doc lengths, etc.
    Returns:
        - profile: dict with summary stats
        - df: pandas DataFrame of metadata + helper columns
        - embeddings: numpy array of all vectors
        - docs: list of raw document strings
        - ids: list of item ids
    """
    n = collection.count()
    print(f"Profiling {n} records in '{collection.name}'...")
    if n == 0:
        return {}, pd.DataFrame(), np.array([]), [], []

    # Fetch all records from ChromaDB (consider batching if huge)
    all_items = collection.get(
        include=['embeddings', 'metadatas', 'documents']
    )
    ids        = all_items['ids']
    embeddings = np.array(all_items['embeddings'])
    metas      = all_items['metadatas']
    docs       = all_items['documents']

    # Build DataFrame from metadata
    df = pd.DataFrame(metas)
    df['id'] = ids

    # Compute embedding norm for stats (helps detect anomalies)
    if embeddings.ndim == 2:
        df['embedding_norm'] = np.linalg.norm(embeddings, axis=1)

    # Document existence and length stats
    df['has_doc'] = [d is not None and len(str(d)) > 0 for d in docs]
    df['doc_len'] = [len(str(d)) if d else 0 for d in docs]

    # Build a profile summary dictionary
    profile = {
        "num_records": n,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "field_completeness": df.notnull().mean().to_dict(),  # fraction non-null for each field
        "has_document_pct": float(df['has_doc'].mean()),
        "doc_length_stats": df['doc_len'].describe().to_dict(),
        "embedding_norm_stats": df['embedding_norm'].describe().to_dict() if 'embedding_norm' in df else {},
        "fields": list(df.columns),
        "top_fields": {
            col: df[col].value_counts().head(5).to_dict()
            for col in df.columns if df[col].dtype == 'O'
        },
        "unique_ids": len(set(ids)),
        "duplicate_ids": [k for k, v in Counter(ids).items() if v > 1],
    }

    # Pretty-print profile
    print("\n--- Profile Summary ---")
    print(json.dumps(profile, indent=2))
    print("----------------------\n")

    return profile, df, embeddings, docs, ids

# ==============================
# Smart Export for Edge ML
# ==============================

def export_for_edge(df, embeddings, docs, ids, top_n=2048, key_fields=None):
    """
    Exports the 'top_n' most informative, recent, or diverse entries for on-device use.
    Filters for deduplication, document existence, and sorts by doc length (as a proxy for content).
    Writes export to 'export_for_edge.json'.
    """
    if key_fields is None:
        key_fields = ['id', 'title', 'summary', 'cve_id', 'published', 'source', 'severity']

    # Remove duplicates (by id) and keep only one
    df = df.drop_duplicates(subset=['id'])

    # Attempt to sort by published date, else by document length
    if 'published' in df.columns:
        try:
            df['published_dt'] = pd.to_datetime(df['published'], errors='coerce')
            df = df.sort_values('published_dt', ascending=False)
        except Exception:
            pass

    # Secondary sort: longest documents first (most informative)
    df = df.sort_values('doc_len', ascending=False)

    # Select the top N entries for export
    selected = df.head(top_n)

    # Build export list
    export_list = []
    for idx, row in selected.iterrows():
        entry = {k: row.get(k, None) for k in key_fields if k in row}
        i = list(df.index).index(idx)  # Map index to correct position
        entry['vector'] = embeddings[i].tolist()
        entry['document'] = docs[i]
        export_list.append(entry)

    print(f"Exporting {len(export_list)} records to edge (JSON format).")

    # Write to JSON file
    with open("export_for_edge.json", "w") as f:
        json.dump(export_list, f, indent=2)

    return export_list

# ==============================
# Main Entry Point
# ==============================

def main():
    """
    1. Connects to your ChromaDB persistent store (edit path as needed)
    2. Lists available collections and profiles the first (edit as needed)
    3. Prints and saves a profile summary for human review
    4. Exports a curated dataset for edge ML use (for iOS, etc)
    """
    # Connect to your persistent chroma DB directory
    client = connect_chromadb("./chroma")  # <-- edit path as needed

    # List all collections and use the first one (or change to your main collection)
    collections = client.list_collections()
    print("Collections available:", [c.name for c in collections])
    collection = client.get_collection(name=collections[0].name)

    # Profile the selected collection
    profile, df, embeddings, docs, ids = profile_collection(collection)

    # Only export records with real documents
    export_for_edge(df[df['has_doc']], embeddings, docs, ids, top_n=2048)

    # Save profile for future reference
    with open("chroma_profile.json", "w") as pf:
        json.dump(profile, pf, indent=2)

# ==============================
# Run if main
# ==============================

if __name__ == "__main__":
    main()
