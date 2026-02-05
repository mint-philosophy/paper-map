#!/usr/bin/env python3
"""
Compute UMAP projection of document embeddings.
"""

import numpy as np
import pandas as pd
import umap
from pathlib import Path

# Paths
DATA_PATH = Path(__file__).parent / "data"
INPUT_PATH = DATA_PATH / "corpus_data.parquet"
OUTPUT_PATH = DATA_PATH / "umap_coords.npy"


def main():
    print(f"Loading data from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} papers")

    # Extract embeddings as numpy array
    print("Extracting embeddings...")
    embeddings = np.array(df['document_embedding'].tolist())
    print(f"Embedding shape: {embeddings.shape}")

    # Run UMAP
    print("Running UMAP (this may take a few minutes)...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        verbose=True
    )

    coords = reducer.fit_transform(embeddings)
    print(f"Output shape: {coords.shape}")

    # Save
    np.save(OUTPUT_PATH, coords)
    print(f"Saved to {OUTPUT_PATH}")

    # Also save document IDs for alignment
    doc_ids = df['document_id'].tolist()
    np.save(DATA_PATH / "document_ids.npy", doc_ids)
    print(f"Saved document IDs to {DATA_PATH / 'document_ids.npy'}")


if __name__ == "__main__":
    main()
