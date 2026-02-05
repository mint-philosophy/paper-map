#!/usr/bin/env python3
"""
Export corpus data from LanceDB for paper map visualization.
"""

import sys
import numpy as np
import pandas as pd
import lancedb
from pathlib import Path

# Paths
LANCEDB_PATH = Path(__file__).parent.parent / "research/corpus-search/data/lancedb"
OUTPUT_PATH = Path(__file__).parent / "data/corpus_data.parquet"


def main():
    print(f"Connecting to LanceDB at {LANCEDB_PATH}")
    db = lancedb.connect(str(LANCEDB_PATH))

    # Load documents table
    docs = db.open_table("documents")
    print(f"Loading {docs.count_rows()} documents...")

    df = docs.to_pandas()

    # Load cluster labels
    clusters = db.open_table("clusters").to_pandas()
    cluster_map = dict(zip(clusters['cluster_id'], clusters['label']))

    # Add cluster label column
    df['cluster_label'] = df['cluster_id'].map(cluster_map)
    df['cluster_label'] = df['cluster_label'].fillna('Uncategorized')

    # Select columns for visualization
    cols = [
        'document_id',
        'title',
        'authors',
        'year',
        'abstract',
        'document_summary',
        'drive_url',
        'macro_category',
        'cluster_label',
        'micro_topic_name',
        'document_embedding'
    ]

    # Filter to columns that exist
    cols = [c for c in cols if c in df.columns]
    df_export = df[cols].copy()

    # Clean up data
    df_export['macro_category'] = df_export['macro_category'].fillna('Uncategorized')
    df_export['year'] = pd.to_numeric(df_export['year'], errors='coerce')
    df_export.loc[df_export['year'] > 2030, 'year'] = np.nan  # Fix bad years
    df_export['year'] = df_export['year'].fillna(2024).astype(int)

    # Filter to documents with embeddings
    has_embedding = df_export['document_embedding'].apply(
        lambda x: x is not None and len(x) > 0 if hasattr(x, '__len__') else False
    )
    df_export = df_export[has_embedding].copy()

    # Remove duplicates by document_id (keep first)
    n_before = len(df_export)
    df_export = df_export.drop_duplicates(subset='document_id', keep='first')
    n_after = len(df_export)
    if n_before != n_after:
        print(f"Removed {n_before - n_after} duplicate document_ids")

    print(f"Exporting {len(df_export)} documents with embeddings")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_export.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

    # Print summary stats
    print(f"\nSummary:")
    print(f"  Documents: {len(df_export)}")
    print(f"  Macro categories: {df_export['macro_category'].nunique()}")
    print(f"  Cluster labels: {df_export['cluster_label'].nunique()}")
    print(f"  Year range: {df_export['year'].min()}-{df_export['year'].max()}")
    print(f"  With drive_url: {df_export['drive_url'].notna().sum()}")


if __name__ == "__main__":
    main()
