#!/usr/bin/env python3
"""
Assign uncategorized papers to their nearest existing cluster.

Uses cosine similarity between document embeddings and cluster centroids.
Does NOT re-run full clustering - just assigns papers to existing clusters.
"""

import lancedb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

LANCEDB_PATH = Path(__file__).parent.parent / "research/corpus-search/data/lancedb"


def compute_cluster_centroids(df: pd.DataFrame) -> dict:
    """Compute centroid embedding for each cluster."""
    centroids = {}

    # Get papers with cluster assignments
    clustered = df[df['cluster_id'].notna() & (df['cluster_id'] >= 0)].copy()

    for cluster_id in clustered['cluster_id'].unique():
        cluster_papers = clustered[clustered['cluster_id'] == cluster_id]

        # Extract embeddings
        embeddings = []
        for _, row in cluster_papers.iterrows():
            emb = row['document_embedding']
            if emb is not None:
                if hasattr(emb, 'tolist'):
                    emb = emb.tolist()
                if isinstance(emb, (list, np.ndarray)) and len(emb) > 0:
                    embeddings.append(np.array(emb))

        if embeddings:
            centroid = np.mean(embeddings, axis=0)
            centroids[int(cluster_id)] = centroid

    return centroids


def find_nearest_cluster(embedding: np.ndarray, centroids: dict) -> tuple[int, float]:
    """Find the cluster with highest cosine similarity to the embedding."""
    if embedding is None or len(embedding) == 0:
        return -1, 0.0

    embedding = np.array(embedding).reshape(1, -1)

    best_cluster = -1
    best_similarity = -1

    for cluster_id, centroid in centroids.items():
        centroid = np.array(centroid).reshape(1, -1)
        sim = cosine_similarity(embedding, centroid)[0][0]

        if sim > best_similarity:
            best_similarity = sim
            best_cluster = cluster_id

    return best_cluster, best_similarity


def main():
    print(f"Connecting to LanceDB at {LANCEDB_PATH}")
    db = lancedb.connect(str(LANCEDB_PATH))

    # Load documents
    docs_table = db.open_table("documents")
    df = docs_table.to_pandas()

    # Load clusters for labels
    clusters_df = db.open_table("clusters").to_pandas()
    cluster_labels = dict(zip(clusters_df['cluster_id'], clusters_df['label']))

    print(f"Total documents: {len(df)}")

    # Find uncategorized papers
    uncategorized = df[df['cluster_id'].isna()].copy()
    print(f"Uncategorized papers: {len(uncategorized)}")

    if len(uncategorized) == 0:
        print("No uncategorized papers found!")
        return

    # Compute cluster centroids from existing assignments
    print("\nComputing cluster centroids...")
    centroids = compute_cluster_centroids(df)
    print(f"Found {len(centroids)} clusters with centroids")

    # Assign each uncategorized paper to nearest cluster
    print("\nAssigning papers to nearest clusters...")
    assignments = []

    for idx, row in uncategorized.iterrows():
        emb = row['document_embedding']
        if emb is not None:
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()

        cluster_id, similarity = find_nearest_cluster(emb, centroids)

        assignments.append({
            'document_id': row['document_id'],
            'title': row['title'],
            'cluster_id': cluster_id,
            'similarity': similarity
        })

    # Show assignment summary
    print("\nAssignment summary:")
    assign_df = pd.DataFrame(assignments)

    for cluster_id in sorted(assign_df['cluster_id'].unique()):
        count = len(assign_df[assign_df['cluster_id'] == cluster_id])
        label = cluster_labels.get(cluster_id, 'Unknown')
        avg_sim = assign_df[assign_df['cluster_id'] == cluster_id]['similarity'].mean()
        print(f"  Cluster {cluster_id} ({label[:30]}): {count} papers (avg sim: {avg_sim:.3f})")

    # Update documents in LanceDB
    print("\nUpdating documents...")

    # Create mapping
    doc_cluster_map = dict(zip(assign_df['document_id'], assign_df['cluster_id']))

    # Update the dataframe
    df['cluster_id'] = df.apply(
        lambda row: doc_cluster_map.get(row['document_id'], row['cluster_id']),
        axis=1
    )

    # Convert to records
    records = []
    for _, row in df.iterrows():
        record = row.to_dict()
        if 'document_embedding' in record and record['document_embedding'] is not None:
            if hasattr(record['document_embedding'], 'tolist'):
                record['document_embedding'] = record['document_embedding'].tolist()
        records.append(record)

    # Recreate table
    db.drop_table("documents")
    db.create_table("documents", records)

    # Verify
    print("\nVerification:")
    new_df = db.open_table("documents").to_pandas()
    still_uncategorized = new_df[new_df['cluster_id'].isna()]
    print(f"  Documents: {len(new_df)}")
    print(f"  With cluster_id: {new_df['cluster_id'].notna().sum()}")
    print(f"  Still uncategorized: {len(still_uncategorized)}")

    # Save assignment log
    log_path = Path(__file__).parent / "data/cluster_assignments.csv"
    assign_df.to_csv(log_path, index=False)
    print(f"\nAssignment log saved to {log_path}")


if __name__ == "__main__":
    main()
