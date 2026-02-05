#!/usr/bin/env python3
"""
Assign macro_category to papers based on their cluster's most common category.
"""

import lancedb
import pandas as pd
from pathlib import Path

LANCEDB_PATH = Path(__file__).parent.parent / "research/corpus-search/data/lancedb"


def main():
    print(f"Connecting to LanceDB at {LANCEDB_PATH}")
    db = lancedb.connect(str(LANCEDB_PATH))

    docs_table = db.open_table("documents")
    df = docs_table.to_pandas()

    print(f"Total documents: {len(df)}")

    # Find papers without macro_category
    needs_category = df['macro_category'].isna() | (df['macro_category'] == '')
    print(f"Papers needing macro_category: {needs_category.sum()}")

    if needs_category.sum() == 0:
        print("All papers have macro_category!")
        return

    # Compute most common macro_category per cluster
    cluster_to_macro = {}
    for cluster_id in df['cluster_id'].unique():
        if pd.isna(cluster_id):
            continue

        cluster_papers = df[(df['cluster_id'] == cluster_id) & ~needs_category]

        if len(cluster_papers) == 0:
            # No existing categorized papers in cluster - use cluster label as fallback
            cluster_to_macro[cluster_id] = 'Other'
            continue

        # Get most common macro_category in this cluster
        most_common = cluster_papers['macro_category'].value_counts().idxmax()
        cluster_to_macro[cluster_id] = most_common

    print(f"\nCluster -> macro_category mapping:")
    for cluster_id in sorted(cluster_to_macro.keys()):
        print(f"  {int(cluster_id)}: {cluster_to_macro[cluster_id]}")

    # Assign macro_category to papers
    assigned_count = 0
    for idx, row in df.iterrows():
        if needs_category.loc[idx]:
            cluster_id = row['cluster_id']
            if cluster_id in cluster_to_macro:
                df.at[idx, 'macro_category'] = cluster_to_macro[cluster_id]
                assigned_count += 1
            elif cluster_id == -1:
                df.at[idx, 'macro_category'] = 'Other'
                assigned_count += 1

    print(f"\nAssigned macro_category to {assigned_count} papers")

    # Convert to records
    records = []
    for _, row in df.iterrows():
        record = row.to_dict()
        if 'document_embedding' in record and record['document_embedding'] is not None:
            if hasattr(record['document_embedding'], 'tolist'):
                record['document_embedding'] = record['document_embedding'].tolist()
        records.append(record)

    # Update table
    db.drop_table("documents")
    db.create_table("documents", records)

    # Verify
    new_df = db.open_table("documents").to_pandas()
    still_needs = new_df['macro_category'].isna() | (new_df['macro_category'] == '')
    print(f"\nVerification:")
    print(f"  Still need macro_category: {still_needs.sum()}")

    # Show category distribution
    print(f"\nMacro category distribution:")
    for cat, count in new_df['macro_category'].value_counts().head(10).items():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
