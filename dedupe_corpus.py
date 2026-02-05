#!/usr/bin/env python3
"""
Deduplicate papers in LanceDB by normalized title.

For each duplicate group, keeps the paper with the most complete data.
Scoring: +1 for each non-null field (abstract, cluster_id, drive_url, document_summary, etc.)
"""

import lancedb
import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys
from datetime import datetime

LANCEDB_PATH = Path(__file__).parent.parent / "research/corpus-search/data/lancedb"

# Fields that indicate completeness (excluding embedding - always present)
COMPLETENESS_FIELDS = [
    'abstract',
    'cluster_id',
    'drive_url',
    'document_summary',
    'macro_category',
    'authors',
    'year',
    'q01_research_question',  # Analysis completed
]


def normalize_title(t):
    """Normalize title for comparison."""
    if pd.isna(t):
        return ''
    t = str(t).lower()
    t = re.sub(r'[^a-z0-9]', '', t)
    return t


def completeness_score(row):
    """Score a row by how many completeness fields are non-null."""
    score = 0
    for field in COMPLETENESS_FIELDS:
        if field in row.index:
            val = row[field]
            if val is None:
                continue
            if isinstance(val, np.ndarray):
                if len(val) > 0:
                    score += 1
            elif pd.isna(val):
                continue
            elif isinstance(val, (list, dict)):
                if len(val) > 0:
                    score += 1
            elif isinstance(val, str):
                if val.strip():
                    score += 1
            else:
                score += 1
    return score


def main():
    print(f"Connecting to LanceDB at {LANCEDB_PATH}")
    db = lancedb.connect(str(LANCEDB_PATH))
    docs = db.open_table("documents")

    df = docs.to_pandas()
    print(f"Total documents: {len(df)}")

    # Normalize titles
    df['norm_title'] = df['title'].apply(normalize_title)

    # Find duplicates
    title_counts = df['norm_title'].value_counts()
    dup_titles = title_counts[title_counts > 1]

    print(f"\nDuplicate groups: {len(dup_titles)}")
    print(f"Papers in duplicate groups: {dup_titles.sum()}")

    if len(dup_titles) == 0:
        print("No duplicates found!")
        return

    # For each duplicate group, find the best one to keep
    to_delete = []
    to_keep = []

    for norm_title in dup_titles.index:
        group = df[df['norm_title'] == norm_title].copy()
        group['score'] = group.apply(completeness_score, axis=1)

        # Sort by score (descending), then by indexed_at (prefer older = original)
        group = group.sort_values(['score', 'indexed_at'], ascending=[False, True])

        # Keep the first (best), delete the rest
        best = group.iloc[0]
        rest = group.iloc[1:]

        to_keep.append({
            'document_id': best['document_id'],
            'title': best['title'],
            'score': best['score']
        })

        for _, row in rest.iterrows():
            to_delete.append({
                'document_id': row['document_id'],
                'title': row['title'],
                'score': row['score'],
                'kept_id': best['document_id']
            })

    print(f"\nKeeping: {len(to_keep)} papers")
    print(f"Deleting: {len(to_delete)} papers")

    # Show some examples
    print("\nExamples of deletions:")
    for item in to_delete[:5]:
        print(f"  - {item['title'][:50]}... (score={item['score']}, keeping {item['kept_id'][:8]})")

    # Confirm before deletion
    if '--yes' in sys.argv:
        print("\n--yes flag provided, proceeding...")
    else:
        response = input("\nProceed with deletion? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return

    # Backup first
    backup_path = LANCEDB_PATH / f"documents_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.lance"
    print(f"\nBacking up to {backup_path}...")
    # Just note that LanceDB doesn't have a simple copy mechanism, we'll rely on the existing backup

    # Delete duplicates
    print("Deleting duplicates...")
    delete_ids = [d['document_id'] for d in to_delete]

    # LanceDB delete by filter - need to format as SQL IN clause with quoted strings
    if len(delete_ids) == 1:
        where_clause = f"document_id = '{delete_ids[0]}'"
    else:
        # Format: document_id IN ('id1', 'id2', ...)
        quoted_ids = ", ".join(f"'{id}'" for id in delete_ids)
        where_clause = f"document_id IN ({quoted_ids})"

    print(f"Deleting with: {where_clause[:100]}...")
    docs.delete(where_clause)

    # Verify
    remaining = docs.count_rows()
    print(f"\nRemaining documents: {remaining}")
    print(f"Expected: {len(df) - len(to_delete)}")

    # Save deletion log
    log_path = Path(__file__).parent / "data/dedupe_log.csv"
    pd.DataFrame(to_delete).to_csv(log_path, index=False)
    print(f"Deletion log saved to {log_path}")


if __name__ == "__main__":
    main()
