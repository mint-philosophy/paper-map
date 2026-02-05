#!/usr/bin/env python3
"""
Fetch citation counts from Semantic Scholar API.
"""

import json
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from urllib.parse import quote

# Paths
DATA_PATH = Path(__file__).parent / "data"
INPUT_PATH = DATA_PATH / "corpus_data.parquet"
OUTPUT_PATH = DATA_PATH / "citations.json"

# Semantic Scholar API
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
RATE_LIMIT = 0.15  # seconds between requests (conservative)


def search_paper(title: str, authors: str = None) -> dict:
    """Search for a paper and return citation info."""
    # Clean title for search
    clean_title = title.strip()[:200]  # API has length limits

    params = {
        "query": clean_title,
        "fields": "citationCount,title,authors,year",
        "limit": 1
    }

    try:
        resp = requests.get(S2_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("data"):
            paper = data["data"][0]
            return {
                "found": True,
                "citation_count": paper.get("citationCount", 0),
                "s2_title": paper.get("title"),
                "s2_year": paper.get("year")
            }
    except Exception as e:
        pass

    return {"found": False, "citation_count": 0}


def main():
    print(f"Loading data from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} papers")

    # Check for existing cache
    citations = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            citations = json.load(f)
        print(f"Loaded {len(citations)} cached citations")

    # Find papers to fetch
    to_fetch = [
        (row['document_id'], row['title'], row.get('authors', ''))
        for _, row in df.iterrows()
        if row['document_id'] not in citations
    ]

    print(f"Fetching citations for {len(to_fetch)} papers...")

    # Fetch with progress bar
    for doc_id, title, authors in tqdm(to_fetch, desc="Fetching"):
        if not title:
            citations[doc_id] = {"found": False, "citation_count": 0}
            continue

        result = search_paper(title, authors)
        citations[doc_id] = result

        # Save periodically
        if len(citations) % 50 == 0:
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(citations, f, indent=2)

        time.sleep(RATE_LIMIT)

    # Final save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(citations, f, indent=2)

    # Stats
    found = sum(1 for v in citations.values() if v.get("found"))
    total_cites = sum(v.get("citation_count", 0) for v in citations.values())

    print(f"\nResults:")
    print(f"  Papers found: {found}/{len(citations)}")
    print(f"  Total citations: {total_cites:,}")
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
