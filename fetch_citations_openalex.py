#!/usr/bin/env python3
"""
Fetch citation counts from OpenAlex API.
OpenAlex is free, has good coverage, and no aggressive rate limiting.
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
OUTPUT_PATH = DATA_PATH / "citations_openalex.json"

# OpenAlex API
OPENALEX_API = "https://api.openalex.org/works"
# Polite pool - add email for faster rate limits
HEADERS = {"User-Agent": "mailto:mintlabjhu@gmail.com"}
RATE_LIMIT = 0.1  # 10 requests/sec is allowed with polite pool


def search_paper(title: str, authors: str = None, year: int = None) -> dict:
    """Search for a paper and return citation info."""
    # Clean title for search
    clean_title = title.strip()[:300]

    # Build search query
    params = {
        "search": clean_title,
        "select": "id,title,cited_by_count,publication_year,authorships",
        "per_page": 1
    }

    # Add year filter if available
    if year and year > 1900 and year < 2030:
        params["filter"] = f"publication_year:{year}"

    try:
        resp = requests.get(OPENALEX_API, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("results"):
            paper = data["results"][0]
            return {
                "found": True,
                "citation_count": paper.get("cited_by_count", 0),
                "openalex_id": paper.get("id"),
                "openalex_title": paper.get("title"),
                "openalex_year": paper.get("publication_year")
            }
    except Exception as e:
        pass

    return {"found": False, "citation_count": 0}


def main():
    print(f"Loading data from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} papers")

    # Load existing cache if any
    citations = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            citations = json.load(f)
        print(f"Loaded {len(citations)} cached citations")

    # Find papers to fetch
    to_fetch = [
        (row['document_id'], row['title'], row.get('authors', ''), row.get('year'))
        for _, row in df.iterrows()
        if row['document_id'] not in citations
    ]

    print(f"Fetching citations for {len(to_fetch)} papers from OpenAlex...")

    # Fetch with progress bar
    for doc_id, title, authors, year in tqdm(to_fetch, desc="Fetching"):
        if not title:
            citations[doc_id] = {"found": False, "citation_count": 0}
            continue

        result = search_paper(title, authors, year)
        citations[doc_id] = result

        # Save periodically
        if len(citations) % 100 == 0:
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(citations, f, indent=2)

        time.sleep(RATE_LIMIT)

    # Final save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(citations, f, indent=2)

    # Stats
    found = sum(1 for v in citations.values() if v.get("found"))
    total_cites = sum(v.get("citation_count", 0) for v in citations.values())

    # Top cited papers
    top_cited = sorted(
        [(k, v.get("citation_count", 0), v.get("openalex_title", ""))
         for k, v in citations.items() if v.get("found")],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    print(f"\nResults:")
    print(f"  Papers found: {found}/{len(citations)} ({100*found/len(citations):.1f}%)")
    print(f"  Total citations: {total_cites:,}")
    print(f"\nTop 10 most cited:")
    for doc_id, cites, title in top_cited:
        print(f"  {cites:,} - {title[:60]}...")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
