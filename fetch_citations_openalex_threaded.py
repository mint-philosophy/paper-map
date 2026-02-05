#!/usr/bin/env python3
"""
Threaded citation fetch from OpenAlex API.
"""

import json
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Paths
DATA_PATH = Path(__file__).parent / "data"
INPUT_PATH = DATA_PATH / "corpus_data.parquet"
OUTPUT_PATH = DATA_PATH / "citations_openalex.json"

# OpenAlex API
OPENALEX_API = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "mailto:mintlabjhu@gmail.com"}
MAX_WORKERS = 15


def search_paper(args) -> tuple:
    """Search for a paper and return citation info."""
    doc_id, title, year = args

    if not title:
        return doc_id, {"found": False, "citation_count": 0}

    clean_title = str(title).strip()[:300]
    params = {
        "search": clean_title,
        "select": "id,title,cited_by_count,publication_year",
        "per_page": 1
    }
    if year and 1900 < int(year) < 2030:
        params["filter"] = f"publication_year:{int(year)}"

    try:
        resp = requests.get(OPENALEX_API, params=params, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("results"):
                paper = data["results"][0]
                return doc_id, {
                    "found": True,
                    "citation_count": paper.get("cited_by_count", 0),
                    "openalex_id": paper.get("id"),
                    "openalex_title": paper.get("title"),
                    "openalex_year": paper.get("publication_year")
                }
    except Exception as e:
        pass

    return doc_id, {"found": False, "citation_count": 0}


def main():
    print(f"Loading data from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} papers")

    # Load existing cache
    citations = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)
        # Only keep entries that were actually found
        citations = {k: v for k, v in existing.items() if v.get("found")}
        print(f"Loaded {len(citations)} cached (found) citations")

    # Papers to fetch
    papers = [(row['document_id'], row['title'], row.get('year', 0))
              for _, row in df.iterrows()
              if row['document_id'] not in citations]

    print(f"Fetching {len(papers)} papers with {MAX_WORKERS} threads...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(search_paper, p): p for p in papers}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching"):
            doc_id, result = future.result()
            citations[doc_id] = result

            # Save periodically
            if len(citations) % 100 == 0:
                with open(OUTPUT_PATH, 'w') as f:
                    json.dump(citations, f)

    # Final save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(citations, f, indent=2)

    # Stats
    found = sum(1 for v in citations.values() if v.get("found"))
    total_cites = sum(v.get("citation_count", 0) for v in citations.values())

    # Top cited
    top_cited = sorted(
        [(k, v.get("citation_count", 0), v.get("openalex_title", ""))
         for k, v in citations.items() if v.get("found")],
        key=lambda x: x[1],
        reverse=True
    )[:15]

    print(f"\nResults:")
    print(f"  Papers found: {found}/{len(citations)} ({100*found/len(citations):.1f}%)")
    print(f"  Total citations: {total_cites:,}")
    print(f"\nTop 15 most cited:")
    for doc_id, cites, title in top_cited:
        print(f"  {cites:>6,} - {(title or 'Unknown')[:55]}...")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
