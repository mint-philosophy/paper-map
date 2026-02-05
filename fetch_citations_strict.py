#!/usr/bin/env python3
"""
Strict citation fetch - only accept close title matches.
"""

import json
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from difflib import SequenceMatcher

DATA_PATH = Path(__file__).parent / "data"
INPUT_PATH = DATA_PATH / "corpus_data.parquet"
OUTPUT_PATH = DATA_PATH / "citations_strict.json"

OPENALEX_API = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "mailto:mintlabjhu@gmail.com"}
MAX_WORKERS = 10
MIN_SIMILARITY = 0.7  # Require 70% title similarity


def normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    import re
    t = str(title).lower().strip()
    t = re.sub(r'[^\w\s]', '', t)  # Remove punctuation
    t = ' '.join(t.split())  # Normalize whitespace
    return t


def title_similarity(t1: str, t2: str) -> float:
    """Compare two titles, return similarity ratio."""
    n1 = normalize_title(t1)
    n2 = normalize_title(t2)
    return SequenceMatcher(None, n1, n2).ratio()


def search_paper(args) -> tuple:
    """Search for a paper with strict matching."""
    doc_id, our_title, year = args

    if not our_title:
        return doc_id, {"found": False, "citation_count": 0, "reason": "no_title"}

    clean_title = str(our_title).strip()[:300]
    params = {
        "search": clean_title,
        "select": "id,title,cited_by_count,publication_year",
        "per_page": 5  # Get top 5 results to find best match
    }

    try:
        resp = requests.get(OPENALEX_API, params=params, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])

            # Find best matching result
            best_match = None
            best_sim = 0

            for paper in results:
                oalex_title = paper.get("title", "")
                sim = title_similarity(our_title, oalex_title)

                if sim > best_sim:
                    best_sim = sim
                    best_match = paper

            # Only accept if similarity is high enough
            if best_match and best_sim >= MIN_SIMILARITY:
                return doc_id, {
                    "found": True,
                    "citation_count": best_match.get("cited_by_count", 0),
                    "openalex_id": best_match.get("id"),
                    "openalex_title": best_match.get("title"),
                    "openalex_year": best_match.get("publication_year"),
                    "similarity": round(best_sim, 3)
                }
            else:
                return doc_id, {
                    "found": False,
                    "citation_count": 0,
                    "reason": f"no_match (best_sim={best_sim:.2f})"
                }

    except Exception as e:
        return doc_id, {"found": False, "citation_count": 0, "reason": str(e)}

    return doc_id, {"found": False, "citation_count": 0, "reason": "api_error"}


def main():
    print(f"Loading data from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} papers")

    papers = [(row['document_id'], row['title'], row.get('year', 0))
              for _, row in df.iterrows()]

    print(f"Fetching with strict matching (min similarity {MIN_SIMILARITY})...")

    citations = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(search_paper, p): p for p in papers}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching"):
            doc_id, result = future.result()
            citations[doc_id] = result

            if len(citations) % 100 == 0:
                with open(OUTPUT_PATH, 'w') as f:
                    json.dump(citations, f)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(citations, f, indent=2)

    # Stats
    found = sum(1 for v in citations.values() if v.get("found"))
    total_cites = sum(v.get("citation_count", 0) for v in citations.values())

    top_cited = sorted(
        [(k, v.get("citation_count", 0), v.get("openalex_title", ""), v.get("similarity", 0))
         for k, v in citations.items() if v.get("found")],
        key=lambda x: x[1],
        reverse=True
    )[:15]

    print(f"\nResults:")
    print(f"  Papers matched: {found}/{len(citations)} ({100*found/len(citations):.1f}%)")
    print(f"  Total citations: {total_cites:,}")
    print(f"\nTop 15 most cited (with similarity scores):")
    for doc_id, cites, title, sim in top_cited:
        print(f"  {cites:>6,} (sim={sim:.2f}) - {(title or 'Unknown')[:50]}...")


if __name__ == "__main__":
    main()
