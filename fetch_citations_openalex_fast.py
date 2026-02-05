#!/usr/bin/env python3
"""
Fast parallel citation fetch from OpenAlex API.
Uses concurrent requests for speed.
"""

import json
import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
from tqdm.asyncio import tqdm

# Paths
DATA_PATH = Path(__file__).parent / "data"
INPUT_PATH = DATA_PATH / "corpus_data.parquet"
OUTPUT_PATH = DATA_PATH / "citations_openalex.json"

# OpenAlex API
OPENALEX_API = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "mailto:mintlabjhu@gmail.com"}
CONCURRENT_REQUESTS = 20  # OpenAlex allows high concurrency with polite pool


async def search_paper(session: aiohttp.ClientSession, doc_id: str, title: str, year: int = None) -> tuple:
    """Search for a paper and return citation info."""
    if not title:
        return doc_id, {"found": False, "citation_count": 0}

    clean_title = title.strip()[:300]
    params = {
        "search": clean_title,
        "select": "id,title,cited_by_count,publication_year",
        "per_page": 1
    }
    if year and 1900 < year < 2030:
        params["filter"] = f"publication_year:{year}"

    try:
        async with session.get(OPENALEX_API, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                data = await resp.json()
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


async def fetch_all(papers: list) -> dict:
    """Fetch citations for all papers concurrently."""
    citations = {}

    # Load existing cache
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            citations = json.load(f)
        print(f"Loaded {len(citations)} cached citations")

    # Filter to papers not yet fetched
    to_fetch = [(doc_id, title, year) for doc_id, title, year in papers if doc_id not in citations]
    print(f"Fetching {len(to_fetch)} papers...")

    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        # Create tasks
        tasks = [search_paper(session, doc_id, title, year) for doc_id, title, year in to_fetch]

        # Run with progress bar
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching"):
            result = await coro
            results.append(result)

            # Save periodically
            if len(results) % 100 == 0:
                for doc_id, data in results:
                    citations[doc_id] = data
                with open(OUTPUT_PATH, 'w') as f:
                    json.dump(citations, f)
                results = []

        # Final update
        for doc_id, data in results:
            citations[doc_id] = data

    return citations


def main():
    print(f"Loading data from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} papers")

    papers = [(row['document_id'], row['title'], row.get('year')) for _, row in df.iterrows()]

    citations = asyncio.run(fetch_all(papers))

    # Save final
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
    )[:10]

    print(f"\nResults:")
    print(f"  Papers found: {found}/{len(citations)} ({100*found/len(citations):.1f}%)")
    print(f"  Total citations: {total_cites:,}")
    print(f"\nTop 10 most cited:")
    for doc_id, cites, title in top_cited:
        print(f"  {cites:>6,} - {title[:55]}...")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
