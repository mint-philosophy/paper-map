
## 2026-02-05 | TerMinty-5b5d

**Work**: Built interactive paper map visualization of MINT corpus (1,352 papers) using datamapplot. Deployed to GitHub Pages at mint-philosophy.github.io/paper-map. Features: UMAP projection, hierarchical labels, search by title+author, click-to-Drive links.
**Decisions**: Used OpenAlex for citations but discovered massive mismatches (3% accuracy) — removed citation sizing. Used custom JS to fix datamapplot click handler bug.
**State**: Live and working. Next agent: dedupe 95 duplicate groups, re-cluster 217 uncategorized papers, ingest Shen & Tamkin 2026.
**Session**: `20260205-195631-5b5d` | Claude: check session json

## 2026-02-05 | TerMinty-4786

**Work**: Deduplicated corpus (116 papers removed, 1249 remaining) and assigned clusters/categories to 147 uncategorized papers. Regenerated paper map with 1,240 papers.
**Scripts**: Created `dedupe_corpus.py`, `assign_clusters.py`, `assign_macro_category.py`
**Decisions**: Used completeness scoring (abstract, cluster_id, drive_url, etc.) to pick best paper from duplicate groups. Assigned uncategorized papers to nearest cluster centroid by cosine similarity, then propagated macro_category from most common in cluster.
**State**: Map live with clean data. Still need: ingest Shen & Tamkin 2026.
**Session**: `20260205-225459-4786`
