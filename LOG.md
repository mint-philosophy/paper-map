
## 2026-03-03 | TerMinty-55f83964

**Work**: Fixed b"" @font-face CSS bug (datamapplot writes Python bytes repr into HTML). Changed label fontWeight from 600 to 400 (normal). Guide page iframe now points at local mintresearch.org copy instead of this repo.
**Decisions**: Font fix applied here for standalone deployment, but primary paper map is now served from mintresearch.org/public/paper-map/.
**State**: Live at mint-philosophy.github.io/paper-map/. Guide page no longer loads from this URL.
**Session**: `3403d776-fc03-404d-8d56-d77855f83964`

## 2026-03-02 | TerMinty-97e1fac9

**Work**: Recovered hung session c0ca2e6b via SpecStory API. Re-clustered full corpus (1891 to 1996 papers, 28 clusters) using fixed `recluster_corpus.py`. Eliminated "Other" category (21 micro-topics reassigned). Added discipline classification (15 buckets). Built interactive filter panel with 3-way toggle (Research Areas / Both / Disciplines). Backfilled 113 embeddings. Queued 139 papers for 41q/RAPTOR re-ingestion via corpus-ingest.
**Decisions**: k=28 from silhouette sweep. "Other" remap via Opus classification. 3-way toggle with AND intersection for combined filters. Re-ingestion (pull from DB, re-ingest via daemon) rather than in-place backfill for 41q/RAPTOR.
**State**: Live at mint-philosophy.github.io/paper-map/ with 1,996 papers. 139 papers queued for re-ingestion. 8 papers need PDFs manually sourced. Re-export needed after re-ingestion completes.
**Session**: `715ac35b-3277-4c9c-a2d6-608197e1fac9`

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
