# MINT Lab Paper Map

Interactive visualization of the MINT Lab research corpus using datamapplot.

**Live**: https://mint-philosophy.github.io/paper-map/

## Features

- 1,352 papers visualized with UMAP projection
- Colored by macro_category (22 categories)
- Hierarchical labels (macro → cluster → title on zoom)
- Click to open Google Drive PDF
- Search by title + author

## Known Issues (for next agent)

### 1. Database Duplicates
**95 duplicate title groups (208 papers total)** in LanceDB from multiple indexing runs.

- 18 duplicate file_paths
- 80 duplicate file_hashes
- 95 duplicate normalized titles

**Fix**: Dedupe by file_hash (safest), then regenerate map.

### 2. Uncategorized Papers
**217 papers** have no cluster assignment (macro_category = null).
**136 papers** assigned to generic "Other" category.

These were likely added after the HDBSCAN clustering was run.

**Fix**: Re-run clustering on full corpus, or manually assign categories.

### 3. New Paper to Ingest
`Shen and Tamkin (2026) How AI Impacts Skill Formation.pdf` in Resources/pdfs/ needs ingestion.

## Scripts

- `export_corpus_data.py` — Export from LanceDB to parquet
- `compute_umap.py` — UMAP projection (3072D → 2D)
- `create_paper_map.py` — Generate interactive HTML

## Regenerating

```bash
python3 export_corpus_data.py
python3 compute_umap.py
python3 create_paper_map.py
cp output/mint_paper_map.html index.html
git add -A && git commit -m "Update" && git push
```
