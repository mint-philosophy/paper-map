#!/usr/bin/env python3
"""
Create interactive paper map visualization using datamapplot.
"""

import json
import colorsys
import numpy as np
import pandas as pd
import datamapplot
from pathlib import Path

# Paths
DATA_PATH = Path(__file__).parent / "data"
ASSETS_PATH = Path(__file__).parent / "assets"
OUTPUT_PATH = Path(__file__).parent / "output"


def load_data():
    """Load all preprocessed data."""
    df = pd.read_parquet(DATA_PATH / "corpus_data.parquet")
    coords = np.load(DATA_PATH / "umap_coords.npy")
    doc_ids = np.load(DATA_PATH / "document_ids.npy", allow_pickle=True)

    coords_df = pd.DataFrame({
        'document_id': doc_ids,
        'umap_x': coords[:, 0],
        'umap_y': coords[:, 1]
    })

    df = coords_df.merge(df, on='document_id', how='left')
    return df, coords


def prepare_labels(df):
    """Prepare hierarchical label arrays."""
    macro_labels = df['macro_category'].fillna('Uncategorized').values
    cluster_labels = df['cluster_label'].fillna('Uncategorized').values
    return macro_labels, cluster_labels


def prepare_hover_text(df):
    """Create rich hover text for each paper."""
    return [row.get('title', 'Unknown') for _, row in df.iterrows()]


def prepare_extra_data(df):
    """Prepare extra data for tooltip template."""
    authors_str = df['authors'].apply(
        lambda x: ', '.join(x[:3]) + (' et al.' if len(x) > 3 else '')
        if isinstance(x, list) else str(x)[:100]
    )

    extra = pd.DataFrame({
        'title': df['title'].fillna('Unknown'),
        'authors': authors_str,
        'year': df['year'].fillna('N/A').astype(str),
        'abstract': df['abstract'].fillna('').apply(
            lambda x: str(x)[:300] + '...' if len(str(x)) > 300 else str(x)
        ),
        'summary': df['document_summary'].fillna('').apply(
            lambda x: str(x)[:400] + '...' if len(str(x)) > 400 else str(x)
        ),
        'category': df['macro_category'].fillna('Uncategorized'),
        'discipline': df['discipline'].fillna('') if 'discipline' in df.columns else '',
        'url': df['drive_url'].fillna(''),
        'searchable': df['title'].fillna('') + ' ' + authors_str,
    })

    return extra


def build_category_index(df):
    """Build per-category point index lists for JS filtering."""
    categories = df['macro_category'].fillna('Uncategorized')
    cat_counts = categories.value_counts()
    cat_index = {}
    for cat in cat_counts.index:
        cat_index[cat] = df.index[categories == cat].tolist()
    return cat_index, cat_counts.to_dict()


def category_color(idx, total):
    """Generate a distinct HSL color for a category."""
    hue = (idx * 360 / max(total, 1)) % 360
    r, g, b = colorsys.hls_to_rgb(hue / 360, 0.55, 0.65)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def create_visualization(df, coords):
    """Create the interactive visualization."""
    print("Preparing visualization data...")

    macro_labels, cluster_labels = prepare_labels(df)
    hover_text = prepare_hover_text(df)
    extra_data = prepare_extra_data(df)
    marker_sizes = np.full(len(df), 5)

    # Build filter data
    cat_index, cat_counts = build_category_index(df)
    categories_ordered = sorted(cat_counts.keys(), key=lambda c: -cat_counts[c])
    cat_index_json = json.dumps(cat_index)

    # Build discipline index
    disciplines = df['discipline'].fillna('') if 'discipline' in df.columns else pd.Series([''] * len(df))
    disc_counts = disciplines[disciplines != ''].value_counts().to_dict()
    disc_index = {}
    for disc in disc_counts:
        disc_index[disc] = df.index[disciplines == disc].tolist()
    disc_index_json = json.dumps(disc_index)
    disciplines_ordered = sorted(disc_counts.keys(), key=lambda d: -disc_counts[d])

    # Build year index
    years = df['year'].fillna(0).astype(int)
    min_year = int(years[years > 1900].min()) if (years > 1900).any() else 2000
    max_year = int(years.max())
    year_index = {}
    for y in range(min_year, max_year + 1):
        indices = df.index[years == y].tolist()
        if indices:
            year_index[str(y)] = indices
    year_index_json = json.dumps(year_index)

    # Custom hover template
    hover_template = """
    <div style="max-width:350px; font-family: system-ui, sans-serif;">
        <div style="font-size:14px; font-weight:600; color:#fff; margin-bottom:6px;">
            {title}
        </div>
        <div style="font-size:11px; color:#aaa; margin-bottom:4px;">
            {authors}
        </div>
        <div style="display:flex; gap:12px; font-size:11px; color:#888; margin-bottom:8px;">
            <span>{year}</span>
            <span style="color:#6b9; font-weight:500;">{category}</span>
            <span style="color:#7eb5d6;">{discipline}</span>
        </div>
        <div style="font-size:11px; color:#ccc; line-height:1.4;">
            {abstract}
        </div>
    </div>
    """

    # --- Build filter chip HTML ---
    chips_html = ""
    for i, cat in enumerate(categories_ordered):
        count = cat_counts[cat]
        color = category_color(i, len(categories_ordered))
        chips_html += (
            f'<button class="filter-chip cat-chip active" data-category="{cat}" '
            f'style="--chip-color:{color}">'
            f'{cat} <span class="chip-count">{count}</span></button>\n'
        )

    disc_chips_html = ""
    disc_colors = ["#7eb5d6", "#d6a87e", "#9ed67e", "#d67eb5", "#d6d67e",
                    "#7ed6c4", "#b57ed6", "#d67e7e", "#7e8bd6", "#c4d67e",
                    "#d6b57e", "#7ed6a8", "#d67ec4", "#8bd67e", "#d6977e"]
    for i, disc in enumerate(disciplines_ordered):
        count = disc_counts[disc]
        color = disc_colors[i % len(disc_colors)]
        disc_chips_html += (
            f'<button class="filter-chip disc-chip active" data-discipline="{disc}" '
            f'style="--chip-color:{color}">'
            f'{disc} <span class="chip-count">{count}</span></button>\n'
        )

    # Custom CSS
    custom_css = """
    #filter-panel {
        position: absolute;
        top: 70px;
        right: 16px;
        background: rgba(20,20,20,0.92);
        backdrop-filter: blur(8px);
        padding: 12px;
        border-radius: 12px;
        font-family: system-ui, sans-serif;
        z-index: 1000;
        max-height: calc(100vh - 100px);
        overflow-y: auto;
        width: 260px;
        pointer-events: auto;
        scrollbar-width: thin;
        scrollbar-color: #444 transparent;
    }
    #filter-panel::-webkit-scrollbar { width: 4px; }
    #filter-panel::-webkit-scrollbar-thumb { background: #444; border-radius: 2px; }
    #filter-panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid #333;
    }
    #filter-panel-header span {
        color: #aaa;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .filter-header-btns {
        display: flex;
        gap: 4px;
    }
    .filter-header-btn {
        background: rgba(255,255,255,0.08);
        border: none;
        color: #888;
        font-size: 10px;
        padding: 3px 8px;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.15s;
    }
    .filter-header-btn:hover {
        background: rgba(255,255,255,0.15);
        color: #fff;
    }
    /* 3-way toggle */
    #mode-toggle {
        display: flex;
        background: rgba(255,255,255,0.06);
        border-radius: 6px;
        padding: 2px;
        margin-bottom: 8px;
    }
    .mode-btn {
        flex: 1;
        background: none;
        border: none;
        color: #666;
        font-size: 10px;
        padding: 5px 4px;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.15s;
        text-align: center;
        white-space: nowrap;
    }
    .mode-btn.active {
        background: rgba(255,255,255,0.12);
        color: #fff;
    }
    .mode-btn:hover:not(.active) {
        color: #aaa;
    }
    .filter-section-label {
        color: #666;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 8px 0 4px 0;
    }
    #category-section, #discipline-section {
        transition: max-height 0.2s ease, opacity 0.2s ease;
        overflow: hidden;
    }
    #category-section.hidden, #discipline-section.hidden {
        max-height: 0;
        opacity: 0;
        margin: 0;
        padding: 0;
    }
    #category-chips, #discipline-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
    }
    .filter-chip {
        background: rgba(255,255,255,0.06);
        border: 1px solid var(--chip-color, #555);
        color: #ccc;
        font-size: 11px;
        padding: 4px 8px;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.15s;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        line-height: 1.2;
    }
    .filter-chip.active {
        background: color-mix(in srgb, var(--chip-color) 25%, transparent);
        border-color: var(--chip-color);
        color: #fff;
    }
    .filter-chip:not(.active) {
        opacity: 0.45;
        border-color: #333;
    }
    .filter-chip:hover {
        opacity: 1;
        background: color-mix(in srgb, var(--chip-color) 35%, transparent);
    }
    .chip-count {
        font-size: 9px;
        color: #888;
        font-weight: 400;
    }
    #year-range {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 4px;
    }
    .year-input {
        width: 55px;
        background: rgba(255,255,255,0.08);
        border: 1px solid #444;
        color: #ccc;
        font-size: 12px;
        padding: 4px 6px;
        border-radius: 4px;
        text-align: center;
        font-family: system-ui, sans-serif;
    }
    .year-input:focus {
        border-color: #6b9;
        outline: none;
    }
    #year-range span {
        color: #666;
        font-size: 11px;
    }
    #filter-count {
        color: #6b9;
        font-size: 11px;
        margin-top: 8px;
        padding-top: 6px;
        border-top: 1px solid #333;
        text-align: center;
    }
    #toggle-filters {
        position: absolute;
        top: 70px;
        right: 16px;
        background: rgba(30,30,30,0.9);
        border: 1px solid #444;
        color: #aaa;
        font-size: 11px;
        padding: 6px 12px;
        border-radius: 6px;
        cursor: pointer;
        z-index: 1001;
        font-family: system-ui, sans-serif;
        pointer-events: auto;
        display: none;
        transition: all 0.15s;
    }
    #toggle-filters:hover {
        color: #fff;
        background: rgba(50,50,50,0.95);
    }
    #stats-display {
        position: absolute;
        bottom: 20px;
        right: 20px;
        background: rgba(30,30,30,0.9);
        padding: 12px 16px;
        border-radius: 8px;
        font-family: system-ui, sans-serif;
        color: #aaa;
        font-size: 12px;
        z-index: 1000;
        pointer-events: auto;
    }
    #credits-btn {
        position: absolute;
        bottom: 20px;
        left: 20px;
        background: rgba(30,30,30,0.9);
        padding: 8px 12px;
        border-radius: 6px;
        font-family: system-ui, sans-serif;
        color: #888;
        font-size: 11px;
        z-index: 1000;
        cursor: pointer;
        text-decoration: none;
        transition: color 0.2s, background 0.2s;
        pointer-events: auto;
    }
    #credits-btn:hover {
        color: #fff;
        background: rgba(50,50,50,0.95);
    }
    """

    # Custom HTML
    custom_html = f"""
    <button id="toggle-filters">Filters</button>
    <div id="filter-panel">
        <div id="filter-panel-header">
            <span>Filters</span>
            <div class="filter-header-btns">
                <button class="filter-header-btn" id="btn-all">All</button>
                <button class="filter-header-btn" id="btn-none">None</button>
                <button class="filter-header-btn" id="btn-close-filters">Close</button>
            </div>
        </div>
        <div id="mode-toggle">
            <button class="mode-btn" data-mode="areas">Research Areas</button>
            <button class="mode-btn active" data-mode="both">Both</button>
            <button class="mode-btn" data-mode="disciplines">Disciplines</button>
        </div>
        <div id="category-section">
            <div class="filter-section-label">Research Areas</div>
            <div id="category-chips">
                {chips_html}
            </div>
        </div>
        <div id="discipline-section">
            <div class="filter-section-label">Disciplines</div>
            <div id="discipline-chips">
                {disc_chips_html}
            </div>
        </div>
        <div class="filter-section-label" style="margin-top:10px;">Year Range</div>
        <div id="year-range">
            <input type="number" class="year-input" id="year-min" value="{min_year}" min="{min_year}" max="{max_year}">
            <span>&ndash;</span>
            <input type="number" class="year-input" id="year-max" value="{max_year}" min="{min_year}" max="{max_year}">
        </div>
        <div id="filter-count">{len(df):,} / {len(df):,} papers</div>
    </div>
    <div id="stats-display">
        <div id="paper-count">{len(df):,} papers</div>
    </div>
    <a id="credits-btn" href="https://github.com/TutteInstitute/datamapplot" target="_blank" title="Built with datamapplot">
        Built with datamapplot
    </a>
    """

    # Custom JS
    custom_js = f"""
    const CAT_INDEX = {cat_index_json};
    const DISC_INDEX = {disc_index_json};
    const YEAR_INDEX = {year_index_json};
    const TOTAL_PAPERS = {len(df)};

    // State
    const activeCategories = new Set(Object.keys(CAT_INDEX));
    const activeDisciplines = new Set(Object.keys(DISC_INDEX));
    const ALL_CAT_COUNT = Object.keys(CAT_INDEX).length;
    const ALL_DISC_COUNT = Object.keys(DISC_INDEX).length;
    let yearMin = {min_year};
    let yearMax = {max_year};
    let filtersApplied = false;
    let currentMode = 'both'; // 'areas', 'both', 'disciplines'

    function applyFilters() {{
        const useCats = currentMode === 'areas' || currentMode === 'both';
        const useDiscs = currentMode === 'disciplines' || currentMode === 'both';

        const allCatsActive = !useCats || activeCategories.size === ALL_CAT_COUNT;
        const allDiscsActive = !useDiscs || activeDisciplines.size === ALL_DISC_COUNT;
        const fullYearRange = yearMin <= {min_year} && yearMax >= {max_year};

        if (allCatsActive && allDiscsActive && fullYearRange) {{
            if (filtersApplied) {{
                datamap.removeSelection('category-filter');
                filtersApplied = false;
            }}
            document.getElementById('filter-count').textContent =
                TOTAL_PAPERS.toLocaleString() + ' / ' + TOTAL_PAPERS.toLocaleString() + ' papers';
            return;
        }}

        // Year filter
        let yearValid = new Set();
        for (let y = yearMin; y <= yearMax; y++) {{
            const indices = YEAR_INDEX[String(y)];
            if (indices) indices.forEach(i => yearValid.add(i));
        }}
        if (yearMin <= {min_year} && YEAR_INDEX["0"]) {{
            YEAR_INDEX["0"].forEach(i => yearValid.add(i));
        }}

        // Category filter
        let catValid = null;
        if (useCats && activeCategories.size < ALL_CAT_COUNT) {{
            catValid = new Set();
            for (const cat of activeCategories) {{
                const indices = CAT_INDEX[cat];
                if (indices) indices.forEach(i => catValid.add(i));
            }}
        }}

        // Discipline filter
        let discValid = null;
        if (useDiscs && activeDisciplines.size < ALL_DISC_COUNT) {{
            discValid = new Set();
            for (const disc of activeDisciplines) {{
                const indices = DISC_INDEX[disc];
                if (indices) indices.forEach(i => discValid.add(i));
            }}
        }}

        // Intersect all active filters
        const matchingIndices = [];
        for (let i = 0; i < TOTAL_PAPERS; i++) {{
            if (!yearValid.has(i)) continue;
            if (catValid !== null && !catValid.has(i)) continue;
            if (discValid !== null && !discValid.has(i)) continue;
            matchingIndices.push(i);
        }}

        datamap.addSelection(matchingIndices, 'category-filter');
        filtersApplied = true;

        document.getElementById('filter-count').textContent =
            matchingIndices.length.toLocaleString() + ' / ' + TOTAL_PAPERS.toLocaleString() + ' papers';
    }}

    function setMode(mode) {{
        currentMode = mode;
        const catSection = document.getElementById('category-section');
        const discSection = document.getElementById('discipline-section');

        // Update toggle buttons
        document.querySelectorAll('.mode-btn').forEach(btn => {{
            btn.classList.toggle('active', btn.dataset.mode === mode);
        }});

        // Show/hide sections
        if (mode === 'areas') {{
            catSection.classList.remove('hidden');
            discSection.classList.add('hidden');
        }} else if (mode === 'disciplines') {{
            catSection.classList.add('hidden');
            discSection.classList.remove('hidden');
        }} else {{
            catSection.classList.remove('hidden');
            discSection.classList.remove('hidden');
        }}

        applyFilters();
    }}

    function initFilters() {{
        // Category chips
        document.querySelectorAll('.cat-chip').forEach(chip => {{
            chip.addEventListener('click', () => {{
                const cat = chip.dataset.category;
                chip.classList.toggle('active');
                if (chip.classList.contains('active')) {{
                    activeCategories.add(cat);
                }} else {{
                    activeCategories.delete(cat);
                }}
                applyFilters();
            }});
        }});

        // Discipline chips
        document.querySelectorAll('.disc-chip').forEach(chip => {{
            chip.addEventListener('click', () => {{
                const disc = chip.dataset.discipline;
                chip.classList.toggle('active');
                if (chip.classList.contains('active')) {{
                    activeDisciplines.add(disc);
                }} else {{
                    activeDisciplines.delete(disc);
                }}
                applyFilters();
            }});
        }});

        // Mode toggle
        document.querySelectorAll('.mode-btn').forEach(btn => {{
            btn.addEventListener('click', () => setMode(btn.dataset.mode));
        }});

        // All / None — applies to visible sections only
        document.getElementById('btn-all').addEventListener('click', () => {{
            const catVisible = !document.getElementById('category-section').classList.contains('hidden');
            const discVisible = !document.getElementById('discipline-section').classList.contains('hidden');
            if (catVisible) {{
                document.querySelectorAll('.cat-chip').forEach(c => {{
                    c.classList.add('active');
                    activeCategories.add(c.dataset.category);
                }});
            }}
            if (discVisible) {{
                document.querySelectorAll('.disc-chip').forEach(c => {{
                    c.classList.add('active');
                    activeDisciplines.add(c.dataset.discipline);
                }});
            }}
            applyFilters();
        }});
        document.getElementById('btn-none').addEventListener('click', () => {{
            const catVisible = !document.getElementById('category-section').classList.contains('hidden');
            const discVisible = !document.getElementById('discipline-section').classList.contains('hidden');
            if (catVisible) {{
                document.querySelectorAll('.cat-chip').forEach(c => {{
                    c.classList.remove('active');
                    activeCategories.delete(c.dataset.category);
                }});
            }}
            if (discVisible) {{
                document.querySelectorAll('.disc-chip').forEach(c => {{
                    c.classList.remove('active');
                    activeDisciplines.delete(c.dataset.discipline);
                }});
            }}
            applyFilters();
        }});

        // Year range
        document.getElementById('year-min').addEventListener('change', function() {{
            yearMin = parseInt(this.value) || {min_year};
            applyFilters();
        }});
        document.getElementById('year-max').addEventListener('change', function() {{
            yearMax = parseInt(this.value) || {max_year};
            applyFilters();
        }});

        // Close / toggle panel
        const panel = document.getElementById('filter-panel');
        const toggleBtn = document.getElementById('toggle-filters');
        document.getElementById('btn-close-filters').addEventListener('click', () => {{
            panel.style.display = 'none';
            toggleBtn.style.display = 'block';
        }});
        toggleBtn.addEventListener('click', () => {{
            panel.style.display = 'block';
            toggleBtn.style.display = 'none';
        }});
    }}

    setTimeout(initFilters, 2000);

    // Fix click handler
    setTimeout(() => {{
        if (typeof datamap !== 'undefined' && datamap.deckgl) {{
            datamap.deckgl.setProps({{
                onClick: ({{index, picked}}) => {{
                    if (picked && datamap.metaData && datamap.metaData.url) {{
                        const url = datamap.metaData.url[index];
                        if (url) window.open(url, '_blank');
                    }}
                }}
            }});
        }}
    }}, 1000);
    """

    print("Creating interactive plot...")

    plot = datamapplot.create_interactive_plot(
        coords,
        macro_labels,
        cluster_labels,
        hover_text=hover_text,
        extra_point_data=extra_data,
        hover_text_html_template=hover_template,
        on_click=None,
        enable_search=True,
        search_field='searchable',
        title="MINT Lab Research Corpus",
        sub_title=f"{len(df):,} papers across {df['macro_category'].nunique()} research areas",
        font_family="Inter",
        darkmode=True,
        marker_size_array=marker_sizes,
        cluster_boundary_polygons=True,
        polygon_alpha=0.5,
        initial_zoom_fraction=0.95,
        custom_css=custom_css,
        custom_html=custom_html,
        custom_js=custom_js,
        offline_mode=False,
        width="100%",
        height=900
    )

    return plot


def main():
    print("Loading data...")
    df, coords = load_data()
    print(f"Loaded {len(df)} papers")

    print("\nData summary:")
    print(f"  Macro categories: {df['macro_category'].nunique()}")
    print(f"  Year range: {df['year'].min()}-{df['year'].max()}")

    plot = create_visualization(df, coords)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_PATH / "mint_paper_map.html"
    plot.save(str(output_file))
    print(f"\nSaved visualization to {output_file}")

    summary = {
        "total_papers": len(df),
        "macro_categories": df['macro_category'].value_counts().to_dict(),
        "year_range": [int(df['year'].min()), int(df['year'].max())]
    }
    with open(OUTPUT_PATH / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
