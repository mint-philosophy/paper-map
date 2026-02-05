#!/usr/bin/env python3
"""
Create interactive paper map visualization using datamapplot.
"""

import json
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

    # Create coords dataframe with document IDs
    coords_df = pd.DataFrame({
        'document_id': doc_ids,
        'umap_x': coords[:, 0],
        'umap_y': coords[:, 1]
    })

    # Merge to align data
    df = coords_df.merge(df, on='document_id', how='left')

    return df, coords


def prepare_labels(df):
    """Prepare hierarchical label arrays."""
    # Layer 1: Macro categories (zoomed out)
    macro_labels = df['macro_category'].fillna('Uncategorized').values

    # Layer 2: Cluster labels (medium zoom)
    cluster_labels = df['cluster_label'].fillna('Uncategorized').values

    # Layer 3: Paper titles (zoomed in) - truncated for display
    title_labels = df['title'].apply(
        lambda x: x[:50] + '...' if len(str(x)) > 50 else str(x)
    ).values

    return macro_labels, cluster_labels, title_labels


def prepare_hover_text(df):
    """Create rich hover text for each paper."""
    hover_texts = []

    for _, row in df.iterrows():
        title = row.get('title', 'Unknown')
        hover_texts.append(f"{title}")

    return hover_texts


def prepare_extra_data(df):
    """Prepare extra data for tooltip template."""
    # Process authors
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
        'url': df['drive_url'].fillna(''),
        # Combined field for search
        'searchable': df['title'].fillna('') + ' ' + authors_str
    })

    return extra


def compute_marker_sizes(df, size=5):
    """Return uniform marker sizes."""
    return np.full(len(df), size)


def create_visualization(df, coords):
    """Create the interactive visualization."""
    print("Preparing visualization data...")

    macro_labels, cluster_labels, title_labels = prepare_labels(df)
    hover_text = prepare_hover_text(df)
    extra_data = prepare_extra_data(df)
    marker_sizes = compute_marker_sizes(df)

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
        </div>
        <div style="font-size:11px; color:#ccc; line-height:1.4;">
            {abstract}
        </div>
    </div>
    """

    # Custom CSS for year filter and legend
    custom_css = """
    #year-filter {
        position: absolute;
        bottom: 20px;
        left: 20px;
        background: rgba(30,30,30,0.9);
        padding: 12px 16px;
        border-radius: 8px;
        font-family: system-ui, sans-serif;
        z-index: 1000;
    }
    #year-filter label {
        color: #aaa;
        font-size: 12px;
        display: block;
        margin-bottom: 6px;
    }
    #year-slider {
        width: 200px;
        accent-color: #6b9;
    }
    #year-display {
        color: #fff;
        font-size: 14px;
        font-weight: 600;
        margin-top: 4px;
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
    }
    """

    # Custom HTML for year filter
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())

    custom_html = f"""
    <div id="year-filter">
        <label>Filter by Year</label>
        <input type="range" id="year-slider" min="{min_year}" max="{max_year}" value="{max_year}">
        <div id="year-display">{min_year} - {max_year}</div>
    </div>
    <div id="stats-display">
        <div id="paper-count">{len(df):,} papers</div>
    </div>
    """

    # Custom JS for year filtering and fixing click handler
    custom_js = """
    const slider = document.getElementById('year-slider');
    const display = document.getElementById('year-display');
    const countDisplay = document.getElementById('paper-count');
    const minYear = parseInt(slider.min);

    slider.addEventListener('input', function() {
        const maxYear = parseInt(this.value);
        display.textContent = minYear + ' - ' + maxYear;
    });

    // Fix click handler - override after datamap initializes
    setTimeout(() => {
        if (typeof datamap !== 'undefined' && datamap.deckgl) {
            datamap.deckgl.setProps({
                onClick: ({index, picked}) => {
                    if (picked && datamap.metaData && datamap.metaData.url) {
                        const url = datamap.metaData.url[index];
                        if (url) window.open(url, '_blank');
                    }
                }
            });
        }
    }, 1000);
    """

    print("Creating interactive plot...")

    # Create the visualization
    plot = datamapplot.create_interactive_plot(
        coords,
        macro_labels,
        cluster_labels,
        hover_text=hover_text,
        extra_point_data=extra_data,
        hover_text_html_template=hover_template,
        # Click handler set via custom_js due to datamapplot template bug
        on_click=None,
        enable_search=True,
        search_field='searchable',
        title="MINT Lab Research Corpus",
        sub_title=f"{len(df):,} papers across {df['macro_category'].nunique()} research areas",
        font_family="Inter",
        darkmode=True,
        marker_size_array=marker_sizes,
        cluster_boundary_polygons=True,
        polygon_alpha=0.3,
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

    # Save
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_PATH / "mint_paper_map.html"
    plot.save(str(output_file))
    print(f"\nSaved visualization to {output_file}")

    # Also save a data summary
    summary = {
        "total_papers": len(df),
        "macro_categories": df['macro_category'].value_counts().to_dict(),
        "year_range": [int(df['year'].min()), int(df['year'].max())]
    }
    with open(OUTPUT_PATH / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
