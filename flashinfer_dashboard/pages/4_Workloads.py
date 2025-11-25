"""Attention Bench Dashboard - Workload Category Analysis Page."""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils import (
    initialize_page_data,
    shorten_approach_name,
    APPROACH_COLORS,
)

# Import workload categorizer
from attention_bench.plotting.workload_categorizer import (
    categorize_workload,
    get_category_description,
    WORKLOAD_CATEGORIES,
)

# Page configuration
st.set_page_config(
    page_title="Workloads - Attention Bench",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📋 Workload Category Analysis")
st.caption("Performance analysis by realistic workload category")

# Workload category descriptions
st.info("""
**Workload Categories:**
- **Code**: Long context code completion (low batch, high KV)
- **Chat**: Conversational AI (medium batch, medium KV)
- **Summarization**: Batch document summarization (high batch, high KV)
""")

# Load data
@st.cache_data
def add_categories(df):
    """Add workload categories to DataFrame."""
    if df.empty:
        return df

    def get_categories(row):
        if row['workload_type'] == 'decode':
            batch = row['batch_size']
            query = 1
        elif row['workload_type'] == 'prefill':
            batch = row['batch_size']
            query = row['query_length']
        else:  # mixed
            # For mixed, categorize based on decode params
            batch = row.get('decode_batch', row['batch_size'])
            query = 1

        kv = row['kv_length'] if row['kv_length'] > 0 else row.get('decode_kv', 0)

        categories = categorize_workload(
            batch, query, kv, row['workload_type']
        )

        return ','.join(categories) if categories else 'uncategorized'

    df = df.copy()
    df['workload_category'] = df.apply(get_categories, axis=1)
    return df

# Initialize page data
raw_df, filtered_df, approaches, filters = initialize_page_data(
    require_approaches=False,
    page_key="workloads"
)

# Add categories to both dataframes
df = add_categories(raw_df)
filtered_df = add_categories(filtered_df)

# Category selector
st.subheader("Select Category")

categories = WORKLOAD_CATEGORIES + ['uncategorized']
selected_categories = st.multiselect(
    "Workload Categories",
    options=categories,
    default=WORKLOAD_CATEGORIES,
    help="Filter by workload category"
)

# Filter by selected categories
category_df = filtered_df[
    filtered_df['workload_category'].apply(
        lambda x: any(cat in x for cat in selected_categories)
    )
]

if category_df.empty:
    st.warning("No scenarios match the selected categories.")
    st.stop()

st.divider()

# Analysis tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔥 Heatmaps by Category", "📈 Best Approach per Category"])

with tab1:
    st.subheader("Category Distribution")

    # Count scenarios per category
    category_counts = {}
    for cat in categories:
        count = len(category_df[category_df['workload_category'].str.contains(cat)])
        if count > 0:
            category_counts[cat] = count

    if category_counts:
        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(category_counts.keys()),
            values=list(category_counts.values()),
            hole=0.4,
            textinfo='label+percent',
        )])
        fig.update_layout(title="Scenarios by Workload Category", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Category details
    st.subheader("Category Details")

    for cat in selected_categories:
        cat_data = category_df[category_df['workload_category'].str.contains(cat)]

        if cat_data.empty:
            continue

        with st.expander(f"**{cat.title()}** ({len(cat_data)} scenarios)", expanded=True):
            # Description
            for wtype in cat_data['workload_type'].unique():
                desc = get_category_description(cat, wtype)
                if desc:
                    st.caption(desc)

            # Stats
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Scenarios", len(cat_data))

            with col2:
                if 'batch_size' in cat_data.columns:
                    st.metric("Batch Range",
                             f"{int(cat_data['batch_size'].min())}-{int(cat_data['batch_size'].max())}")

            with col3:
                if 'kv_length' in cat_data.columns:
                    min_kv = cat_data['kv_length'].min()
                    max_kv = cat_data['kv_length'].max()
                    st.metric("KV Range", f"{int(min_kv//1024)}k-{int(max_kv//1024)}k")

with tab2:
    st.subheader("Performance by Category")

    for cat in selected_categories:
        cat_data = category_df[category_df['workload_category'].str.contains(cat)]

        if cat_data.empty or len(approaches) < 2:
            continue

        st.markdown(f"### {cat.title()}")

        # Bar chart comparing approaches
        avg_times = {}
        for approach in approaches:
            col = f"{approach}_median"
            if col in cat_data.columns:
                times = cat_data[col].dropna() * 1000  # Convert to ms
                if not times.empty:
                    avg_times[shorten_approach_name(approach)] = times.mean()

        if avg_times:
            # Get colors for shortened approach names
            colors = []
            for short_name in avg_times.keys():
                # Try to get color using shortened name first, then try original approach names
                color = APPROACH_COLORS.get(short_name)
                if color is None:
                    # Find original approach name that maps to this shortened name
                    for orig in approaches:
                        if shorten_approach_name(orig) == short_name:
                            color = APPROACH_COLORS.get(orig, '#808080')
                            break
                if color is None:
                    color = '#808080'
                colors.append(color)

            fig = go.Figure(data=[go.Bar(
                x=list(avg_times.keys()),
                y=list(avg_times.values()),
                marker_color=colors
            )])
            fig.update_layout(
                title=f"Average Time by Approach ({cat.title()})",
                xaxis_title="Approach",
                yaxis_title="Average Time (ms)",
                height=350,
                plot_bgcolor='white',
            )
            st.plotly_chart(fig, use_container_width=True)

            # Identify winner
            if avg_times:
                winner = min(avg_times.items(), key=lambda x: x[1])
                st.success(f"**Best for {cat.title()}:** {winner[0]} ({winner[1]:.2f}ms avg)")

        st.divider()

with tab3:
    st.subheader("Optimal Approach Recommendation")

    # Build recommendation table
    recommendations = []

    for cat in WORKLOAD_CATEGORIES:
        cat_data = category_df[category_df['workload_category'].str.contains(cat)]

        if cat_data.empty:
            continue

        # Calculate average times for each approach
        avg_times = {}
        for approach in approaches:
            col = f"{approach}_median"
            if col in cat_data.columns:
                times = cat_data[col].dropna()
                if not times.empty:
                    avg_times[approach] = times.mean() * 1000  # Convert to ms

        if avg_times:
            # Find best approach
            best_approach = min(avg_times.items(), key=lambda x: x[1])
            sorted_times = sorted(avg_times.items(), key=lambda x: x[1])

            # Calculate margin over second place
            if len(sorted_times) > 1:
                margin = (sorted_times[1][1] / sorted_times[0][1] - 1) * 100
            else:
                margin = 0

            recommendations.append({
                "Category": cat.title(),
                "Best Approach": shorten_approach_name(best_approach[0]),
                "Avg Time (ms)": f"{best_approach[1]:.2f}",
                "Margin": f"+{margin:.1f}%",
                "Scenarios": len(cat_data),
            })

    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

        # Summary
        st.subheader("Summary")
        st.markdown("""
        The table above shows the **best performing approach** for each workload category
        based on median execution time. Use these recommendations to select the optimal
        approach for your specific use case:

        - **Code**: Long-running code completion tasks with large context windows
        - **Chat**: Interactive conversational AI with moderate context
        - **Summarization**: Batch processing of documents with high throughput needs
        """)
    else:
        st.info("Not enough data to generate recommendations. Try adjusting filters.")

# Footer
st.divider()
with st.expander("ℹ️ About Workload Categories"):
    st.markdown("""
    ### Decode Workloads
    - **Code**: Batch 4-16, KV 16k-1M (long context code completion)
    - **Chat**: Batch 16-128, KV 1k-128k (conversational AI)
    - **Summarization**: Batch 32-256, KV 8k-512k (batch document processing)

    ### Prefill Workloads
    - **Code**: Chunk 32-4k, KV 16k-1M
    - **Chat**: Chunk 32-4k, KV 1k-128k
    - **Summarization**: Chunk 2k-8k, KV 8k-512k

    Scenarios may match multiple categories if their parameters fall within overlapping ranges.
    """)
