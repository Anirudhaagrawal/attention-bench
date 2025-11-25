"""Attention Bench Dashboard - Data Explorer Page."""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import json

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    get_cached_data,
    get_available_runs,
    get_run_metadata,
    apply_filters,
    extract_approaches_from_df,
    shorten_approach_name,
    create_csv_download,
    create_json_download,
)

# Page configuration
st.set_page_config(
    page_title="Explorer - Attention Bench",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🔍 Data Explorer")
st.caption("Browse and search raw benchmark data")

# Load data
df, results_path = get_cached_data(return_path=True)

if df.empty:
    st.warning("No benchmark results found.")
    st.stop()

# Sidebar - Run Browser
with st.sidebar:
    st.header("🗂️ Run Browser")

    available_runs = get_available_runs(results_path)

    if available_runs:
        selected_run = st.selectbox(
            "Select Run",
            options=["All Runs"] + available_runs,
            help="Browse a specific benchmark run"
        )

        if selected_run != "All Runs":
            metadata = get_run_metadata(selected_run, results_path)
            st.markdown("**Run Details:**")
            st.write(f"- Files: {len(metadata.get('files', []))}")
            st.write(f"- Models: {', '.join(metadata.get('models', []))}")
            st.write(f"- TP: {', '.join(map(str, metadata.get('tp_degrees', [])))}")
    else:
        selected_run = "All Runs"

    st.divider()

    # Quick filters
    st.header("🔧 Filters")

    workload_filter = st.selectbox(
        "Workload Type",
        options=sorted(df['workload_type'].unique().tolist())
    )

    model_filter = st.multiselect(
        "Models",
        options=sorted(df['model'].unique().tolist()),
        default=sorted(df['model'].unique().tolist())
    )

    tp_filter = st.multiselect(
        "TP Degrees",
        options=sorted(df['tp_degree'].unique().tolist()),
        default=sorted(df['tp_degree'].unique().tolist())
    )

# Apply filters
filtered_df = df.copy()

if selected_run != "All Runs":
    filtered_df = filtered_df[filtered_df['run_id'] == selected_run]

if workload_filter:
    filtered_df = filtered_df[filtered_df['workload_type'] == workload_filter]

if model_filter:
    filtered_df = filtered_df[filtered_df['model'].isin(model_filter)]

if tp_filter:
    filtered_df = filtered_df[filtered_df['tp_degree'].isin(tp_filter)]

# Main content
st.subheader(f"📁 Showing {len(filtered_df)} scenarios")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔎 Search", "📋 Browse", "📊 Statistics"])

with tab1:
    st.subheader("Search Scenarios")

    # Search box
    search_term = st.text_input(
        "Search scenario names",
        placeholder="e.g., decode_b64_kv4k",
        help="Search for scenarios by name pattern"
    )

    if search_term:
        search_results = filtered_df[
            filtered_df['scenario_name'].str.contains(search_term, case=False, na=False)
        ]
        st.write(f"Found {len(search_results)} matching scenarios")

        if not search_results.empty:
            # Show results
            for _, row in search_results.head(20).iterrows():
                with st.expander(f"**{row['scenario_name']}** ({row['model']} TP={row['tp_degree']})"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write("**Configuration:**")
                        st.write(f"- Workload: {row['workload_type']}")
                        st.write(f"- Batch: {row['batch_size']}")
                        st.write(f"- Query: {row['query_length']}")
                        st.write(f"- KV: {row['kv_length']}")

                    with col2:
                        st.write("**Timing (median, ms):**")
                        approaches = extract_approaches_from_df(pd.DataFrame([row]))
                        for approach in approaches:
                            col = f"{approach}_median"
                            if col in row and pd.notna(row[col]):
                                time_ms = row[col] * 1000
                                st.write(f"- {shorten_approach_name(approach)}: {time_ms:.2f}")

                    with col3:
                        st.write("**Run Info:**")
                        st.write(f"- Run: {row['run_id']}")
                        st.write(f"- CUDA Graphs: {row.get('used_cuda_graphs', False)}")

            if len(search_results) > 20:
                st.info(f"Showing first 20 of {len(search_results)} results")
    else:
        st.info("Enter a search term to find scenarios")

with tab2:
    st.subheader("Browse All Data")

    # Column selector
    all_columns = filtered_df.columns.tolist()
    default_cols = ['scenario_name', 'model', 'tp_degree', 'workload_type',
                   'batch_size', 'kv_length']
    default_cols = [c for c in default_cols if c in all_columns]

    selected_cols = st.multiselect(
        "Select columns to display",
        options=all_columns,
        default=default_cols,
        help="Choose which columns to show"
    )

    if selected_cols:
        display_df = filtered_df[selected_cols]

        # Sorting
        col1, col2 = st.columns([2, 1])
        with col1:
            sort_col = st.selectbox(
                "Sort by",
                options=selected_cols,
                index=0
            )
        with col2:
            sort_asc = st.checkbox("Ascending", value=True)

        display_df = display_df.sort_values(sort_col, ascending=sort_asc)

        st.dataframe(display_df, use_container_width=True, height=500)

        # Export
        st.divider()
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            create_csv_download(display_df, "attention_bench_export.csv")
        with col2:
            create_json_download(display_df.to_dict(orient='records'), "attention_bench_export.json")

with tab3:
    st.subheader("Data Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Counts")
        st.write(f"**Total Scenarios:** {len(filtered_df)}")
        st.write(f"**Unique Runs:** {filtered_df['run_id'].nunique()}")
        st.write(f"**Models:** {filtered_df['model'].nunique()}")
        st.write(f"**TP Degrees:** {len(filtered_df['tp_degree'].unique())}")

        st.markdown("### Workload Distribution")
        workload_counts = filtered_df['workload_type'].value_counts()
        for wtype, count in workload_counts.items():
            st.write(f"- {wtype}: {count} ({100*count/len(filtered_df):.1f}%)")

    with col2:
        st.markdown("### Model Distribution")
        model_counts = filtered_df['model'].value_counts()
        for model, count in model_counts.items():
            st.write(f"- {model}: {count} ({100*count/len(filtered_df):.1f}%)")

        st.markdown("### TP Distribution")
        tp_counts = filtered_df['tp_degree'].value_counts().sort_index()
        for tp, count in tp_counts.items():
            st.write(f"- TP={tp}: {count} ({100*count/len(filtered_df):.1f}%)")

    # Parameter ranges
    st.markdown("### Parameter Ranges")
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'batch_size' in filtered_df.columns:
            st.metric("Batch Size",
                     f"{int(filtered_df['batch_size'].min())} - {int(filtered_df['batch_size'].max())}")

    with col2:
        if 'query_length' in filtered_df.columns:
            st.metric("Query Length",
                     f"{int(filtered_df['query_length'].min())} - {int(filtered_df['query_length'].max())}")

    with col3:
        if 'kv_length' in filtered_df.columns:
            st.metric("KV Length",
                     f"{int(filtered_df['kv_length'].min())} - {int(filtered_df['kv_length'].max())}")
