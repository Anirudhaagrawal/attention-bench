"""Export utilities for Attention Bench dashboard."""

import streamlit as st
import pandas as pd
from typing import List, Any
import json
from datetime import datetime


def create_csv_download(df: pd.DataFrame, filename: str = "benchmark_results.csv") -> None:
    """Create a download button for CSV export.

    Args:
        df: DataFrame to export
        filename: Name for downloaded file
    """
    if df.empty:
        st.warning("No data to export")
        return

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def create_json_download(data: Any, filename: str = "benchmark_results.json") -> None:
    """Create a download button for JSON export.

    Args:
        data: Data to export (dict or list)
        filename: Name for downloaded file
    """
    if not data:
        st.warning("No data to export")
        return

    # Convert DataFrame to dict if needed
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient='records')

    json_str = json.dumps(data, indent=2, default=str)
    st.download_button(
        label="📥 Download as JSON",
        data=json_str,
        file_name=filename,
        mime="application/json",
    )


def create_markdown_report(
    df: pd.DataFrame,
    approaches: List[str],
    filters: dict = None,
    metric: str = "median"
) -> str:
    """Generate a markdown report from benchmark data.

    Args:
        df: DataFrame with benchmark results
        approaches: List of approaches analyzed
        filters: Applied filters (for documentation)
        metric: Metric used for analysis

    Returns:
        Markdown string
    """
    report = []

    # Header
    report.append("# Attention Bench Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Summary
    report.append("## Summary")
    report.append(f"- **Total Scenarios**: {len(df)}")
    report.append(f"- **Runs**: {df['run_id'].nunique()}")
    report.append(f"- **Models**: {', '.join(df['model'].unique())}")
    report.append(f"- **TP Degrees**: {', '.join(map(str, df['tp_degree'].unique()))}")
    report.append(f"- **Workload Types**: {', '.join(df['workload_type'].unique())}")
    report.append(f"- **Approaches Compared**: {', '.join(approaches)}")
    report.append("")

    # Filters applied
    if filters:
        report.append("## Filters Applied")
        for key, value in filters.items():
            if value and value not in ['All', [], None]:
                report.append(f"- **{key}**: {value}")
        report.append("")

    # Approach comparison
    report.append("## Approach Performance Summary")
    report.append("")

    for approach in approaches:
        col = f"{approach}_{metric}"
        if col in df.columns:
            times = df[col].dropna() * 1000  # Convert to ms

            if not times.empty:
                report.append(f"### {approach}")
                report.append(f"- Mean: {times.mean():.2f}ms")
                report.append(f"- Median: {times.median():.2f}ms")
                report.append(f"- Min: {times.min():.2f}ms")
                report.append(f"- Max: {times.max():.2f}ms")
                report.append(f"- Std Dev: {times.std():.2f}ms")
                report.append("")

    return "\n".join(report)


def create_report_download(
    df: pd.DataFrame,
    approaches: List[str],
    filters: dict = None,
    metric: str = "median"
) -> None:
    """Create a download button for markdown report.

    Args:
        df: DataFrame with benchmark results
        approaches: List of approaches analyzed
        filters: Applied filters
        metric: Metric used for analysis
    """
    if df.empty:
        st.warning("No data for report")
        return

    report = create_markdown_report(df, approaches, filters, metric)

    st.download_button(
        label="📥 Download Report (Markdown)",
        data=report,
        file_name=f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
    )


def export_sidebar(df: pd.DataFrame, approaches: List[str], filters: dict = None) -> None:
    """Add export options to sidebar.

    Args:
        df: DataFrame with benchmark results
        approaches: List of approaches
        filters: Applied filters
    """
    with st.sidebar:
        st.divider()
        st.subheader("📤 Export")

        col1, col2 = st.columns(2)

        with col1:
            if not df.empty:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="CSV",
                    data=csv,
                    file_name="benchmark_results.csv",
                    mime="text/csv",
                )

        with col2:
            if not df.empty:
                report = create_markdown_report(df, approaches, filters)
                st.download_button(
                    label="Report",
                    data=report,
                    file_name="benchmark_report.md",
                    mime="text/markdown",
                )
