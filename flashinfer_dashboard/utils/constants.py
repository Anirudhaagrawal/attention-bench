"""Constants for FlashInfer dashboard.

This module centralizes all magic numbers and configuration constants
to provide a single source of truth and prevent bugs from inconsistent values.
"""

# =============================================================================
# COMPARISON THRESHOLDS
# =============================================================================

# Speedup comparison thresholds
SPEEDUP_TIE_THRESHOLD = 0.01  # 1% tolerance for considering results equal
SPEEDUP_FASTER_THRESHOLD = 1.01  # Speedup > 1.01x = definitively faster
SPEEDUP_SLOWER_THRESHOLD = 0.99  # Speedup < 0.99x = definitively slower

# Minimum log deviation for colorscale visibility
MIN_LOG_DEVIATION = 0.05

# =============================================================================
# UNIT CONVERSIONS
# =============================================================================

# Time conversions
SECONDS_TO_MS = 1000  # Multiply seconds by 1000 to get milliseconds

# Size conversions
KB_BYTES = 1024  # 1 kilobyte = 1024 bytes
MB_BYTES = 1024 * 1024  # 1 megabyte = 1048576 bytes

# =============================================================================
# CHART DIMENSIONS
# =============================================================================

# Heatmap dimensions
HEATMAP_BASE_HEIGHT = 600  # Minimum heatmap height in pixels
HEATMAP_ROW_HEIGHT = 70  # Height per row in dynamic heatmaps
HEATMAP_MARGIN_BOTTOM = 200  # Bottom margin for legends
HEATMAP_MARGIN_TOP = 250  # Top margin for titles

# Standard chart heights
CHART_DEFAULT_HEIGHT = 500  # Default height for bar/distribution charts
LINE_GRAPH_HEIGHT = 600  # Height for line graphs
CATEGORY_CHART_HEIGHT = 450  # Height for category charts

# Margins (left, right, top, bottom)
DEFAULT_MARGINS = dict(l=50, r=10, t=80, b=80)

# =============================================================================
# HEATMAP INTENSITY SCALING
# =============================================================================

# Best performer heatmap intensity parameters
INTENSITY_MIN = 0.4  # Minimum color intensity (low margin of victory)
INTENSITY_MAX = 1.0  # Maximum color intensity (high margin of victory)
INTENSITY_SCALE_FACTOR = 1.2  # Scaling factor for intensity calculation
# Formula: intensity = min(1.0, INTENSITY_MIN + (speedup - 1.0) * INTENSITY_SCALE_FACTOR)

# =============================================================================
# DATA FILTERING
# =============================================================================

# Prefill KV lengths to exclude (too small to be meaningful)
PREFILL_SKIP_KV_LENGTHS = [32, 64]

# =============================================================================
# COLORBAR CONFIGURATION
# =============================================================================

# Speedup colorbar tick values (in linear scale)
SPEEDUP_TICK_VALUES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

# Colorbar styling
COLORBAR_THICKNESS = 15  # Horizontal colorbar thickness
COLORBAR_LENGTH = 0.8  # Colorbar length as fraction of plot width
COLORBAR_Y_POSITION = -0.15  # Vertical position (negative = below plot)

# =============================================================================
# LEGEND CONFIGURATION
# =============================================================================

# Approach legend positioning
LEGEND_Y_POSITION = -0.12  # Vertical position for approach legend
LEGEND_RECT_HEIGHT = 0.03  # Height of legend color rectangles
LEGEND_RECT_WIDTH = 0.03  # Width of legend color rectangles

# =============================================================================
# TEXT FORMATTING
# =============================================================================

# Font sizes
FONT_SIZE_NORMAL = 12  # Normal text
FONT_SIZE_LARGE = 14  # Large text (titles, axes)
FONT_SIZE_SMALL = 10  # Small text (minimum for uniformtext)

# Text formatting
CELL_TEXT_MIN_SIZE = 10  # Minimum text size in heatmap cells
