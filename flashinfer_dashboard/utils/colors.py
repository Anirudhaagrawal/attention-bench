"""Color management for FlashInfer dashboard.

This module centralizes all color-related configuration and utilities:
- Approach color palette
- Approach grouping definitions
- Color conversion and manipulation utilities
"""

from typing import Tuple, Dict, List


# Color palette for approaches - vibrant colors with better contrast
APPROACH_COLORS = {
    "official_fa3": "#E74C3C",                # Bright red
    "flashinfer_batch_attention": "#34495E",  # Dark slate
    "flashinfer_mixed_fa2": "#4A90E2",        # Vibrant blue
    "flashinfer_mixed_fa3": "#F5A623",        # Warm orange
    "flashinfer_separated_fa2": "#9B59B6",    # Rich purple
    "flashinfer_separated_fa3": "#27AE60",    # Fresh green
    # Grouped approach colors
    "fi2": "#4A90E2",                         # Vibrant blue (same as mix2)
    "fi3": "#F5A623",                         # Warm orange (same as mix3)
    "fi_dec": "#9B59B6",                      # Rich purple (same as sep2)
}

# Approach groupings by workload type
# For prefill: mix2==sep2 -> fi2, mix3==sep3 -> fi3
# For decode: sep2==sep3 -> fi_dec
APPROACH_GROUPS = {
    "prefill": {
        "fi2": ["flashinfer_mixed_fa2", "flashinfer_separated_fa2"],
        "fi3": ["flashinfer_mixed_fa3", "flashinfer_separated_fa3"],
    },
    "decode": {
        "fi_dec": ["flashinfer_separated_fa2", "flashinfer_separated_fa3"],
    },
}


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple.

    Args:
        hex_color: Color in hex format (e.g., "#FF5733" or "FF5733")

    Returns:
        Tuple of (red, green, blue) values (0-255)

    Example:
        >>> hex_to_rgb("#FF5733")
        (255, 87, 51)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color.

    Args:
        rgb: Tuple of (red, green, blue) values (0-255)

    Returns:
        Color in hex format with leading #

    Example:
        >>> rgb_to_hex((255, 87, 51))
        '#ff5733'
    """
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def adjust_color_saturation(hex_color: str, intensity: float) -> str:
    """Adjust color brightness/saturation based on intensity.

    Higher intensity = brighter, more vibrant color
    Lower intensity = blended towards background

    This is useful for creating color gradients in heatmaps where
    the same base color needs different intensities.

    Args:
        hex_color: Base color in hex format
        intensity: Value between 0 and 1 (1 = full brightness)

    Returns:
        Adjusted color in hex format

    Example:
        >>> base = "#4A90E2"
        >>> adjust_color_saturation(base, 0.5)  # Medium intensity
        >>> adjust_color_saturation(base, 1.0)  # Full brightness
    """
    r, g, b = hex_to_rgb(hex_color)

    # Blend towards dark gray background instead of black
    # This keeps colors visible even at low intensity
    bg = 45  # Dark background gray value
    min_intensity = 0.5  # Minimum brightness floor
    adjusted_intensity = min_intensity + (1 - min_intensity) * intensity

    new_r = int(bg + (r - bg) * adjusted_intensity)
    new_g = int(bg + (g - bg) * adjusted_intensity)
    new_b = int(bg + (b - bg) * adjusted_intensity)

    return rgb_to_hex((new_r, new_g, new_b))
