"""Workload configuration for different benchmark types."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class WorkloadConfig:
    """Configuration for workload-specific axis labels and data columns.

    This allows us to unify decode, prefill, and mixed workload handling
    by using configuration instead of conditional logic.
    """
    workload_type: str
    x_axis_column: str
    y_axis_column: str
    x_axis_label: str
    y_axis_label: str

    # Optional prefill-specific columns for mixed workloads
    prefill_query_column: Optional[str] = None
    prefill_kv_column: Optional[str] = None

    @classmethod
    def for_workload(cls, workload_type: str) -> "WorkloadConfig":
        """Factory method to create appropriate config for workload type.

        Args:
            workload_type: Type of workload ("decode", "prefill", "mixed")

        Returns:
            WorkloadConfig instance with appropriate settings
        """
        if workload_type == "decode":
            return cls(
                workload_type="decode",
                x_axis_column="kv_length",
                y_axis_column="batch_size",
                x_axis_label="KV Length",
                y_axis_label="Batch Size"
            )
        elif workload_type == "prefill":
            return cls(
                workload_type="prefill",
                x_axis_column="kv_length",
                y_axis_column="query_length",
                x_axis_label="KV Length",
                y_axis_label="Query Length"
            )
        elif workload_type == "mixed":
            return cls(
                workload_type="mixed",
                x_axis_column="decode_kv",
                y_axis_column="decode_batch",
                x_axis_label="Decode KV Length",
                y_axis_label="Decode Batch Size",
                prefill_query_column="prefill_query",
                prefill_kv_column="prefill_kv"
            )
        else:
            raise ValueError(f"Unknown workload type: {workload_type}")

    def is_mixed(self) -> bool:
        """Check if this is a mixed workload configuration."""
        return self.workload_type == "mixed"

    def get_scenario_identifier(self, row) -> tuple:
        """Get unique scenario identifier from a DataFrame row.

        Args:
            row: DataFrame row with scenario data

        Returns:
            Tuple of (x_value, y_value) for this scenario
        """
        x_val = row[self.x_axis_column]
        y_val = row[self.y_axis_column]
        return (x_val, y_val)

    def format_x_axis_value(self, value: int) -> str:
        """Format x-axis value for display.

        Args:
            value: Raw x-axis value

        Returns:
            Formatted string
        """
        if value >= 1024 * 1024:
            return f"{value // (1024 * 1024)}M"
        elif value >= 1024:
            return f"{value // 1024}k"
        return str(value)

    def format_y_axis_value(self, value: int) -> str:
        """Format y-axis value for display.

        Args:
            value: Raw y-axis value

        Returns:
            Formatted string
        """
        # Batch sizes and query lengths are typically shown as-is
        # unless they're very large
        if value >= 1024 * 1024:
            return f"{value // (1024 * 1024)}M"
        elif value >= 1024:
            return f"{value // 1024}k"
        return str(value)


def get_axis_config(workload_type: str):
    """Convenience function to get workload configuration.

    Args:
        workload_type: Type of workload

    Returns:
        WorkloadConfig instance

    Example:
        >>> config = get_axis_config("decode")
        >>> print(config.x_axis_label)
        'KV Length'
    """
    return WorkloadConfig.for_workload(workload_type)
