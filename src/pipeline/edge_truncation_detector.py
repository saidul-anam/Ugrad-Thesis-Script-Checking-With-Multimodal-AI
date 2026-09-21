"""
Re-export edge truncation detector from src.utils to preserve backward compatibility.
"""

from src.utils.edge_truncation_detector import (
    extract_line_end_tokens,
    stitch_cross_line_truncations,
    is_right_edge_truncation,
)

__all__ = [
    "extract_line_end_tokens",
    "stitch_cross_line_truncations",
    "is_right_edge_truncation",
]
