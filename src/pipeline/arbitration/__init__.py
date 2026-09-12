"""Stage 3b evidence-fused handwriting ambiguity gate."""

from src.pipeline.arbitration.gate import EvidenceArbitrationGate
from src.pipeline.arbitration.localizer import attribute_error_to_page

__all__ = ["EvidenceArbitrationGate", "attribute_error_to_page"]
