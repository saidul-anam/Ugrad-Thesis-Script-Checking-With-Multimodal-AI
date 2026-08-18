"""
Abstract base class for LLM / Vision-Language Model serving.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from PIL import Image


class ModelClient(ABC):
    """Abstract interface for model inference."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate completion for the given prompt and optional image(s).
        
        Args:
            prompt: Text prompt with instructions and schemas
            images: List of PIL images for multimodal models (or None for text-only)
            temperature: Sampling temperature (default from config)
            max_tokens: Max generated tokens
            **kwargs: Extra parameters
            
        Returns:
            Generated response string (expected to contain JSON or structured text)
        """
        pass
