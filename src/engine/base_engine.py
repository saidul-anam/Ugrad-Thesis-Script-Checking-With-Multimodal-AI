from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from PIL import Image


class BaseVLMEngine(ABC):
    """Abstract interface for Multimodal Vision-Language and Text Inference Engines."""

    @abstractmethod
    def generate_multimodal(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 3072,
        thinking_mode: bool = False,
        **kwargs: Any
    ) -> str:
        """Generate response from an input image and prompt."""
        pass

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 3072,
        thinking_mode: bool = False,
        **kwargs: Any
    ) -> str:
        """Generate response for text-only input."""
        pass

    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """Return engine metadata, hardware profile, and runtime configuration."""
        pass

    def get_last_usage(self) -> Dict[str, Any]:
        """Return token usage of the most recent inference request."""
        return getattr(self, "last_usage", {})

    def format_last_usage(self) -> str:
        """Format last token usage as a concise context string, e.g.: [Context: 1,185/4,096 tokens (28.9%) | in:1,050 out:135]"""
        usage = self.get_last_usage()
        if not usage or not usage.get("total_tokens"):
            return ""
        total = usage.get("total_tokens", 0)
        max_ctx = usage.get("context_window", 4096)
        pct = (total / max_ctx * 100) if max_ctx > 0 else 0
        prompt_t = usage.get("prompt_tokens", 0)
        comp_t = usage.get("completion_tokens", 0)
        return f"[Context: {total:,}/{max_ctx:,} tokens ({pct:.1f}%) | in:{prompt_t:,} out:{comp_t:,}]"
