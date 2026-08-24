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
        max_new_tokens: int = 4096,
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
        max_new_tokens: int = 4096,
        thinking_mode: bool = False,
        **kwargs: Any
    ) -> str:
        """Generate response for text-only input."""
        pass

    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """Return engine metadata, hardware profile, and runtime configuration."""
        pass
