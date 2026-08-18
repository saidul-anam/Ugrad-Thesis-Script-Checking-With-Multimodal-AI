"""
Model Client Factory.
Supports:
- 'easyocr' (Real local OCR on physical script images)
- 'gemini' (Real multimodal VLM via Google Gemini API)
- 'transformers' (Local HuggingFace model, e.g. Gemma 3)
- 'vllm' (High-throughput serving)
- 'mock' (Deterministic offline unit tests)
"""

try:
    from .base import ModelClient
    from .mock_client import MockClient
    from .easyocr_client import EasyOCRClient
    from .gemini_client import GeminiClient
    from ..schemas import ModelConfig
except (ImportError, ValueError):
    from model_client.base import ModelClient
    from model_client.mock_client import MockClient
    from model_client.easyocr_client import EasyOCRClient
    from model_client.gemini_client import GeminiClient
    from schemas import ModelConfig


def get_model_client(config: ModelConfig) -> ModelClient:
    """Instantiate appropriate model client based on config backend."""
    backend = (config.backend or "mock").lower()

    if backend in ["easyocr", "real_ocr", "ocr"]:
        return EasyOCRClient(config=config)
    elif backend in ["gemini", "google_genai"]:
        return GeminiClient(config=config)
    elif backend == "mock":
        return MockClient(config=config)
    elif backend == "vllm":
        try:
            from .vllm_client import VLLMClient
            return VLLMClient(config=config)
        except Exception:
            try:
                from .transformers_client import TransformersClient
                return TransformersClient(config=config)
            except Exception:
                return EasyOCRClient(config=config)
    elif backend == "transformers":
        try:
            from .transformers_client import TransformersClient
            return TransformersClient(config=config)
        except Exception:
            return EasyOCRClient(config=config)
    else:
        return EasyOCRClient(config=config)
