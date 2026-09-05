from typing import Optional
from src.core.config import PipelineConfig
from src.engine.base_engine import BaseVLMEngine
from src.engine.gemma_cuda_engine import GemmaCudaEngine
from src.engine.mock_engine import MockGemmaEngine
from src.engine.local_api_engine import LocalAPIEngine


def create_engine(
    config: PipelineConfig,
    force_mock: bool = False,
    force_api: bool = False,
    api_url: Optional[str] = None
) -> BaseVLMEngine:
    """
    Instantiate the appropriate Vision-Language Engine.
    
    Priority:
      1. force_mock -> MockGemmaEngine (fast CPU simulation)
      2. force_api / api_url / engine_type == "api" -> LocalAPIEngine (OpenAI-compatible server like LM Studio)
      3. Default -> GemmaCudaEngine (direct CUDA loading on RTX 5090)
    """
    if force_mock:
        print(f"[EngineFactory] Initializing Mock Engine for model '{config.model.model_id}'...")
        return MockGemmaEngine(model_id=config.model.model_id)

    resolved_api_url = api_url or getattr(config.model, "api_url", None)
    is_api_mode = force_api or getattr(config.model, "engine_type", "cuda") == "api" or resolved_api_url is not None

    if is_api_mode:
        target_url = resolved_api_url or "http://localhost:1234/v1"
        print(f"[EngineFactory] Initializing Local API Engine at '{target_url}'...")
        return LocalAPIEngine(api_url=target_url, model_id=config.model.model_id)

    print(f"[EngineFactory] Initializing Gemma 4 31B IT CUDA Engine '{config.model.model_id}'...")
    return GemmaCudaEngine(
        model_id=config.model.model_id,
        quantization=config.model.quantization,
        torch_dtype=config.model.torch_dtype,
        device_map=config.model.device_map,
        trust_remote_code=config.model.trust_remote_code,
        use_flash_attention_2=config.model.use_flash_attention_2,
        attn_implementation=getattr(config.model, "attn_implementation", "sdpa")
    )

