from typing import Optional
from src.core.config import PipelineConfig
from src.engine.base_engine import BaseVLMEngine
from src.engine.gemma_cuda_engine import GemmaCudaEngine
from src.engine.mock_engine import MockGemmaEngine


def create_engine(config: PipelineConfig, force_mock: bool = False) -> BaseVLMEngine:
    """
    Instantiate the appropriate Vision-Language Engine.
    
    If `force_mock=True`, returns MockGemmaEngine for local testing without GPU.
    Otherwise, initializes GemmaCudaEngine for CUDA execution (e.g. RTX 5090).
    """
    if force_mock:
        print(f"[EngineFactory] Initializing Mock Engine for model '{config.model.model_id}'...")
        return MockGemmaEngine(model_id=config.model.model_id)

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

