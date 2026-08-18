"""
vLLM backend client supporting Vision-Language Models (e.g. Gemma-3),
quantization (W4A16, AWQ, GPTQ) and FP8 KV Cache.
"""

from typing import Any, Dict, List, Optional
from PIL import Image
import os

try:
    from .base import ModelClient
    from ..schemas import ModelConfig
except (ImportError, ValueError):
    from model_client.base import ModelClient
    from schemas import ModelConfig


class VLLMClient(ModelClient):
    """Client implementing inference via vLLM engine."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.llm = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Lazily initialize vLLM LLM instance."""
        try:
            from vllm import LLM, SamplingParams
            self._SamplingParams = SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed. Please run `pip install vllm` with internet enabled "
                "or switch model backend to 'transformers' or 'mock'."
            ) from e

        engine_args: Dict[str, Any] = {
            "model": self.config.checkpoint,
            "trust_remote_code": True,
            "max_model_len": self.config.max_tokens or 4096,
        }

        # Quantization settings
        quant = (self.config.quantization or "").lower()
        if quant == "w4a16" or "awq" in quant:
            engine_args["quantization"] = "awq"
        elif "gptq" in quant:
            engine_args["quantization"] = "gptq"
        elif "bitsandbytes" in quant:
            engine_args["quantization"] = "bitsandbytes"

        # KV cache quantization
        kv_cache = (self.config.kv_cache or "").lower()
        if kv_cache == "fp8":
            engine_args["kv_cache_dtype"] = "fp8"

        self.llm = LLM(**engine_args)

    def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        temp = temperature if temperature is not None else self.config.temperature
        max_t = max_tokens if max_tokens is not None else self.config.max_tokens

        sampling_params = self._SamplingParams(
            temperature=temp,
            max_tokens=max_t,
            **kwargs
        )

        if images and len(images) > 0:
            # Multi-modal prompt payload for vLLM
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": images[0] if len(images) == 1 else images
                }
            }
            outputs = self.llm.generate([inputs], sampling_params=sampling_params)
        else:
            outputs = self.llm.generate([prompt], sampling_params=sampling_params)

        if outputs and len(outputs) > 0 and outputs[0].outputs:
            return outputs[0].outputs[0].text
        return ""
