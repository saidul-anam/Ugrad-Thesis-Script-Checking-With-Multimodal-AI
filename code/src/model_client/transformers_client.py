"""
Hugging Face Transformers fallback backend client supporting Vision-Language Models
(e.g., AutoModelForVision2Seq, Gemma3ForConditionalGeneration) with BitsAndBytes quantization.
"""

from typing import Any, Dict, List, Optional
from PIL import Image
import torch

try:
    from .base import ModelClient
    from ..schemas import ModelConfig
except (ImportError, ValueError):
    from model_client.base import ModelClient
    from schemas import ModelConfig


class TransformersClient(ModelClient):
    """Client implementing inference via Hugging Face transformers pipeline."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.processor = None
        self.model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "Transformers is not installed. Please run `pip install transformers accelerate bitsandbytes`."
            ) from e

        quant = (self.config.quantization or "").lower()
        bnb_config = None

        if "4bit" in quant or quant == "w4a16":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        elif "8bit" in quant:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        checkpoint = self.config.checkpoint
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                checkpoint,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        except Exception:
            # Fallback to CausalLM if not a Vision2Seq model
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

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

        do_sample = temp > 0.0

        if images and len(images) > 0:
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        else:
            inputs = self.processor(text=prompt, return_tensors="pt")

        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_t,
            "do_sample": do_sample,
            **kwargs
        }
        if do_sample:
            gen_kwargs["temperature"] = temp

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated_tokens = outputs[0][input_len:]
        decoded = self.processor.decode(generated_tokens, skip_special_tokens=True)
        return decoded
