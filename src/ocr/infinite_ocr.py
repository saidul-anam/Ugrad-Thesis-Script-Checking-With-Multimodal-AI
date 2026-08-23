"""
Infinite-OCR / Nanonets / Vision-Language Model Backend for Handwritten Document Transcription.
Supports full-page handwritten English text extraction natively on GPU and CPU.
"""

import os
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from PIL import Image
import torch

# Direct all model downloads and caches to D: drive (146+ GB available)
os.environ["HF_HOME"] = "D:\\hf_cache\\huggingface"
os.environ["TRANSFORMERS_CACHE"] = "D:\\hf_cache\\huggingface\\hub"
os.environ["TORCH_HOME"] = "D:\\hf_cache\\torch"

from src.ocr.base import OCRBackend
from src.ocr.schemas import OCRResult, OCRContent, InputMetadata, ExecutionMetadata, OCRSegment
from src.preprocessing.image import safe_normalize_text
from src.confidence.estimator import ConfidenceEstimator
from src.utils.hashing import compute_image_hash
from src.utils.env_info import get_peak_gpu_memory_mb, get_process_memory_mb


class InfiniteOCRBackend(OCRBackend):
    """
    Infinite-OCR Backend utilizing Vision-Language Models (e.g. nanonets/Nanonets-OCR2-3B or Qwen2.5-VL).
    Uses the exact training prompt format to prevent language hallucination and Chinese token fallback.
    """

    # Official fine-tuning prompt for Nanonets-OCR2-3B / Qwen2.5-VL OCR
    PROMPT_PLAIN_TEXT = (
        "Extract the text from the above document as if you were reading it naturally. "
        "The document is handwritten English. Transcribe all handwritten English text verbatim."
    )

    def __init__(
        self,
        model_name: str = "nanonets/Nanonets-OCR2-3B",
        device: str = "auto",
        max_new_tokens: int = 2048,
        torch_dtype: str = "auto"
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # Hardware resolution
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[InfiniteOCRBackend] Initializing {self.model_name} on {self.device}...")
        self.model = None
        self.processor = None
        self._init_model(torch_dtype)

    def _init_model(self, torch_dtype: str) -> None:
        try:
            from transformers import AutoProcessor
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
            except ImportError:
                from transformers import AutoModelForImageTextToText as ModelClass

            if self.device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                load_kwargs = {
                    "torch_dtype": dtype,
                    "device_map": "auto",
                }
            else:
                dtype = torch.float32
                load_kwargs = {
                    "torch_dtype": dtype,
                    "low_cpu_mem_usage": True,
                }

            self.model = ModelClass.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            if self.device == "cpu" and not hasattr(self.model, "hf_device_map"):
                self.model = self.model.to("cpu")
            self.model.eval()

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            print(f"[InfiniteOCRBackend] Loaded {self.model_name} successfully.")
        except Exception as e:
            print(f"[InfiniteOCRBackend] Model initialization deferred or failed: {e}")

    def extract(
        self,
        image: Image.Image,
        image_path: str,
        script_id: str,
        **kwargs: Any
    ) -> OCRResult:
        start_time = time.perf_counter()
        image_hash = compute_image_hash(image)

        if self.model is None or self.processor is None:
            raise RuntimeError(
                f"Model '{self.model_name}' is not initialized. Ensure required weights/packages are available."
            )

        rgb_img = image.convert("RGB")

        # Official Qwen2.5-VL / Nanonets message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_img},
                    {"type": "text", "text": self.PROMPT_PLAIN_TEXT},
                ],
            }
        ]

        try:
            from qwen_vl_utils import process_vision_info
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except Exception:
            # Fallback when qwen_vl_utils is not installed
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )

        # Send inputs to model device
        model_dev = next(self.model.parameters()).device
        inputs = inputs.to(model_dev)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=1.05,
                do_sample=False,
            )

        # Extract generated tokens
        in_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[0][in_len:]

        raw_text = self.processor.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()

        normalized_text = safe_normalize_text(raw_text)

        # Derive observable confidence signal
        derived_conf = ConfidenceEstimator.estimate_derived_confidence(normalized_text)
        conf_type = "derived" if derived_conf is not None else "none"
        conf_avail = derived_conf is not None

        elapsed_time = round(time.perf_counter() - start_time, 4)

        return OCRResult(
            script_id=script_id,
            input=InputMetadata(
                image_path=image_path,
                image_hash=image_hash,
                image_size=[image.width, image.height]
            ),
            ocr=OCRContent(
                backend="infinite_ocr",
                model=self.model_name,
                raw_text=raw_text,
                normalized_text=normalized_text,
                confidence=derived_conf,
                confidence_type=conf_type,
                confidence_available=conf_avail,
                segments=[]
            ),
            metadata=ExecutionMetadata(
                processing_time_seconds=elapsed_time,
                device=str(model_dev),
                gpu_memory_peak_mb=get_peak_gpu_memory_mb(),
                cpu_memory_mb=get_process_memory_mb(),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        )
