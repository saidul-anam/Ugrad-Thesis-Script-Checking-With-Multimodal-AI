"""
Infinite-OCR / Nanonets / Vision-Language Model Backend for Handwritten Document Transcription.
Supports full-page handwritten English text extraction directly.
"""

import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from PIL import Image
import torch

from src.ocr.base import OCRBackend
from src.ocr.schemas import OCRResult, OCRContent, InputMetadata, ExecutionMetadata, OCRSegment
from src.preprocessing.image import safe_normalize_text
from src.confidence.estimator import ConfidenceEstimator
from src.utils.hashing import compute_image_hash
from src.utils.env_info import get_peak_gpu_memory_mb, get_process_memory_mb


class InfiniteOCRBackend(OCRBackend):
    """
    Infinite-OCR Backend utilizing Vision-Language Models (e.g. nanonets/Nanonets-OCR2-3B or Qwen2.5-VL).
    Reads full-page handwriting natively without requiring line-by-line cropping.
    """

    PROMPT_PLAIN_TEXT = (
        "Extract all the handwritten English text from this examination answer image exactly as written, "
        "in natural reading order. Return only the extracted student text with no commentary, translation, or correction."
    )

    def __init__(
        self,
        model_name: str = "nanonets/Nanonets-OCR2-3B",
        device: str = "auto",
        max_new_tokens: int = 4096,
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
        self.tokenizer = None
        self._init_model(torch_dtype)

    def _init_model(self, torch_dtype: str) -> None:
        try:
            from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

            dtype = "auto" if self.device == "cuda" else torch.float32

            load_kwargs = {
                "torch_dtype": dtype,
            }
            if self.device == "cuda":
                load_kwargs["device_map"] = "auto"

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            print(f"[InfiniteOCRBackend] Model loaded successfully.")
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

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.PROMPT_PLAIN_TEXT},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Extract generated tokens
        in_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[0][in_len:]

        raw_text = self.processor.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()

        normalized_text = safe_normalize_text(raw_text)

        # Derive observable confidence signal since autoregressive models do not output scalar OCR confidence
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
                device=self.device,
                gpu_memory_peak_mb=get_peak_gpu_memory_mb(),
                cpu_memory_mb=get_process_memory_mb(),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        )
