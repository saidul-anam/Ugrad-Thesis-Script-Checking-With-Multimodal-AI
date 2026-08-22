"""
TrOCR Handwriting OCR Backend (microsoft/trocr-base-handwritten).
"""

import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from PIL import Image
import torch
import numpy as np

from src.ocr.base import OCRBackend
from src.ocr.schemas import OCRResult, OCRContent, InputMetadata, ExecutionMetadata, OCRSegment
from src.preprocessing.image import safe_normalize_text
from src.confidence.aggregation import aggregate_confidence
from src.utils.hashing import compute_image_hash
from src.utils.env_info import get_peak_gpu_memory_mb, get_process_memory_mb


class TrOCRBackend(OCRBackend):
    """
    TrOCR Backend using line segmentation or single-line inference.
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        device: str = "auto",
        max_new_tokens: int = 256
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[TrOCRBackend] Initializing {self.model_name} on {self.device}...")
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._init_model()

    def _init_model(self) -> None:
        try:
            from transformers import AutoImageProcessor, XLMRobertaTokenizer, VisionEncoderDecoderModel

            self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
            if self.device == "cuda":
                self.model = self.model.half()
            self.model.eval()
            print(f"[TrOCRBackend] Loaded {self.model_name} successfully.")
        except Exception as e:
            print(f"[TrOCRBackend] Model initialization deferred or failed: {e}")

    def extract(
        self,
        image: Image.Image,
        image_path: str,
        script_id: str,
        **kwargs: Any
    ) -> OCRResult:
        start_time = time.perf_counter()
        image_hash = compute_image_hash(image)

        if self.model is None or self.image_processor is None or self.tokenizer is None:
            raise RuntimeError(f"TrOCR model '{self.model_name}' is not initialized.")

        # If single line or crop
        rgb_img = image.convert("RGB")
        pixel_values = self.image_processor(images=rgb_img, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                pixel_values,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True
            )

        sequences = output.sequences
        raw_text = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0].strip()
        normalized_text = safe_normalize_text(raw_text)

        # Compute sequence log-probability confidence if scores present
        confidence = None
        if hasattr(output, "scores") and output.scores:
            try:
                probs = []
                for step_logits in output.scores:
                    step_probs = torch.softmax(step_logits, dim=-1)
                    max_p, _ = torch.max(step_probs, dim=-1)
                    probs.append(float(max_p.cpu().item()))
                confidence = float(np.mean(probs)) if probs else 0.85
            except Exception:
                confidence = 0.85
        else:
            confidence = 0.85

        elapsed_time = round(time.perf_counter() - start_time, 4)

        return OCRResult(
            script_id=script_id,
            input=InputMetadata(
                image_path=image_path,
                image_hash=image_hash,
                image_size=[image.width, image.height]
            ),
            ocr=OCRContent(
                backend="trocr",
                model=self.model_name,
                raw_text=raw_text,
                normalized_text=normalized_text,
                confidence=round(confidence, 4) if confidence is not None else None,
                confidence_type="raw" if confidence is not None else "none",
                confidence_available=confidence is not None,
                segments=[OCRSegment(text=raw_text, confidence=confidence)]
            ),
            metadata=ExecutionMetadata(
                processing_time_seconds=elapsed_time,
                device=self.device,
                gpu_memory_peak_mb=get_peak_gpu_memory_mb(),
                cpu_memory_mb=get_process_memory_mb(),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        )
