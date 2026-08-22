"""
EasyOCR Baseline Backend.
"""

import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from PIL import Image
import numpy as np

from src.ocr.base import OCRBackend
from src.ocr.schemas import OCRResult, OCRContent, InputMetadata, ExecutionMetadata, OCRSegment
from src.preprocessing.image import safe_normalize_text
from src.confidence.aggregation import aggregate_confidence
from src.utils.hashing import compute_image_hash
from src.utils.env_info import get_peak_gpu_memory_mb, get_process_memory_mb


class EasyOCRBackend(OCRBackend):
    """
    EasyOCR Backend (CRAFT detector + BiLSTM recognizer).
    """

    def __init__(self, device: str = "auto", language: str = "en"):
        self.language = language
        self.device = device
        self.reader = None
        self._init_reader()

    def _init_reader(self) -> None:
        try:
            import easyocr
            import torch
            use_gpu = torch.cuda.is_available() if self.device == "auto" else (self.device == "cuda")
            self.reader = easyocr.Reader([self.language], gpu=use_gpu, verbose=False)
            self.device_str = "cuda" if use_gpu else "cpu"
        except Exception as e:
            print(f"[EasyOCRBackend] Initialization deferred or failed: {e}")
            self.reader = None
            self.device_str = "cpu"

    def extract(
        self,
        image: Image.Image,
        image_path: str,
        script_id: str,
        aggregation_method: str = "length_weighted_mean",
        **kwargs: Any
    ) -> OCRResult:
        start_time = time.perf_counter()
        image_hash = compute_image_hash(image)

        if self.reader is None:
            raise RuntimeError("EasyOCR reader is not initialized.")

        np_img = np.array(image.convert("RGB"))
        ocr_results = self.reader.readtext(np_img, paragraph=False)

        segments = []
        confs = []
        weights = []
        raw_lines = []

        for bbox, text, conf in ocr_results:
            clean_t = str(text).strip()
            if clean_t:
                raw_lines.append(clean_t)
                c_val = float(conf)
                confs.append(c_val)
                weights.append(float(len(clean_t)))
                segments.append(OCRSegment(
                    text=clean_t,
                    confidence=round(c_val, 4),
                    bbox=[[int(p[0]), int(p[1])] for p in bbox]
                ))

        raw_text = " ".join(raw_lines)
        normalized_text = safe_normalize_text(raw_text)

        agg_conf = aggregate_confidence(confs, weights=weights, method=aggregation_method)

        elapsed_time = round(time.perf_counter() - start_time, 4)

        return OCRResult(
            script_id=script_id,
            input=InputMetadata(
                image_path=image_path,
                image_hash=image_hash,
                image_size=[image.width, image.height]
            ),
            ocr=OCRContent(
                backend="easyocr",
                model="easyocr-craft-bilstm",
                raw_text=raw_text,
                normalized_text=normalized_text,
                confidence=round(agg_conf, 4) if agg_conf is not None else None,
                confidence_type="raw" if agg_conf is not None else "none",
                confidence_available=agg_conf is not None,
                segments=segments
            ),
            metadata=ExecutionMetadata(
                processing_time_seconds=elapsed_time,
                device=self.device_str,
                gpu_memory_peak_mb=get_peak_gpu_memory_mb(),
                cpu_memory_mb=get_process_memory_mb(),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        )
