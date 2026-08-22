"""
TrOCR Handwriting OCR Backend (microsoft/trocr-base-handwritten).
Supports full-page line-segmented extraction as well as single-line inference.
"""

import os
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image
import torch
import numpy as np

# Direct all model downloads and caches to D: drive (146+ GB available)
os.environ["HF_HOME"] = "D:\\hf_cache\\huggingface"
os.environ["TRANSFORMERS_CACHE"] = "D:\\hf_cache\\huggingface\\hub"
os.environ["TORCH_HOME"] = "D:\\hf_cache\\torch"

from src.ocr.base import OCRBackend
from src.ocr.schemas import OCRResult, OCRContent, InputMetadata, ExecutionMetadata, OCRSegment
from src.preprocessing.image import safe_normalize_text
from src.confidence.aggregation import aggregate_confidence
from src.utils.hashing import compute_image_hash
from src.utils.env_info import get_peak_gpu_memory_mb, get_process_memory_mb


class TrOCRBackend(OCRBackend):
    """
    TrOCR Backend with CRAFT-based line segmentation for multi-line handwritten pages.
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        device: str = "auto",
        max_new_tokens: int = 128
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[TrOCRBackend] Initializing {self.model_name} on {self.device}...")
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.craft_reader = None
        self._init_model()
        self._init_craft()

    def _init_model(self) -> None:
        try:
            from transformers import AutoImageProcessor, RobertaTokenizer, VisionEncoderDecoderModel

            self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
            if self.device == "cuda":
                self.model = self.model.half()
            self.model.eval()
            print(f"[TrOCRBackend] Loaded {self.model_name} successfully with RobertaTokenizer.")
        except Exception as e:
            print(f"[TrOCRBackend] Model initialization deferred or failed: {e}")

    def _init_craft(self) -> None:
        """Initialize CRAFT detector for line segmentation."""
        try:
            import easyocr
            use_gpu = (self.device == "cuda")
            self.craft_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        except Exception as e:
            print(f"[TrOCRBackend] CRAFT reader deferred or failed: {e}")
            self.craft_reader = None

    def _cluster_boxes_into_lines(self, boxes: List[List[List[int]]]) -> List[List[List[List[int]]]]:
        """Cluster word boxes into horizontal text lines."""
        if not boxes:
            return []

        box_centers = []
        for box in boxes:
            y_coords = [pt[1] for pt in box]
            y_center = sum(y_coords) / len(y_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)
            box_centers.append((y_center, y_min, y_max, box))

        box_centers.sort(key=lambda x: x[0])
        heights = [bc[2] - bc[1] for bc in box_centers]
        median_height = sorted(heights)[len(heights) // 2] if heights else 30
        line_gap_threshold = max(15, median_height * 0.6)

        lines = []
        current_line = [box_centers[0]]
        for i in range(1, len(box_centers)):
            prev_center = current_line[-1][0]
            curr_center = box_centers[i][0]
            if abs(curr_center - prev_center) <= line_gap_threshold:
                current_line.append(box_centers[i])
            else:
                lines.append(current_line)
                current_line = [box_centers[i]]
        lines.append(current_line)

        result = []
        for line in lines:
            sorted_boxes = sorted(line, key=lambda x: min(pt[0] for pt in x[3]))
            result.append([item[3] for item in sorted_boxes])
        return result

    def _transcribe_crop(self, crop_img: Image.Image) -> Tuple[str, float]:
        """Transcribe a single horizontal line crop."""
        w, h = crop_img.size
        if h <= 0 or w <= 0:
            return "", 0.0

        target_h = 48
        ratio = target_h / float(h)
        new_w = max(1, int(w * ratio))
        resized = crop_img.resize((new_w, target_h), Image.Resampling.BILINEAR)

        pixel_values = self.image_processor(images=resized.convert("RGB"), return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                pixel_values,
                max_new_tokens=self.max_new_tokens,
                num_beams=2,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True
            )

        seq = output.sequences[0]
        text = self.tokenizer.decode(seq, skip_special_tokens=True).strip()

        # Score confidence
        conf = 0.85
        if hasattr(output, "scores") and output.scores:
            try:
                probs = []
                for step_logits in output.scores:
                    step_probs = torch.softmax(step_logits, dim=-1)
                    max_p, _ = torch.max(step_probs, dim=-1)
                    probs.append(float(max_p.cpu().item()))
                conf = float(np.mean(probs)) if probs else 0.85
            except Exception:
                conf = 0.85
        return text, conf

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

        if self.model is None or self.image_processor is None or self.tokenizer is None:
            raise RuntimeError(f"TrOCR model '{self.model_name}' is not initialized.")

        rgb_img = image.convert("RGB")
        w, h = rgb_img.size

        # If already a single line (small height)
        if h <= 80:
            text, conf = self._transcribe_crop(rgb_img)
            segments = [OCRSegment(text=text, confidence=conf)]
            raw_text = text
            agg_conf = conf
        else:
            # Multi-line full page: use CRAFT line detector
            boxes = []
            if self.craft_reader:
                scale = 1280.0 / max(w, h) if max(w, h) > 1280 else 1.0
                scaled_img = rgb_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR) if scale != 1.0 else rgb_img
                np_img = np.array(scaled_img)
                try:
                    raw_dets = self.craft_reader.readtext(np_img, canvas_size=1280, mag_ratio=1.0, paragraph=False)
                    for bbox, det_text, det_conf in raw_dets:
                        if str(det_text).strip():
                            scaled_bbox = [[int(pt[0] / scale), int(pt[1] / scale)] for pt in bbox]
                            boxes.append(scaled_bbox)
                except Exception:
                    boxes = []

            if not boxes:
                # Fallback: transcribe direct crop
                text, conf = self._transcribe_crop(rgb_img)
                segments = [OCRSegment(text=text, confidence=conf)]
                raw_text = text
                agg_conf = conf
            else:
                lines = self._cluster_boxes_into_lines(boxes)
                segments = []
                line_texts = []
                confs = []
                weights = []

                for line_boxes in lines:
                    all_x = [pt[0] for b in line_boxes for pt in b]
                    all_y = [pt[1] for b in line_boxes for pt in b]
                    x_min = max(0, min(all_x) - 8)
                    y_min = max(0, min(all_y) - 8)
                    x_max = min(w, max(all_x) + 8)
                    y_max = min(h, max(all_y) + 8)

                    crop = rgb_img.crop((x_min, y_min, x_max, y_max))
                    l_text, l_conf = self._transcribe_crop(crop)
                    if l_text:
                        line_texts.append(l_text)
                        confs.append(l_conf)
                        weights.append(float(len(l_text)))
                        segments.append(OCRSegment(
                            text=l_text,
                            confidence=round(l_conf, 4),
                            bbox=[[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                        ))

                raw_text = " ".join(line_texts)
                agg_conf = aggregate_confidence(confs, weights=weights, method=aggregation_method)

        normalized_text = safe_normalize_text(raw_text)
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
                confidence=round(agg_conf, 4) if agg_conf is not None else None,
                confidence_type="raw" if agg_conf is not None else "none",
                confidence_available=agg_conf is not None,
                segments=segments
            ),
            metadata=ExecutionMetadata(
                processing_time_seconds=elapsed_time,
                device=self.device,
                gpu_memory_peak_mb=get_peak_gpu_memory_mb(),
                cpu_memory_mb=get_process_memory_mb(),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        )
