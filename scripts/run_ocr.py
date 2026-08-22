"""
Command-Line Interface for Single-Image and Batch English Handwritten Exam OCR.
Usage:
    python scripts/run_ocr.py --input data/samples/exam_001.jpg
    python scripts/run_ocr.py --input data/images/ --output outputs/ocr/
"""

import os
import sys
import argparse
import yaml
import glob
from pathlib import Path
from typing import List

# Add workspace to path
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.preprocessing.image import ImagePreprocessor
from src.ocr.factory import get_ocr_backend
from src.ocr.schemas import OCRResult


def load_yaml_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def process_single_image(
    image_path: str,
    preprocessor: ImagePreprocessor,
    ocr_backend,
    output_dir: str = "outputs/ocr",
    script_id: str = None
) -> OCRResult:
    path = Path(image_path)
    if not script_id:
        script_id = path.stem

    # 1. Preprocess while preserving original image
    orig_img, proc_img, img_hash = preprocessor.load_and_preprocess(str(path))

    # 2. Extract OCR
    result: OCRResult = ocr_backend.extract(
        image=proc_img,
        image_path=str(path),
        script_id=script_id
    )

    # 3. Save JSON Output
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{script_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(result.to_json(indent=2))

    return result


def main():
    parser = argparse.ArgumentParser(description="Single-Pass Handwritten English Exam OCR")
    parser.add_argument("--input", required=True, help="Path to single image file or directory of images")
    parser.add_argument("--output", default="outputs/ocr", help="Output directory for JSON results")
    parser.add_argument("--config", default="configs/ocr.yaml", help="Path to OCR config YAML")
    parser.add_argument("--backend", default=None, help="Override OCR backend (infinite_ocr, trocr, easyocr, mock)")
    parser.add_argument("--model", default=None, help="Override model checkpoint")
    parser.add_argument("--device", default=None, help="Override compute device (cuda, cpu, auto)")
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal output")
    args = parser.parse_args()

    # Load configuration
    ocr_cfg = load_yaml_config(args.config).get("ocr", {})
    prep_cfg = load_yaml_config("configs/preprocessing.yaml").get("preprocessing", {})

    if args.backend:
        ocr_cfg["backend"] = args.backend
    if args.model:
        ocr_cfg["model_name"] = args.model
    if args.device:
        ocr_cfg["device"] = args.device

    preprocessor = ImagePreprocessor(
        enabled=prep_cfg.get("enabled", True),
        resize=prep_cfg.get("resize", True),
        max_image_side=prep_cfg.get("max_image_side", 2048),
        deskew=prep_cfg.get("deskew", False),
        denoise=prep_cfg.get("denoise", False),
        contrast_normalization=prep_cfg.get("contrast_normalization", False)
    )

    backend = get_ocr_backend(ocr_cfg)

    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.pdf"]:
            image_files.extend(list(input_path.glob(ext)))
            image_files.extend(list(input_path.glob(ext.upper())))
    else:
        print(f"Error: Input path {args.input} not found.")
        sys.exit(1)

    if not image_files:
        print(f"No image files found in {args.input}")
        sys.exit(0)

    for img_file in image_files:
        res = process_single_image(
            image_path=str(img_file),
            preprocessor=preprocessor,
            ocr_backend=backend,
            output_dir=args.output
        )

        if not args.quiet:
            conf_str = f"{res.ocr.confidence:.4f}" if res.ocr.confidence is not None else "N/A (Confidence unavailable)"
            print("\n" + "=" * 40)
            print("OCR RESULT")
            print("=" * 40)
            print(f"Script ID:       {res.script_id}")
            print(f"Backend:         {res.ocr.backend} ({res.ocr.model})")
            print(f"Language:        English")
            print(f"Confidence:      {conf_str} (type: {res.ocr.confidence_type})")
            print(f"Processing time: {res.metadata.processing_time_seconds:.2f} seconds")
            print(f"Device:          {res.metadata.device}")
            print("Extracted text:")
            print("-" * 40)
            print(res.ocr.normalized_text if res.ocr.normalized_text else "[Empty Extraction]")
            print("-" * 40)
            print(f"Saved to:        {args.output}/{res.script_id}.json\n")


if __name__ == "__main__":
    main()
