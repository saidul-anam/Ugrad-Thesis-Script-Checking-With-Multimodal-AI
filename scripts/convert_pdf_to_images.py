"""
PDF to High-Resolution Image Converter for Handwritten Examination Scripts.

Usage:
    # 1. Convert top 5 PDFs from a local directory:
    python scripts/convert_pdf_to_images.py --pdf_dir data/raw_pdfs --output_dir data/samples --top 5

    # 2. Download from Google Drive folder and convert top 3 PDFs:
    python scripts/convert_pdf_to_images.py --gdrive_url "https://drive.google.com/drive/folders/1bIQMLlBYwyPb_f6tuycvk5iHz3lMkQYi?usp=sharing" --top 3

    # 3. Convert all PDFs in a folder at 200 DPI:
    python scripts/convert_pdf_to_images.py --pdf_dir data/raw_pdfs --output_dir data/images/ --dpi 200
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from typing import List, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF (fitz) is required. Install via: pip install pymupdf")
    sys.exit(1)


def download_gdrive_folder(gdrive_url: str, target_dir: str) -> None:
    """Download PDFs from Google Drive folder if gdown is available."""
    try:
        import gdown
    except ImportError:
        print("gdown is required for Google Drive downloads. Install via: pip install gdown")
        return

    os.makedirs(target_dir, exist_ok=True)
    print(f"\n[GDown] Downloading folder contents from: {gdrive_url}")
    print(f"[GDown] Target download directory: {target_dir}")
    try:
        gdown.download_folder(url=gdrive_url, output=target_dir, quiet=False, use_cookies=False)
        print("[GDown] Download complete!")
    except Exception as e:
        print(f"[GDown] Download error (You can also download the folder manually to {target_dir}): {e}")


def convert_pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 200,
    first_page_only: bool = False
) -> List[str]:
    """
    Renders all pages (or first page) of a PDF to high-resolution PNG images.
    Returns list of saved image paths.
    """
    pdf_file = Path(pdf_path)
    script_id = pdf_file.stem
    saved_images = []

    try:
        doc = fitz.open(str(pdf_file))
    except Exception as e:
        print(f"  [!] Failed to open {pdf_file.name}: {e}")
        return []

    num_pages = len(doc)
    # PyMuPDF default resolution is 72 DPI. Zoom factor = dpi / 72.0
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    pages_to_process = [0] if first_page_only else range(num_pages)

    for page_idx in pages_to_process:
        if page_idx >= num_pages:
            break
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        if num_pages == 1 or first_page_only:
            out_filename = f"{script_id}.png"
        else:
            out_filename = f"{script_id}_p{page_idx + 1:02d}.png"

        out_path = Path(output_dir) / out_filename
        pix.save(str(out_path))
        saved_images.append(str(out_path))

    doc.close()
    return saved_images


def main():
    parser = argparse.ArgumentParser(
        description="Convert Exam Script PDFs to High-Resolution Images for OCR Pipeline"
    )
    parser.add_argument(
        "--gdrive_url",
        type=str,
        default=None,
        help="Google Drive folder URL or ID to download PDFs from"
    )
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="data/raw_pdfs",
        help="Directory containing PDF files (default: data/raw_pdfs)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/samples",
        help="Directory to save extracted images (default: data/samples)"
    )
    parser.add_argument(
        "--top",
        "--limit",
        type=int,
        default=None,
        dest="top",
        help="Number of PDF scripts to process (e.g. --top 5). Omit to process all."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Image rendering resolution in DPI (default: 200 DPI, clean for handwriting)"
    )
    parser.add_argument(
        "--first_page_only",
        action="store_true",
        help="Convert only the first page of each script PDF"
    )

    args = parser.parse_args()

    # Step 1: Optional Google Drive download
    if args.gdrive_url:
        download_gdrive_folder(args.gdrive_url, args.pdf_dir)

    # Step 2: Discover PDF files
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"Error: PDF directory '{args.pdf_dir}' does not exist.")
        print(f"Tip: Place PDF files into '{args.pdf_dir}' or pass --gdrive_url to download.")
        sys.exit(1)

    pdf_files = sorted(list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF")))
    total_found = len(pdf_files)

    if total_found == 0:
        print(f"No PDF files found in '{args.pdf_dir}'.")
        sys.exit(0)

    # Step 3: Apply --top limit
    if args.top and args.top > 0:
        selected_pdfs = pdf_files[:args.top]
        print(f"\n[PDF Converter] Found {total_found} PDFs. Processing TOP {len(selected_pdfs)} scripts (--top {args.top})...")
    else:
        selected_pdfs = pdf_files
        print(f"\n[PDF Converter] Processing all {total_found} PDF scripts...")

    # Step 4: Render images
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_generated_images = []
    print("-" * 60)
    for idx, pdf_path in enumerate(selected_pdfs, 1):
        print(f"[{idx}/{len(selected_pdfs)}] Processing: {pdf_path.name}")
        images = convert_pdf_to_images(
            str(pdf_path),
            str(out_dir),
            dpi=args.dpi,
            first_page_only=args.first_page_only
        )
        for img_p in images:
            print(f"   -> Saved: {img_p}")
            all_generated_images.append(img_p)

    print("-" * 60)
    print(f" Successfully converted {len(selected_pdfs)} PDF scripts into {len(all_generated_images)} images!")
    print(f" Output directory: {args.output_dir}")
    print("\nNext step to run OCR on these converted images:")
    print(f"  python scripts/run_ocr.py --input {args.output_dir}/ --backend trocr")
    print(f"  python scripts/run_ocr.py --input {args.output_dir}/ --backend infinite_ocr\n")


if __name__ == "__main__":
    main()
