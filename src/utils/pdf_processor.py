import os
from pathlib import Path
from typing import List, Union, Tuple, Optional
from PIL import Image


def is_pdf(filepath: str) -> bool:
    """Check if file has a PDF extension."""
    return filepath.lower().endswith(".pdf")


def extract_images_from_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
    dpi: int = 200,
    first_page_only: bool = False
) -> List[Tuple[int, Image.Image, str]]:
    """
    Renders PDF pages to high-resolution PIL Images.
    
    Returns a list of tuples: (page_number_1_indexed, PIL_Image, saved_image_path)
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF processing. Install via: pip install pymupdf"
        )

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    script_id = pdf_file.stem
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = []
    doc = fitz.open(str(pdf_file))
    num_pages = len(doc)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    pages_to_process = [0] if first_page_only else range(num_pages)

    for page_idx in pages_to_process:
        if page_idx >= num_pages:
            break
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert pixmap to PIL Image
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if num_pages == 1 or first_page_only:
            out_filename = f"{script_id}.png"
        else:
            out_filename = f"{script_id}_page_{page_idx + 1:02d}.png"

        saved_path = os.path.join(output_dir, out_filename) if output_dir else ""
        if output_dir:
            pix.save(saved_path)

        results.append((page_idx + 1, pil_img, saved_path))

    doc.close()
    return results
