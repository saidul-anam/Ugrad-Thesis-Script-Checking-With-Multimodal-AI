"""
PDF loader and rasterizer module using pypdfium2 / PyMuPDF (fitz) or PIL for images.
"""

import os
from typing import List, Union
from PIL import Image
import io

# Disable PIL decompression bomb warning for large scans
Image.MAX_IMAGE_PIXELS = None


def load_images_from_file(file_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Load page images from a file (PDF or single/multi-page image).
    
    Args:
        file_path: Absolute or relative path to PDF or image file
        dpi: Resolution for PDF rasterization (default 300)
        
    Returns:
        List of PIL Image objects in RGB format
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return rasterize_pdf(file_path, dpi=dpi)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]:
        img = Image.open(file_path).convert("RGB")
        return [img]
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def rasterize_pdf(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Rasterize PDF into a list of PIL Images at specified DPI.
    Tries pypdfium2 first, then PyMuPDF (fitz).
    """
    images: List[Image.Image] = []

    # Method 1: pypdfium2
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        scale = dpi / 72.0  # 72 points per inch standard PDF resolution
        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil().convert("RGB")
            images.append(pil_image)
        return images
    except ImportError:
        pass
    except Exception as e:
        # Fallback to fitz if pypdfium2 fails
        pass

    # Method 2: PyMuPDF (fitz)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append(img)
        return images
    except ImportError:
        pass

    # Method 3: pdf2image
    try:
        from pdf2image import convert_from_path
        return convert_from_path(pdf_path, dpi=dpi)
    except ImportError:
        raise ImportError(
            "No PDF rasterizer available. Please install `pypdfium2` or `pymupdf` or `pdf2image`."
        )
