"""
Remove red pen marks from scanned handwritten Bangla PDFs (batch mode).

TWO-STEP APPROACH
-----------------
1. **Whitening**: blend every pixel toward a warm-white targ  et color, in
   proportion to how light it already is. Dark pixels (ink) are left alone;
   only light/background-ish pixels get pulled toward white. Gentle and
   uniform, so it doesn't harshly rewrite the background.
2. **Red Mark Removal**: detect red ink via HSV hue+saturation (not a plain
   RGB comparison) and fill those pixels with the target background color.

USAGE
-----
    python remove_red_marks.py                       # uses defaults below
    python remove_red_marks.py <input_dir> <output_dir>
"""

import os
import sys
import glob
import io
import numpy as np
from PIL import Image, ImageFilter
import pypdfium2 as pdfium
import img2pdf

# Large scans at 300 DPI legitimately exceed PIL's default pixel-count
# safety limit -- this isn't a decompression bomb, so raise the cap
# instead of letting PIL warn/throw on every page.
Image.MAX_IMAGE_PIXELS = None

# ----------------------------------------------------------------------
# TUNE THESE IF NEEDED
# ----------------------------------------------------------------------
DPI = 300               # render resolution (300 keeps handwriting sharp)
JPEG_QUALITY = 92        # output image quality inside the PDF

# Whitening (per-pixel blend toward a warm-white target)
BLACK_THRESH = 90        # pixels at/below this brightness = ink, untouched
WHITE_THRESH = 200       # pixels at/above this brightness = fully whitened
TARGET_WHITE = (250, 248, 240)   # soft warm-white the background is pulled toward

# Red-mark detection (HSV: hue + saturation, not plain RGB)
RED_HUE_DEGREES = 20     # how far from true red (0/360 deg) counts as marker
RED_SAT_MIN = 70         # min saturation (0-255) to count as marker ink
RED_VAL_MIN = 60         # min brightness (0-255), avoids pure shadow/black
# ----------------------------------------------------------------------


def whiten_background(img: Image.Image) -> Image.Image:
    """Blend each pixel toward TARGET_WHITE, weighted by how bright it
    already is. Pure PIL/numpy -- no OpenCV, no local background estimate.

    weight = 0   -> pixel <= BLACK_THRESH: left completely alone (this is
                    what protects handwriting from being touched at all)
    weight = 1   -> pixel >= WHITE_THRESH: fully replaced by TARGET_WHITE
    in between   -> smooth linear blend

    Using max(R,G,B) as the "how bright" measure (rather than a
    weighted-average luminance) means a reddish background pixel is
    judged by its brightest channel, not dragged down by the color cast
    itself -- so warm/reddish pages get whitened the same as pale ones.
    """
    arr = np.array(img.convert("RGB")).astype(np.float32)
    brightness = arr.max(axis=-1)

    weight = (brightness - BLACK_THRESH) / (WHITE_THRESH - BLACK_THRESH)
    weight = np.clip(weight, 0.0, 1.0)[..., None]

    target = np.array(TARGET_WHITE, dtype=np.float32)
    norm = arr * (1 - weight) + target * weight
    return Image.fromarray(np.clip(norm, 0, 255).astype(np.uint8), mode="RGB")


def remove_red_marks(img: Image.Image) -> Image.Image:
    """Detect red marker ink via HSV (hue + saturation) using PIL's
    built-in HSV conversion, then fill those pixels with TARGET_WHITE.
    Since whiten_background already pulled the real background close to
    that same color, the fill blends in rather than leaving a mismatched
    patch."""
    arr = np.array(img.convert("RGB"))
    hsv = np.array(img.convert("HSV"))
    h, s, v = hsv[..., 0].astype(np.int16), hsv[..., 1], hsv[..., 2]

    # PIL's H channel is 0-255 mapped to 0-360 degrees
    hue_max = int(RED_HUE_DEGREES / 360 * 255)
    red_mask = ((h <= hue_max) | (h >= 255 - hue_max)) & (s > RED_SAT_MIN) & (v > RED_VAL_MIN)

    if not red_mask.any():
        return img.convert("RGB")

    # Dilate the mask slightly so the fill fully covers ink edges
    mask_img = Image.fromarray((red_mask.astype(np.uint8) * 255))
    mask_img = mask_img.filter(ImageFilter.MaxFilter(3))
    dilated_mask = np.array(mask_img) > 0

    out = arr.copy()
    out[dilated_mask] = TARGET_WHITE
    return Image.fromarray(out, mode="RGB")


def remove_red(img: Image.Image) -> Image.Image:
    """Step 1: whiten the background gently (dark ink untouched).
    Step 2: remove red marker ink via HSV, fill with the target white."""
    whitened = whiten_background(img)
    return remove_red_marks(whitened)


def process_pdf(in_path: str, out_path: str, dpi: int = DPI) -> None:
    """Render every page, strip red marks, rebuild a clean PDF."""
    try:
        pdf = pdfium.PdfDocument(in_path)
        if len(pdf) == 0:
            print(f"WARNING: PDF has 0 pages")
            pdf.close()
            return

        cleaned_pages_bytes = []

        for page_index in range(len(pdf)):
            try:
                page = pdf[page_index]
                bitmap = page.render(scale=dpi / 72)
                pil_image = bitmap.to_pil()

                cleaned = remove_red(pil_image)

                buf = io.BytesIO()
                cleaned.save(buf, format="JPEG", quality=JPEG_QUALITY)
                cleaned_pages_bytes.append(buf.getvalue())

                page.close()
            except Exception as e:
                print(f"ERROR processing page {page_index}: {e}")
                continue

        pdf.close()

        if cleaned_pages_bytes:
            with open(out_path, "wb") as f:
                f.write(img2pdf.convert(cleaned_pages_bytes))
        else:
            print(f"WARNING: No pages processed")
    except Exception as e:
        print(f"ERROR: {e}")


def process_folder(input_folder: str, output_folder: str) -> None:
    os.makedirs(output_folder, exist_ok=True)
    pdf_files = sorted(glob.glob(os.path.join(input_folder, "*.pdf")))

    if not pdf_files:
        print(f"No PDF files found in: {input_folder}")
        return

    print(f"Found {len(pdf_files)} PDF file(s). Starting...\n")

    for i, in_path in enumerate(pdf_files, 1):
        filename = os.path.basename(in_path)
        out_path = os.path.join(output_folder, filename)
        print(f"[{i}/{len(pdf_files)}] Processing: {filename} ...", end=" ", flush=True)
        try:
            process_pdf(in_path, out_path)
            print("done")
        except Exception as e:
            print(f"FAILED ({e})")

    print(f"\nAll done. Cleaned PDFs saved to: {output_folder}")


if __name__ == "__main__":
    # Default folder names - change these, or pass as command-line args:
    #   python remove_red_marks.py my_pdfs cleaned_pdfs
    INPUT_FOLDER = "./datasets"
    OUTPUT_FOLDER = "cleaned_pdfs"

    if len(sys.argv) == 3:
        INPUT_FOLDER = sys.argv[1]
        OUTPUT_FOLDER = sys.argv[2]

    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
