"""Stage 1 — PDF -> images.

PyMuPDF, 300 DPI, PNG, deterministic naming (page_01.png). Renders the full page
(not the embedded scan xref) so watermark + student ink + examiner red flatten
into one image. Skips conversion when outputs already exist unless forced.
Reports per script: page count, dimensions, file sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from src.config import Config


class ConversionError(RuntimeError):
    """Raised when a script fails to convert; carries the script id."""

    def __init__(self, script_id: str, message: str) -> None:
        self.script_id = script_id
        super().__init__(f"[{script_id}] {message}")


@dataclass
class PageResult:
    page_num: int
    path: Path
    width: int
    height: int
    size_bytes: int


@dataclass
class ScriptResult:
    script_id: str
    page_count: int
    pages: list[PageResult] = field(default_factory=list)
    skipped: bool = False


def script_id_from_path(pdf: Path) -> str:
    """Script id is the PDF filename stem, e.g. SE_11_Q1_0001."""
    return pdf.stem


def discover_scripts(cfg: Config) -> list[tuple[str, Path]]:
    """All PDFs in the raw_pdfs dir, sorted by filename, as (script_id, path)."""
    pdfs = sorted(cfg.paths.raw_pdfs.glob("*.pdf"))
    return [(script_id_from_path(p), p) for p in pdfs]


def _page_filename(page_num: int, image_format: str) -> str:
    """Deterministic, zero-padded name: page_01.png (1-indexed)."""
    return f"page_{page_num:02d}.{image_format}"


def _existing_pages(out_dir: Path, image_format: str) -> list[Path]:
    return sorted(out_dir.glob(f"page_*.{image_format}"))


def convert_script(
    script_id: str,
    pdf_path: Path,
    cfg: Config,
    *,
    force: bool = False,
) -> ScriptResult:
    """Render every page of one PDF to a PNG. Skips if already complete.

    Failures are wrapped in ConversionError with the script id attached so the
    caller can record the script as failed without silently losing the reason.
    """
    out_dir = cfg.paths.images / script_id
    fmt = cfg.pdf.image_format

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise ConversionError(script_id, f"cannot open PDF: {exc}") from exc

    try:
        page_count = doc.page_count

        # Resumable skip: complete output already present.
        if not force:
            existing = _existing_pages(out_dir, fmt)
            if len(existing) == page_count and page_count > 0:
                return ScriptResult(
                    script_id=script_id,
                    page_count=page_count,
                    pages=[
                        PageResult(
                            page_num=i + 1,
                            path=p,
                            width=0,
                            height=0,
                            size_bytes=p.stat().st_size,
                        )
                        for i, p in enumerate(existing)
                    ],
                    skipped=True,
                )

        out_dir.mkdir(parents=True, exist_ok=True)

        # On force (or a partial/mismatched dir) clear stale renders first.
        for stale in _existing_pages(out_dir, fmt):
            stale.unlink()

        pages: list[PageResult] = []
        for index in range(page_count):
            page_num = index + 1
            try:
                page = doc[index]
                # RGB, no alpha: keeps red/blue/black ink channels intact.
                pix = page.get_pixmap(dpi=cfg.pdf.dpi, alpha=False)
                out_path = out_dir / _page_filename(page_num, fmt)
                pix.save(out_path)
            except Exception as exc:  # noqa: BLE001
                raise ConversionError(
                    script_id, f"failed rendering page {page_num}: {exc}"
                ) from exc

            pages.append(
                PageResult(
                    page_num=page_num,
                    path=out_path,
                    width=pix.width,
                    height=pix.height,
                    size_bytes=out_path.stat().st_size,
                )
            )

        return ScriptResult(
            script_id=script_id,
            page_count=page_count,
            pages=pages,
            skipped=False,
        )
    finally:
        doc.close()
