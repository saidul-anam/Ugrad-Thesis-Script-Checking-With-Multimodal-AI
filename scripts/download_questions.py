#!/usr/bin/env python3
"""
Question Paper Downloader with Smart Local Caching.

Downloads exam question PDFs from Google Drive (reading links from .env or CLI)
into data/questions/<lang>/, skipping existing local files.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_default_question_url(lang: str) -> Optional[str]:
    """Retrieve Google Drive URL for question papers from environment or defaults."""
    env_var = f"GDRIVE_{lang.upper()}_QUESTION"
    url = os.getenv(env_var, "").strip()
    return url if url else None


def download_question_pdfs(
    gdrive_url: Optional[str] = None,
    target_dir: str = "data/questions/english",
    top_limit: Optional[int] = None,
    skip_existing: bool = True,
    skip_download: bool = False
) -> List[str]:
    """
    Downloads question PDFs from Google Drive folder or file link using gdown.
    Skips downloading if files already exist locally or if skip_download is True.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    existing_pdfs = sorted(
        list(target_path.glob("*.pdf")) +
        list(target_path.glob("*.PDF")) +
        list(target_path.glob("*.png")) +
        list(target_path.glob("*.jpg"))
    )

    if skip_download:
        print(f"[Question Downloader] Local-only mode active. Found {len(existing_pdfs)} question file(s) in '{target_dir}'.")
        if top_limit and top_limit > 0:
            return [str(p) for p in existing_pdfs[:top_limit]]
        return [str(p) for p in existing_pdfs]

    if skip_existing and existing_pdfs:
        print(f"[Question Downloader] Found {len(existing_pdfs)} existing question file(s) in '{target_dir}'.")
        if top_limit and len(existing_pdfs) >= top_limit:
            print(f"[Question Downloader] Already have {len(existing_pdfs)} files (requested top {top_limit}). Skipping download.")
            return [str(p) for p in existing_pdfs[:top_limit]]

    if not gdrive_url:
        print(f"\n[Question Downloader] No Google Drive URL configured for '{target_dir}'.")
        print(f"You can place question PDFs directly into '{target_dir}/' (e.g. 'SE_11_Q1.pdf', 'SB_11_Q1.pdf').")
        print(f"Or configure GDRIVE_ENGLISH_QUESTION / GDRIVE_BANGLA_QUESTION in your .env file.")
        return [str(p) for p in existing_pdfs]

    print(f"\n[Question Downloader] Downloading questions from: {gdrive_url}")
    print(f"[Question Downloader] Target directory: {target_dir}")

    try:
        import gdown
        if "folders" in gdrive_url:
            gdown.download_folder(
                url=gdrive_url,
                output=str(target_path),
                quiet=False,
                use_cookies=False
            )
        else:
            # Single file download
            gdown.download(
                url=gdrive_url,
                output=str(target_path / "question.pdf"),
                quiet=False,
                fuzzy=True
            )
    except ImportError:
        print("\n[WARN] 'gdown' package is not installed. Install with: pip install gdown")
        print(f"Alternatively, manually place your question PDFs into '{target_dir}/'.")
    except Exception as e:
        print(f"\n[Question Downloader] Download note: {e}")
        print(f"You can place question PDFs manually into '{target_dir}/'.")

    # Refresh list of questions
    all_pdfs = sorted(
        list(target_path.glob("*.pdf")) +
        list(target_path.glob("*.PDF")) +
        list(target_path.glob("*.png")) +
        list(target_path.glob("*.jpg"))
    )
    if top_limit and top_limit > 0:
        return [str(p) for p in all_pdfs[:top_limit]]
    return [str(p) for p in all_pdfs]


def main():
    parser = argparse.ArgumentParser(description="Download Exam Question PDFs from Google Drive")
    parser.add_argument(
        "--lang",
        type=str,
        choices=["bangla", "english"],
        default="english",
        help="Language / Subject question dataset to download ('english' or 'bangla')"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Google Drive folder or file URL (overrides .env)"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Local directory to store question PDFs (defaults to data/questions/<lang>)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Limit number of question PDFs to retain / process"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if question PDFs already exist locally"
    )
    parser.add_argument(
        "--skip-download",
        "--local-only",
        action="store_true",
        help="List existing local question PDFs without contacting Google Drive"
    )

    args = parser.parse_args()
    target_dir = args.target_dir or f"data/questions/{args.lang}"
    gdrive_url = args.url or get_default_question_url(args.lang)

    pdfs = download_question_pdfs(
        gdrive_url=gdrive_url,
        target_dir=target_dir,
        top_limit=args.top,
        skip_existing=not args.force_download,
        skip_download=args.skip_download
    )
    print(f"\nTotal question files available in '{target_dir}': {len(pdfs)}")
    for idx, p in enumerate(pdfs, 1):
        print(f"  [{idx}] {p}")


if __name__ == "__main__":
    main()
