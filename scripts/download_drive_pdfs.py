#!/usr/bin/env python3
"""
Google Drive PDF Downloader with Smart Caching:
Downloads exam script PDFs from Google Drive folder and skips already downloaded files.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional


GDRIVE_FOLDERS = {
    "bangla": "https://drive.google.com/drive/folders/1tUwj_h5fCMhzymloXhUEJqTeTFKD-ogS",
    "english": "https://drive.google.com/drive/folders/11spWhJTncBfM_qsOvpH17AgduhyQpqSN"
}
DEFAULT_GDRIVE_FOLDER_BANGLA = GDRIVE_FOLDERS["bangla"]
DEFAULT_GDRIVE_FOLDER_ENGLISH = GDRIVE_FOLDERS["english"]
DEFAULT_GDRIVE_FOLDER = DEFAULT_GDRIVE_FOLDER_BANGLA


def download_drive_pdfs(
    gdrive_url: str = DEFAULT_GDRIVE_FOLDER,
    target_dir: str = "data/raw_pdfs/bangla",
    top_limit: Optional[int] = None,
    skip_existing: bool = True
) -> List[str]:
    """
    Downloads exam script PDFs from Google Drive folder using gdown.
    Skips downloading if files already exist locally.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Check already existing PDFs
    existing_pdfs = sorted(list(target_path.glob("*.pdf")) + list(target_path.glob("*.PDF")))
    
    if skip_existing and existing_pdfs:
        print(f"[GDrive Downloader] Found {len(existing_pdfs)} existing PDFs in '{target_dir}'.")
        if top_limit and len(existing_pdfs) >= top_limit:
            print(f"[GDrive Downloader] Already have {len(existing_pdfs)} PDFs (requested top {top_limit}). Skipping download.")
            return [str(p) for p in existing_pdfs[:top_limit]]

    print(f"\n[GDrive Downloader] Downloading folder from: {gdrive_url}")
    print(f"[GDrive Downloader] Target directory: {target_dir}")

    try:
        import gdown
        gdown.download_folder(
            url=gdrive_url,
            output=str(target_path),
            quiet=False,
            use_cookies=False,
            remaining_ok=True
        )
    except ImportError:
        print("\n[WARN] 'gdown' package is not installed. Install with: pip install gdown")
        print(f"Alternatively, manually place your PDF files into '{target_dir}/'.")
    except Exception as e:
        print(f"\n[GDrive Downloader] Download note: {e}")
        print(f"If network blocks direct folder download, you can place PDFs manually into '{target_dir}/'.")

    # Refresh list of PDFs
    all_pdfs = sorted(list(target_path.glob("*.pdf")) + list(target_path.glob("*.PDF")))
    if top_limit and top_limit > 0:
        return [str(p) for p in all_pdfs[:top_limit]]
    return [str(p) for p in all_pdfs]


def main():
    parser = argparse.ArgumentParser(description="Download Exam Script PDFs from Google Drive")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_GDRIVE_FOLDER,
        help="Google Drive folder URL"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data/raw_pdfs",
        help="Local directory to store PDFs"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Limit number of PDFs to retain / process"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if PDFs already exist"
    )

    args = parser.parse_args()
    pdfs = download_drive_pdfs(
        gdrive_url=args.url,
        target_dir=args.target_dir,
        top_limit=args.top,
        skip_existing=not args.force_download
    )
    print(f"\nTotal PDF scripts available: {len(pdfs)}")
    for p in pdfs:
        print(f" - {p}")


if __name__ == "__main__":
    main()
