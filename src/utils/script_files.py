"""
Locating raw exam-script files on disk.

Scripts are stored one folder per question paper, the folder named after the paper ID in
lower case:

    data/raw_pdfs/english/se_11_q1/SE_11_Q1_0001.pdf
    data/raw_pdfs/english/se_10_q1/SE_10_Q1_0001.pdf

A flat folder (files directly under data/raw_pdfs/<lang>/) keeps working. Script IDs are unique
across papers, so a script can always be found from its ID alone.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from src.utils.question_utils import extract_question_id

SCRIPT_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".bmp")


def paper_folder_name(script_name: str, lang: Optional[str] = None) -> Optional[str]:
    """'SE_10_Q1_0005.pdf' -> 'se_10_q1'; None when the name carries no question-paper ID."""
    stem = Path(script_name).stem
    paper = extract_question_id(stem, lang=lang)
    return paper.lower() if paper and paper != stem else None


def resolve_paper_folder(script_path_or_id: str, lang: Optional[str] = None) -> Optional[str]:
    """
    Resolve the canonical paper folder name (e.g. 'se_11_q1', 'se_10_q1', 'se_09_q1', 'se_08_q1')
    from a script filename, full path, directory path, or script identifier.
    """
    if not script_path_or_id:
        return None

    # 1. Direct check on name / ID
    pname = paper_folder_name(script_path_or_id, lang=lang)
    if pname:
        return pname

    p = Path(script_path_or_id)
    # 2. Check path components (filename, immediate parent, grandparent)
    for part in [p.name, p.parent.name, p.parent.parent.name]:
        part_clean = part.lower().strip()
        if not part_clean or part_clean in [
            ".", "raw_pdfs", "english", "bangla", "data", "samples", "outputs", "extracted", "runs", "evaluated"
        ]:
            continue
        part_paper = paper_folder_name(part_clean, lang=lang)
        if part_paper:
            return part_paper
        # If the directory itself is named after the paper, e.g. 'se_11_q1', '11_q1', 'sb_10_q1'
        if re.search(r'(?:(?:se|sb)_)?([0-9]{1,4}_q[0-9]{1,3})', part_clean, re.IGNORECASE):
            qid = extract_question_id(part_clean, lang=lang)
            if qid:
                return qid.lower()

    return None


def resolve_extracted_output_dir(
    base_output_dir: Optional[str],
    script_path_or_id: str,
    lang: str = "english",
    extracted_root: str = "outputs/extracted"
) -> str:
    """
    Determine the proper extracted output directory for a script, incorporating the
    paper/level subfolder (e.g. 'outputs/extracted/english/se_11_q1').
    """
    paper = resolve_paper_folder(script_path_or_id, lang=lang)
    default_lang_dir = os.path.join(extracted_root, lang)

    if not base_output_dir or base_output_dir.strip() in [
        "",
        extracted_root,
        f"{extracted_root}/",
        default_lang_dir,
        f"{default_lang_dir}/",
        "outputs/extracted",
        "outputs/extracted/bangla",
        "outputs/extracted/english",
    ]:
        if paper:
            return os.path.join(extracted_root, lang, paper)
        return default_lang_dir

    norm_base = base_output_dir.rstrip("/")
    if paper and not norm_base.lower().endswith(paper.lower()):
        if norm_base.endswith(f"/{lang}") or norm_base.endswith("/extracted"):
            return os.path.join(norm_base, paper)

    return base_output_dir


def move_extracted_into_paper_folders(
    extracted_lang_dir: str,
    lang: str = "english",
    create_symlinks: bool = True
) -> List[str]:
    """
    Move loose script artifact directories in `extracted_lang_dir` (e.g. outputs/extracted/english/SE_11_Q1_0001)
    into their paper subfolder (e.g. outputs/extracted/english/se_11_q1/SE_11_Q1_0001).
    Optionally keeps relative symlinks at the old locations for backwards compatibility.
    """
    root = Path(extracted_lang_dir)
    if not root.is_dir():
        return []

    moved = []
    for item in sorted(root.iterdir()):
        if not item.is_dir() or item.is_symlink():
            continue
        paper = resolve_paper_folder(item.name, lang=lang)
        if not paper or item.name.lower() == paper.lower():
            continue
        # Verify it looks like an extracted script directory
        is_extracted_dir = (
            (item / "extraction_result.json").exists()
            or (item / "stage1_transcription.json").exists()
            or (item / "checkpoints").is_dir()
        )
        if not is_extracted_dir:
            continue

        dest_paper_dir = root / paper
        dest_paper_dir.mkdir(parents=True, exist_ok=True)
        dest_script_dir = dest_paper_dir / item.name

        if not dest_script_dir.exists():
            os.replace(str(item), str(dest_script_dir))
            moved.append(str(dest_script_dir))
            if create_symlinks:
                try:
                    rel_target = os.path.join(paper, item.name)
                    os.symlink(rel_target, str(item))
                except Exception:
                    pass

    # Also handle dataset csv if loose in root
    loose_csv = root / "raw_tier_dataset.csv"
    if loose_csv.is_file() and not loose_csv.is_symlink():
        for pdir in sorted(root.iterdir()):
            if pdir.is_dir() and not pdir.is_symlink() and resolve_paper_folder(pdir.name, lang=lang) == pdir.name:
                p_csv = pdir / "raw_tier_dataset.csv"
                if not p_csv.exists():
                    shutil.copy2(str(loose_csv), str(p_csv))

    return moved


def list_script_files(directory: str, extensions: Sequence[str] = SCRIPT_EXTENSIONS) -> List[str]:
    """Script files directly inside `directory` (not its sub-folders), sorted."""
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(str(p) for p in root.iterdir() if p.is_file() and p.suffix.lower() in extensions)


def paper_folders(directory: str, extensions: Sequence[str] = SCRIPT_EXTENSIONS) -> Dict[str, int]:
    """Sub-folders of `directory` that hold script files -> number of script files in each."""
    root = Path(directory)
    if not root.is_dir():
        return {}
    counts = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        n = len(list_script_files(str(sub), extensions))
        if n:
            counts[sub.name] = n
    return counts


def all_script_files(directory: str, extensions: Sequence[str] = SCRIPT_EXTENSIONS) -> List[str]:
    """Script files directly inside `directory` plus those in its paper sub-folders."""
    files = list_script_files(directory, extensions)
    for name in paper_folders(directory, extensions):
        files += list_script_files(os.path.join(directory, name), extensions)
    return files


def find_script_file(
    target: str,
    search_dirs: Iterable[str],
    extensions: Sequence[str] = SCRIPT_EXTENSIONS,
) -> Optional[str]:
    """
    Resolve `target` to an existing script file. `target` may be an existing path, a script ID
    ('SE_10_Q1_0005'), or a path from before the per-paper layout
    ('data/raw_pdfs/english/SE_11_Q1_0010.pdf'); the latter two are looked up by file stem in each
    search dir and its paper sub-folders.
    """
    if os.path.isfile(target):
        return target
    name = Path(target).name
    stem = (Path(name).stem if Path(name).suffix.lower() in extensions else name).lower()
    for d in search_dirs:
        for f in all_script_files(d, extensions):
            if Path(f).stem.lower() == stem:
                return f
    return None


def describe_paper_folders(directory: str, extensions: Sequence[str] = SCRIPT_EXTENSIONS) -> Optional[str]:
    """If `directory` is organised into paper sub-folders, a hint naming them and how to pick one."""
    folders = paper_folders(directory, extensions)
    if not folders:
        return None
    listing = ", ".join(f"{name} ({n} scripts)" for name, n in folders.items())
    example = os.path.join(directory, next(iter(folders)))
    return f"'{directory}' is organised by question paper: {listing}. Choose one with --pdf-dir {example}"


def move_into_paper_folders(directory: str, extensions: Sequence[str] = (".pdf",)) -> List[str]:
    """
    Move loose files in `directory` whose name carries a paper ID into `<directory>/<paper>/`.
    A file of the same name already in the paper folder is replaced (the loose copy is the fresh
    download). Returns the new paths.
    """
    moved = []
    for f in list_script_files(directory, extensions):
        paper = paper_folder_name(f)
        if not paper:
            continue
        dest_dir = os.path.join(directory, paper)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(f))
        os.replace(f, dest)
        moved.append(dest)
    return moved
