from pathlib import Path

from src.utils.script_files import (
    paper_folder_name,
    resolve_paper_folder,
    resolve_extracted_output_dir,
    list_script_files,
    paper_folders,
    all_script_files,
    find_script_file,
    describe_paper_folders,
    move_into_paper_folders,
    move_extracted_into_paper_folders,
)


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _per_paper_root(tmp_path: Path) -> Path:
    root = tmp_path / "english"
    for n in ("0001", "0002"):
        _touch(root / "se_11_q1" / f"SE_11_Q1_{n}.pdf")
    _touch(root / "se_10_q1" / "SE_10_Q1_0001.pdf")
    return root


def test_paper_folder_name():
    assert paper_folder_name("SE_10_Q1_0005.pdf") == "se_10_q1"
    assert paper_folder_name("data/raw_pdfs/english/SE_11_Q1_0010.pdf") == "se_11_q1"
    assert paper_folder_name("SB_11_Q1_0002.pdf.pdf") == "sb_11_q1"
    assert paper_folder_name("notes.pdf") is None


def test_listing_is_not_recursive_and_counts_paper_folders(tmp_path):
    root = _per_paper_root(tmp_path)
    _touch(root / "se_10_q1" / "readme.txt")
    (root / "empty").mkdir()
    assert list_script_files(str(root)) == []
    assert paper_folders(str(root)) == {"se_10_q1": 1, "se_11_q1": 2}
    assert [Path(p).name for p in all_script_files(str(root))] == [
        "SE_10_Q1_0001.pdf", "SE_11_Q1_0001.pdf", "SE_11_Q1_0002.pdf"]


def test_flat_folder_still_works(tmp_path):
    flat = tmp_path / "bangla"
    _touch(flat / "SB_11_Q1_0002.pdf")
    _touch(flat / "page.PNG")
    assert [Path(p).name for p in list_script_files(str(flat))] == ["SB_11_Q1_0002.pdf", "page.PNG"]
    assert paper_folders(str(flat)) == {}
    assert describe_paper_folders(str(flat)) is None


def test_find_script_file_by_id_old_path_or_existing_path(tmp_path):
    root = _per_paper_root(tmp_path)
    expected = str(root / "se_11_q1" / "SE_11_Q1_0002.pdf")
    assert find_script_file("SE_11_Q1_0002", [str(root)]) == expected
    assert find_script_file(str(root / "SE_11_Q1_0002.pdf"), [str(root)]) == expected  # pre-reorganisation path
    assert find_script_file(expected, []) == expected
    assert find_script_file("SE_10_Q1_0099", [str(root)]) is None


def test_describe_paper_folders_names_folders_and_flag(tmp_path):
    root = _per_paper_root(tmp_path)
    hint = describe_paper_folders(str(root))
    assert "se_10_q1 (1 scripts)" in hint and "se_11_q1 (2 scripts)" in hint
    assert f"--pdf-dir {root / 'se_10_q1'}" in hint


def test_move_into_paper_folders_files_downloads_and_replaces(tmp_path):
    root = _per_paper_root(tmp_path)
    _touch(root / "SE_11_Q1_0002.pdf", "fresh")  # re-downloaded copy
    _touch(root / "SE_10_Q1_0002.pdf")
    _touch(root / "notes.pdf")  # no paper ID: left alone
    moved = move_into_paper_folders(str(root))
    assert sorted(Path(p).name for p in moved) == ["SE_10_Q1_0002.pdf", "SE_11_Q1_0002.pdf"]
    assert (root / "se_11_q1" / "SE_11_Q1_0002.pdf").read_text() == "fresh"
    assert [Path(p).name for p in list_script_files(str(root))] == ["notes.pdf"]


def test_resolve_paper_folder_across_levels():
    # Direct filenames with different levels: 11, 10, 09, 08
    assert resolve_paper_folder("data/raw_pdfs/english/se_11_q1/SE_11_Q1_0001.pdf") == "se_11_q1"
    assert resolve_paper_folder("data/raw_pdfs/english/se_10_q1/SE_10_Q1_0001.pdf") == "se_10_q1"
    assert resolve_paper_folder("data/raw_pdfs/english/se_09_q1/SE_09_Q1_0001.pdf") == "se_09_q1"
    assert resolve_paper_folder("data/raw_pdfs/english/se_08_q1/SE_08_Q1_0001.pdf") == "se_08_q1"
    assert resolve_paper_folder("data/raw_pdfs/bangla/sb_11_q1/SB_11_Q1_0001.pdf") == "sb_11_q1"
    # Stem / ID alone
    assert resolve_paper_folder("SE_11_Q1_0001") == "se_11_q1"
    assert resolve_paper_folder("SE_10_Q1_0005") == "se_10_q1"
    assert resolve_paper_folder("SE_09_Q1_0002") == "se_09_q1"
    assert resolve_paper_folder("SE_08_Q1_0003") == "se_08_q1"
    # Directory paths
    assert resolve_paper_folder("data/raw_pdfs/english/se_11_q1") == "se_11_q1"
    assert resolve_paper_folder("data/raw_pdfs/english/se_10_q1") == "se_10_q1"
    # Unmatched / none
    assert resolve_paper_folder("notes.pdf") is None
    assert resolve_paper_folder("") is None


def test_resolve_extracted_output_dir():
    # Default / None base_output_dir with various levels
    assert resolve_extracted_output_dir(None, "data/raw_pdfs/english/se_11_q1/SE_11_Q1_0001.pdf", "english") == "outputs/extracted/english/se_11_q1"
    assert resolve_extracted_output_dir("outputs/extracted/english", "data/raw_pdfs/english/se_10_q1/SE_10_Q1_0001.pdf", "english") == "outputs/extracted/english/se_10_q1"
    assert resolve_extracted_output_dir(None, "data/raw_pdfs/english/se_09_q1/SE_09_Q1_0001.pdf", "english") == "outputs/extracted/english/se_09_q1"
    assert resolve_extracted_output_dir(None, "data/raw_pdfs/english/se_08_q1/SE_08_Q1_0001.pdf", "english") == "outputs/extracted/english/se_08_q1"
    assert resolve_extracted_output_dir("outputs/extracted", "data/raw_pdfs/bangla/sb_11_q1/SB_11_Q1_0001.pdf", "bangla") == "outputs/extracted/bangla/sb_11_q1"
    # Already includes paper folder
    assert resolve_extracted_output_dir("outputs/extracted/english/se_11_q1", "SE_11_Q1_0001", "english") == "outputs/extracted/english/se_11_q1"
    # Custom base output directory preserved
    assert resolve_extracted_output_dir("custom/output/dir", "SE_11_Q1_0001", "english") == "custom/output/dir"


def test_move_extracted_into_paper_folders(tmp_path):
    ext_dir = tmp_path / "outputs" / "extracted" / "english"
    # Create loose script directories
    s1 = _touch(ext_dir / "SE_11_Q1_0001" / "extraction_result.json", "{}")
    s2 = _touch(ext_dir / "SE_10_Q1_0001" / "extraction_result.json", "{}")
    csv = _touch(ext_dir / "raw_tier_dataset.csv", "header\n")

    moved = move_extracted_into_paper_folders(str(ext_dir), lang="english", create_symlinks=True)
    assert len(moved) == 2
    # Verify moved targets exist
    assert (ext_dir / "se_11_q1" / "SE_11_Q1_0001" / "extraction_result.json").exists()
    assert (ext_dir / "se_10_q1" / "SE_10_Q1_0001" / "extraction_result.json").exists()
    # Verify dataset csv copied to paper folders
    assert (ext_dir / "se_11_q1" / "raw_tier_dataset.csv").exists()
    assert (ext_dir / "se_10_q1" / "raw_tier_dataset.csv").exists()
    # Verify symlinks created
    assert (ext_dir / "SE_11_Q1_0001").is_symlink()
    assert (ext_dir / "SE_10_Q1_0001").is_symlink()

