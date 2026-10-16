import pytest
from pathlib import Path
from scripts.setup_data_without_pipeline import (
    parse_script_pages,
    is_student_script_file,
    create_extraction_result_from_txt,
    setup_single_script,
    discover_input_script_files,
)
from src.utils.export_utils import load_extraction_artifacts
from scripts.evaluate_scripts import find_extracted_scripts


SAMPLE_SCRIPT_TEXT = """--- PAGE 1 ---
Answer to the question number - 01

(a) iv. the practice of keeping ourselves clean
(b) ii. next to godliness
(c) iii. our body

--- PAGE 2 ---
Answer to the question number 02
(a)
The word hygiene means the practice of keeping ourselves clean.
(b)
Hygiene is thought to be next to godliness.
"""


def test_parse_script_pages():
    pages = parse_script_pages(SAMPLE_SCRIPT_TEXT)
    assert len(pages) == 2
    assert pages[0][0] == 1
    assert "question number - 01" in pages[0][1]
    assert pages[1][0] == 2
    assert "question number 02" in pages[1][1]


def test_is_student_script_file():
    assert is_student_script_file(Path("SE_08_Q1_0001.txt"), paper_id="SE_08_Q1") is True
    assert is_student_script_file(Path("SE_08_Q1_0036.txt"), paper_id="SE_08_Q1") is True
    assert is_student_script_file(Path("SE_08_Q1.txt"), paper_id="SE_08_Q1") is False
    assert is_student_script_file(Path("SE_08_Q1.csv"), paper_id="SE_08_Q1") is False


def test_create_extraction_result_from_txt():
    ext_result, gt_marks, gt_source = create_extraction_result_from_txt(
        script_id="SE_08_Q1_0001",
        raw_text=SAMPLE_SCRIPT_TEXT,
        paper_id="SE_08_Q1",
        lang="english",
        sync_gt=True
    )
    assert ext_result.script_id == "SE_08_Q1_0001"
    assert len(ext_result.pages) == 2
    assert ext_result.stage1_transcription.word_count > 0
    assert "practice of keeping ourselves clean" in ext_result.stage2_verification.verified_transcript
    # Teacher marks should be discovered from SE_08_Q1_teacher_marks.csv
    assert gt_marks is not None
    assert "1" in gt_marks or "total_marks" in gt_marks
    assert ext_result.metadata.get("ground_truth_marks") == gt_marks


def test_setup_single_script_and_load(tmp_path):
    input_file = tmp_path / "SE_08_Q1_0001.txt"
    input_file.write_text(SAMPLE_SCRIPT_TEXT, encoding="utf-8")

    out_root = tmp_path / "data_without_pipeline"
    res = setup_single_script(
        file_path=input_file,
        output_root=out_root,
        paper_id="SE_08_Q1",
        lang="english",
        sync_gt=True
    )

    assert res["script_id"] == "SE_08_Q1_0001"
    assert res["pages"] == 2
    assert Path(res["target_dir"]).exists()

    # Verify artifacts exist
    target_p = Path(res["target_dir"])
    assert (target_p / "extraction_result.json").exists()
    assert (target_p / "stage1_transcription.json").exists()
    assert (target_p / "stage2_verification.json").exists()
    assert (target_p / "stage3_errors.json").exists()
    assert (target_p / "source_transcript.txt").exists()

    # Verify load_extraction_artifacts can load it directly
    loaded = load_extraction_artifacts(target_p)
    assert loaded.script_id == "SE_08_Q1_0001"
    assert len(loaded.pages) == 2

    # Verify find_extracted_scripts discovers it recursively
    discovered = find_extracted_scripts(str(out_root))
    assert any(d.name == "SE_08_Q1_0001" for d in discovered)
