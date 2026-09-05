import pytest
from pathlib import Path
from src.core.schemas import ExtractedQuestion
from src.utils.question_utils import (
    extract_question_id,
    save_extracted_question,
    load_question_for_script,
    find_question_artifact
)


def test_extract_question_id():
    # English prefix
    assert extract_question_id("SE_11_Q1_0001.pdf") == "SE_11_Q1"
    assert extract_question_id("SE_11_Q1_0024.pdf") == "SE_11_Q1"
    assert extract_question_id("SE_11_Q1.pdf") == "SE_11_Q1"
    assert extract_question_id("SE_11_Q2.pdf") == "SE_11_Q2"

    # Bangla prefix
    assert extract_question_id("SB_11_Q1_0002.pdf") == "SB_11_Q1"
    assert extract_question_id("SB_11_Q1.pdf") == "SB_11_Q1"
    assert extract_question_id("SB_07_Q1.pdf") == "SB_07_Q1"

    # Fallback with lang
    assert extract_question_id("11_Q1_0001.pdf", lang="english") == "SE_11_Q1"
    assert extract_question_id("11_Q1_0001.pdf", lang="bangla") == "SB_11_Q1"


def test_question_save_and_load(tmp_path):
    q = ExtractedQuestion(
        question_id="SE_11_Q1",
        language="english",
        title="AI in Education Essay",
        question_text="Write an essay on the impact of AI on education.",
        total_marks=10.0,
        source_file="data/questions/english/SE_11_Q1.pdf"
    )

    saved_path = save_extracted_question(q, output_dir=str(tmp_path))
    assert Path(saved_path).exists()
    assert (tmp_path / "english" / "SE_11_Q1.json").exists()
    assert (tmp_path / "english" / "SE_11_Q1.md").exists()

    # Match by script ID
    loaded = load_question_for_script("SE_11_Q1_0001.pdf", lang="english", questions_root=str(tmp_path))
    assert loaded is not None
    assert loaded.question_id == "SE_11_Q1"
    assert loaded.total_marks == 10.0
    assert "impact of AI" in loaded.question_text


def test_bangla_question_save_and_load(tmp_path):
    q = ExtractedQuestion(
        question_id="SB_11_Q1",
        language="bangla",
        title="বাংলা সৃজনশীল প্রশ্ন",
        question_text="উদ্দীপকটি পড়িয়া নিচের প্রশ্নগুলোর উত্তর দাও...",
        total_marks=10.0,
        source_file="data/questions/bangla/SB_11_Q1.pdf"
    )

    saved_path = save_extracted_question(q, output_dir=str(tmp_path))
    assert Path(saved_path).exists()
    assert (tmp_path / "bangla" / "SB_11_Q1.json").exists()
    assert (tmp_path / "bangla" / "SB_11_Q1.md").exists()

    # Match by Bangla script ID
    loaded = load_question_for_script("SB_11_Q1_0002.pdf", lang="bangla", questions_root=str(tmp_path))
    assert loaded is not None
    assert loaded.question_id == "SB_11_Q1"
    assert loaded.language == "bangla"
    assert loaded.total_marks == 10.0

