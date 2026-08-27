import os
import tempfile
import numpy as np
from PIL import Image, ImageDraw
from src.core.config import PipelineConfig
from src.engine.mock_engine import MockGemmaEngine
from src.pipeline.orchestrator import ScriptCheckingPipeline
from src.pipeline.stage0_red_ink_detector import RedInkDetector
from src.pipeline.stage0b_teacher_marks import extract_marks, Stage0bTeacherMarkExtractor


def test_stage0_red_ink_detector():
    detector = RedInkDetector(min_pixel_threshold=50)

    # 1. Negative image: Pure white with black text
    black_img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw_black = ImageDraw.Draw(black_img)
    draw_black.text((10, 10), "Regular Student Handwriting", fill=(0, 0, 0))

    res_neg = detector.detect(black_img)
    assert not res_neg.has_red_ink
    assert res_neg.red_pixel_count < 50

    # 2. Positive image: Red ink markings
    red_img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw_red = ImageDraw.Draw(red_img)
    draw_red.rectangle([50, 50, 80, 80], fill=(220, 20, 20))  # 900 red pixels

    res_pos = detector.detect(red_img)
    assert res_pos.has_red_ink
    assert res_pos.red_pixel_count >= 50


def test_stage0b_teacher_mark_extractor():
    # Valid output string
    valid_output = """
    [
      {"question_no": "1", "mark_value": "7/10", "location": "margin next to answer 1"},
      {"question_no": "2", "mark_value": "4", "location": "bottom right"}
    ]
    """
    marks = extract_marks(valid_output)
    assert marks is not None
    assert len(marks) == 2
    assert marks[0]["mark_value"] == "7/10"

    # Empty array
    empty_marks = extract_marks("[]")
    assert empty_marks == []

    # Malformed output -> returns None for manual review routing
    bad_marks = extract_marks("Not a json array")
    assert bad_marks is None


def test_full_pipeline_mock_image():
    # 1. Create a dummy image with red marks
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Sample Bangla Exam Text", fill=(0, 0, 0))
    draw.rectangle([200, 200, 240, 240], fill=(220, 20, 20))  # Red teacher mark area

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "sample_script.png")
        img.save(img_path)

        cfg = PipelineConfig()
        cfg.pipeline.output_dir = os.path.join(tmpdir, "outputs")
        cfg.model.model_id = "google/gemma-4-31b-it"

        engine = MockGemmaEngine(model_id=cfg.model.model_id)
        pipeline = ScriptCheckingPipeline(
            engine=engine,
            config=cfg
        )

        report = pipeline.evaluate_script(input_source=img_path)

        # Assert Stage 0 & 0b
        assert report.has_red_ink
        assert len(report.teacher_marks) > 0

        # Assert Stage 1
        assert len(report.stage1_transcription.raw_transcript) > 0
        assert report.stage1_transcription.word_count > 0

        # Assert Stage 2
        assert len(report.stage2_verification.verified_transcript) > 0
        assert report.stage2_verification.total_corrections_count >= 0

        # Assert Stage 3
        assert report.stage3_errors.total_error_count >= 0

        # Assert Stage 4
        assert report.stage4_evaluation.final_score > 0
        assert report.stage4_evaluation.total_max_marks == 10.0

        # Check stage-by-stage files exist
        script_dir = os.path.join(cfg.pipeline.output_dir, report.script_id)
        assert os.path.exists(os.path.join(script_dir, "stage0b_teacher_marks.json"))
        assert os.path.exists(os.path.join(script_dir, "stage1_transcription.json"))
        assert os.path.exists(os.path.join(script_dir, "stage1_raw_transcript.txt"))
        assert os.path.exists(os.path.join(script_dir, "stage2_verification.json"))
        assert os.path.exists(os.path.join(script_dir, "stage2_verified_transcript.txt"))
        assert os.path.exists(os.path.join(script_dir, "stage3_errors.json"))
        assert os.path.exists(os.path.join(script_dir, "stage3_errors.csv"))
        assert os.path.exists(os.path.join(script_dir, "stage4_evaluation.json"))
        assert os.path.exists(os.path.join(script_dir, "complete_report.json"))
        assert os.path.exists(os.path.join(script_dir, "evaluation_report.md"))
        assert os.path.exists(os.path.join(script_dir, "raw_tier_records.csv"))
        assert os.path.exists(os.path.join(cfg.pipeline.output_dir, "raw_tier_dataset.csv"))


def test_separate_extraction_and_evaluation():
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Test Script for Separate Extraction and Evaluation", fill=(0, 0, 0))

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "isolated_script.png")
        img.save(img_path)

        cfg = PipelineConfig()
        cfg.pipeline.output_dir = os.path.join(tmpdir, "extracted")
        cfg.model.model_id = "google/gemma-4-31b-it"

        engine = MockGemmaEngine(model_id=cfg.model.model_id)
        pipeline = ScriptCheckingPipeline(
            engine=engine,
            config=cfg
        )

        # 1. Step 1: Standalone Extraction (Stages 0, 0b, 1-3)
        extraction = pipeline.extract_script(input_source=img_path)
        assert extraction.script_id == "isolated_script"
        assert len(extraction.stage1_transcription.raw_transcript) > 0
        assert len(extraction.stage2_verification.verified_transcript) > 0
        assert extraction.stage3_errors.total_error_count >= 0

        script_dir = os.path.join(cfg.pipeline.output_dir, "isolated_script")
        assert os.path.exists(os.path.join(script_dir, "extraction_result.json"))
        assert os.path.exists(os.path.join(script_dir, "extraction_summary.md"))
        assert os.path.exists(os.path.join(script_dir, "stage3_errors.csv"))
        assert os.path.exists(os.path.join(script_dir, "raw_tier_records.csv"))
        assert os.path.exists(os.path.join(cfg.pipeline.output_dir, "raw_tier_dataset.csv"))

        # 2. Step 2: Standalone Evaluation (Stage 4) loading from extracted directory
        report = pipeline.evaluate_extracted_script(extraction_input=script_dir)
        assert report.script_id == "isolated_script"
        assert report.stage4_evaluation.final_score > 0
        assert os.path.exists(os.path.join(script_dir, "stage4_evaluation.json"))
        assert os.path.exists(os.path.join(script_dir, "complete_report.json"))
        assert os.path.exists(os.path.join(script_dir, "evaluation_report.md"))
