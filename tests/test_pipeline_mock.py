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
        eval_dir = os.path.join(tmpdir, "evaluated", "isolated_script")
        report = pipeline.evaluate_extracted_script(extraction_input=script_dir, output_dir=eval_dir)
        assert report.script_id == "isolated_script"
        assert report.stage4_evaluation.final_score > 0
        # Evaluation files saved in evaluation directory
        assert os.path.exists(os.path.join(eval_dir, "stage4_evaluation.json"))
        assert os.path.exists(os.path.join(eval_dir, "complete_report.json"))
        assert os.path.exists(os.path.join(eval_dir, "evaluation_report.md"))
        # Extraction directory has NO stage4 or report files (clean separation)
        assert not os.path.exists(os.path.join(script_dir, "stage4_evaluation.json"))
        assert not os.path.exists(os.path.join(script_dir, "complete_report.json"))
        assert not os.path.exists(os.path.join(script_dir, "evaluation_report.md"))


def test_pipeline_fast_mode_and_checkpointing():
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Fast Mode Test Script", fill=(0, 0, 0))

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "fast_script.png")
        img.save(img_path)

        cfg = PipelineConfig()
        cfg.pipeline.output_dir = os.path.join(tmpdir, "outputs")
        engine = MockGemmaEngine(model_id=cfg.model.model_id)
        pipeline = ScriptCheckingPipeline(engine=engine, config=cfg)

        # 1. Run in Fast Mode (skip_stage2=True)
        extraction = pipeline.extract_script(
            input_source=img_path,
            skip_stage2=True
        )

        assert extraction.stage2_verification.total_corrections_count == 0
        assert "Fast mode" in extraction.stage2_verification.verification_notes
        assert len(extraction.pages) == 1

        script_dir = os.path.join(cfg.pipeline.output_dir, "fast_script")
        ckpt_file = os.path.join(script_dir, "checkpoints", "page_1.json")
        assert os.path.exists(ckpt_file)

        # 2. Re-run without force_extract to verify checkpoint loading
        resumed_extraction = pipeline.extract_script(
            input_source=img_path,
            skip_stage2=True,
            force_extract=False
        )
        assert resumed_extraction.script_id == "fast_script"
        assert len(resumed_extraction.pages) == 1
        assert resumed_extraction.stage3_errors.total_error_count >= 0


def test_token_usage_persistence_in_reports_and_csv():
    import csv
    import json
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Token Usage Persistence Test", fill=(0, 0, 0))

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "tok_script.png")
        img.save(img_path)

        cfg = PipelineConfig()
        cfg.pipeline.output_dir = os.path.join(tmpdir, "outputs")
        engine = MockGemmaEngine(model_id=cfg.model.model_id)
        pipeline = ScriptCheckingPipeline(engine=engine, config=cfg)

        report = pipeline.evaluate_script(input_source=img_path)
        script_dir = os.path.join(cfg.pipeline.output_dir, report.script_id)

        # 1. Check extraction_result.json metadata
        ext_json_path = os.path.join(script_dir, "extraction_result.json")
        with open(ext_json_path, "r", encoding="utf-8") as f:
            ext_data = json.load(f)
        assert "token_usage" in ext_data["metadata"]
        ext_tok = ext_data["metadata"]["token_usage"]
        assert ext_tok["total_tokens"] > 0
        assert "pages" in ext_tok
        assert "page_1" in ext_tok["pages"]

        # 2. Check page checkpoint json
        ckpt_path = os.path.join(script_dir, "checkpoints", "page_1.json")
        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt_data = json.load(f)
        assert "token_usage" in ckpt_data
        assert ckpt_data["token_usage"]["total_tokens"] > 0

        # 3. Check extraction_summary.md
        summary_md_path = os.path.join(script_dir, "extraction_summary.md")
        with open(summary_md_path, "r", encoding="utf-8") as f:
            summary_md = f.read()
        assert "## Context & Token Usage Breakdown" in summary_md
        assert "Cumulative Extraction Tokens" in summary_md

        # 4. Check complete_report.json metadata
        comp_json_path = os.path.join(script_dir, "complete_report.json")
        with open(comp_json_path, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
        assert "stage4_token_usage" in comp_data["metadata"]
        assert comp_data["metadata"]["stage4_token_usage"]["total_tokens"] > 0
        assert "extraction_token_usage" in comp_data["metadata"]

        # 5. Check evaluation_report.md
        eval_md_path = os.path.join(script_dir, "evaluation_report.md")
        with open(eval_md_path, "r", encoding="utf-8") as f:
            eval_md = f.read()
        assert "Stage 4 Token Usage" in eval_md
        assert "## Pipeline Context & Token Usage Breakdown" in eval_md

        # 6. Check raw_tier_dataset.csv (13 columns, tokens in ocr_flags)
        csv_path = os.path.join(cfg.pipeline.output_dir, "raw_tier_dataset.csv")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        row = rows[0]
        assert len(row) == 13
        assert "tokens:" in row["ocr_flags"]
        assert "4096" in row["ocr_flags"]

