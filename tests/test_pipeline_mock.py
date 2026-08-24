import os
import tempfile
from PIL import Image, ImageDraw
from src.core.config import PipelineConfig
from src.engine.mock_engine import MockGemmaEngine
from src.pipeline.orchestrator import ScriptCheckingPipeline


def test_full_pipeline_mock_image():
    # 1. Create a dummy image
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Sample Bangla Exam Text", fill=(0, 0, 0))

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
        assert os.path.exists(os.path.join(script_dir, "stage1_transcription.json"))
        assert os.path.exists(os.path.join(script_dir, "stage1_raw_transcript.txt"))
        assert os.path.exists(os.path.join(script_dir, "stage2_verification.json"))
        assert os.path.exists(os.path.join(script_dir, "stage2_verified_transcript.txt"))
        assert os.path.exists(os.path.join(script_dir, "stage3_errors.json"))
        assert os.path.exists(os.path.join(script_dir, "stage3_errors.csv"))
        assert os.path.exists(os.path.join(script_dir, "stage4_evaluation.json"))
        assert os.path.exists(os.path.join(script_dir, "complete_report.json"))
        assert os.path.exists(os.path.join(script_dir, "evaluation_report.md"))
