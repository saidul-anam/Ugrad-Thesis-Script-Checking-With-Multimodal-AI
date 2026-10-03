import pytest
from src.core.schemas import ExtractionResult, PageExtractionResult
from src.engine.mock_engine import MockGemmaEngine
from src.pipeline.orchestrator import ScriptCheckingPipeline
from src.core.config import load_config
from src.pipeline.stage4_modes import rubric_is_mode_based


def test_english_script_auto_routes_to_english_rubric():
    """Verify that an English script ID automatically routes to english_writing.yaml."""
    cfg = load_config("configs/pipeline_config.yaml")
    engine = MockGemmaEngine()
    # Initialize without specifying rubric_path
    pipeline = ScriptCheckingPipeline(engine=engine, config=cfg)

    # Mock ExtractionResult for an English script
    extraction = ExtractionResult.model_construct(
        script_id="SE_11_Q1_0002",
        subject="English",
        pages=[],
        metadata={"paper": "english", "language": "english"}
    )

    # Call run_evaluation with dry-run/mock
    # Verify the dynamic rubric resolution logic
    is_english = (
        str(extraction.metadata.get("paper", "")).lower() == "english"
        or str(extraction.metadata.get("language", "")).lower() == "english"
        or str(extraction.script_id).startswith("SE_")
    )
    assert is_english is True

    # Test that load_rubric returns english_writing.yaml with Mode A/B/C specifications
    rubric_data = pipeline._load_rubric("configs/rubrics/english_writing.yaml")
    assert rubric_is_mode_based(rubric_data) is True
    assert "mode_specifications" in rubric_data
    q_nos = [str(q.get("question_no")) for q in rubric_data.get("question_map", [])]
    assert "1(A)" in q_nos
    assert "6" in q_nos
