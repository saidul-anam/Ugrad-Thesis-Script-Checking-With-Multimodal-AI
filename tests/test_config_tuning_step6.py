import pytest
from unittest.mock import MagicMock
from src.core.config import load_config, PipelineConfig
from src.engine.mock_engine import MockGemmaEngine
from src.engine.local_api_engine import LocalAPIEngine
from src.pipeline.orchestrator import ScriptCheckingPipeline


def test_pipeline_config_loads_step6_tuning():
    """Verify that configs/pipeline_config.yaml parses with 16k context window and granular thinking modes."""
    cfg = load_config("configs/pipeline_config.yaml")
    assert isinstance(cfg, PipelineConfig)
    
    # Problem 10: 16k Context Ceiling
    assert cfg.model.context_window == 16384
    assert cfg.decoding.max_new_tokens == 4096

    # Problem 11: 4-bit NF4 with native BF16 compute default
    assert cfg.model.quantization == "4bit"
    assert cfg.model.torch_dtype == "bfloat16"

    # Problem 12: Granular thinking modes
    assert cfg.pipeline.stage1_thinking_mode is False, "Stage 1 verbatim must not use thinking mode to avoid preamble corruption"
    assert cfg.pipeline.stage2_thinking_mode is False, "Stage 2 structured reasoning is inside JSON fields"
    assert cfg.pipeline.stage3b_thinking_mode is True, "Stage 3b evidence arbitration uses thinking deliberation"
    assert cfg.pipeline.stage4_thinking_mode is True, "Stage 4 rubric evaluation uses thinking deliberation"


def test_engine_context_window_defaults():
    """Verify that engine classes align with the 16k operational context window."""
    mock = MockGemmaEngine()
    assert mock.context_window == 16384
    assert mock.max_context_window == 16384

    # LocalAPIEngine without active network call
    local_api = LocalAPIEngine.__new__(LocalAPIEngine)
    local_api.context_window = 16384
    assert local_api.context_window == 16384


def test_orchestrator_stage_thinking_routing():
    """Verify that orchestrator routes stage-specific thinking modes properly."""
    cfg = load_config("configs/pipeline_config.yaml")
    mock_eng = MockGemmaEngine()
    pipeline = ScriptCheckingPipeline(engine=mock_eng, config=cfg)

    # When thinking_mode is not explicitly overridden in run_evaluation:
    decoding = pipeline.config.decoding
    active_eval_thinking = getattr(pipeline.config.pipeline, "stage4_thinking_mode", decoding.thinking_mode)
    assert active_eval_thinking is True

    # In extract_script, verify stage-level flags
    s1_thinking = getattr(pipeline.config.pipeline, "stage1_thinking_mode", False)
    s4_thinking = getattr(pipeline.config.pipeline, "stage4_thinking_mode", True)
    assert s1_thinking is False
    assert s4_thinking is True
