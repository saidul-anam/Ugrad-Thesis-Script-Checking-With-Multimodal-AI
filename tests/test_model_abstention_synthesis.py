import pytest
from unittest.mock import MagicMock
from PIL import Image
from src.engine.base_engine import BaseVLMEngine
from src.pipeline.stage2_verifier import Stage2Verifier


class DummyEngine(BaseVLMEngine):
    def __init__(self, response_text: str):
        self.response_text = response_text

    def generate_multimodal(self, *args, **kwargs) -> str:
        return self.response_text

    def generate_text(self, *args, **kwargs) -> str:
        return self.response_text

    def get_engine_info(self):
        return {"name": "dummy"}


def test_ambiguous_reason_synthesizes_unclear_tag():
    """Verify that when Stage 2 notes/reason declare handwriting ambiguity, [unclear: s1 | act] is synthesized."""
    dummy_json = """{
      "silent_corrections_fixed": [
        {
          "stage1_output": "cot",
          "actual_handwritten": "cat",
          "reason": "Handwriting is ambiguous and cursive loop is unclear between cot and cat",
          "context_snippet": "The cot sat on the mat"
        }
      ],
      "total_corrections_count": 1,
      "verified_transcript": "The cat sat on the mat",
      "verification_notes": "Ambiguous handwriting on line 1"
    }"""
    engine = DummyEngine(dummy_json)
    verifier = Stage2Verifier(engine)

    img = Image.new("RGB", (100, 100), color="white")
    res = verifier.run(
        image=img,
        stage1_transcript="The cot sat on the mat."
    )

    assert len(res.silent_corrections_fixed) == 1
    diff = res.silent_corrections_fixed[0]
    assert diff.actual_handwritten == "[unclear: cot | cat]"
    assert "[unclear: cot | cat]" in res.verified_transcript


def test_confident_correction_does_not_synthesize_unclear():
    """Verify that clear, confident corrections (e.g. reverted autocorrection) apply normally."""
    dummy_json = """{
      "silent_corrections_fixed": [
        {
          "stage1_output": "village",
          "actual_handwritten": "villagc",
          "reason": "Reverting autocorrection; student wrote 'villagc' with a clear c at the end",
          "context_snippet": "in our village"
        }
      ],
      "total_corrections_count": 1,
      "verified_transcript": "in our villagc",
      "verification_notes": "Reverted autocorrection"
    }"""
    engine = DummyEngine(dummy_json)
    verifier = Stage2Verifier(engine)

    img = Image.new("RGB", (100, 100), color="white")
    res = verifier.run(
        image=img,
        stage1_transcript="They lived in our village."
    )

    assert len(res.silent_corrections_fixed) == 1
    diff = res.silent_corrections_fixed[0]
    assert diff.actual_handwritten == "villagc"
    assert "villagc" in res.verified_transcript
    assert "[unclear:" not in res.verified_transcript
