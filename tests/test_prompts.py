from src.prompts.stage1_verbatim import build_stage1_prompt
from src.prompts.stage2_verification import build_stage2_prompt
from src.prompts.stage3_errors import build_stage3_prompt
from src.prompts.stage4_rubric import build_stage4_prompt


def test_stage1_prompt_rules():
    prompt = build_stage1_prompt()
    assert "You are transcribing a handwritten exam script" in prompt
    assert "[illegible]" in prompt
    assert "[unclear: your reading]" in prompt
    assert "Do NOT fix them" in prompt


def test_stage1_prompt_few_shots():
    few_shots = [
        {"description": "Snippet 1", "transcription": "Sample transcript with errror"}
    ]
    prompt = build_stage1_prompt(few_shot_examples=few_shots)
    assert "Few-Shot Demonstration Examples" in prompt
    assert "Snippet 1" in prompt
    assert "Sample transcript with errror" in prompt


def test_stage2_prompt():
    prompt = build_stage2_prompt(stage1_transcript="student wrote text")
    assert "student wrote text" in prompt
    assert "silent_corrections_fixed" in prompt


def test_stage3_prompt():
    prompt = build_stage3_prompt(verified_transcript="student verified text")
    assert "student verified text" in prompt
    assert "spelling | grammar | syntax | punctuation" in prompt


def test_stage4_prompt():
    rubric = {
        "subject": "Bangla",
        "question_type": "Creative Question",
        "total_marks": 10.0,
        "criteria": [{"id": "c1", "name": "Content", "max_marks": 5.0, "description": "Depth"}],
        "penalties": {"spelling_error_deduction": 0.25}
    }
    errors = {"errors": []}
    prompt = build_stage4_prompt(
        verified_transcript="Sample text",
        error_list=errors,
        rubric_data=rubric,
        thematic_context="Historical context"
    )
    assert "Bangla" in prompt
    assert "Creative Question" in prompt
    assert "Historical context" in prompt
