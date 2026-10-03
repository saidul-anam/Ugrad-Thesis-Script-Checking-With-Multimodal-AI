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
    assert "NORMALIZE VISUAL TRANSCRIPTION GLITCHES" in prompt


def test_stage3_prompt():
    prompt = build_stage3_prompt(verified_transcript="student verified text")
    assert "student verified text" in prompt
    assert "spelling | grammar | syntax" in prompt
    assert "BENEFIT OF THE DOUBT" in prompt
    assert "ZERO PUNCTUATION MARKS" in prompt


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


def test_stage1_prompt_with_syllabus():
    syllabus = [
        {"q_no": "5", "name": "Cloze Test without Clues (Education)"},
        {"q_no": "9", "name": "Story Completion (Lion and Mouse)"}
    ]
    prompt = build_stage1_prompt(question_syllabus=syllabus)
    assert "EXAM QUESTION SYLLABUS & HEADER DISAMBIGUATION" in prompt
    assert "Q9: Story Completion (Lion and Mouse)" in prompt
    assert "CRITICAL DIRECTIVE ON QUESTION HEADERS" in prompt


def test_stage2_prompt_with_syllabus():
    syllabus = [
        {"q_no": "5", "name": "Cloze Test without Clues (Education)"},
        {"q_no": "9", "name": "Story Completion (Lion and Mouse)"}
    ]
    prompt = build_stage2_prompt(stage1_transcript="Ans to the Question No-05\nA lion and a mouse", question_syllabus=syllabus)
    assert "EXAM SYLLABUS REFERENCE:" in prompt
    assert "Q9: Story Completion (Lion and Mouse)" in prompt
    assert "Question Header Digits" in prompt


def test_stage1_prompt_strikethrough_advisory():
    prompt = build_stage1_prompt(strikethrough_detected=True, strikethrough_region_count=3)
    assert "OPTICAL STRIKETHROUGH ADVISORY" in prompt
    assert "3 candidate line segment(s)" in prompt
    assert "NEVER invent or force struck text" in prompt
    assert "REVERSE-SIDE BLEED-THROUGH & GHOST INK" in prompt


def test_stage1_prompt_strikethrough_spatial_coordinates():
    from src.pipeline.stage0_strikethrough_detector import StrikethroughRegion
    regions = [
        StrikethroughRegion(x=100, y=250, w=450, h=10, y_pct=25.0, y2_pct=26.0, x_pct=10.0, x2_pct=55.0, is_multi_word=True, angle=12.0),
        StrikethroughRegion(x=300, y=500, w=120, h=6, y_pct=50.0, y2_pct=50.6, x_pct=30.0, x2_pct=42.0, is_multi_word=False, angle=0.0),
    ]
    prompt = build_stage1_prompt(strikethrough_regions=regions)
    assert "OPTICAL STRIKETHROUGH ADVISORY" in prompt
    assert "Region 1: near ~25.0% down the page" in prompt
    assert "multi-word clause strike" in prompt
    assert "angle: ~12°" in prompt
    assert "Region 2: near ~50.0% down the page" in prompt
    assert "word cross-out" in prompt


def test_stage2_prompt_strikethrough_regions():
    from src.pipeline.stage0_strikethrough_detector import StrikethroughRegion
    regions = [
        StrikethroughRegion(x=150, y=300, w=200, h=8, y_pct=30.0, y2_pct=30.8, x_pct=15.0, x2_pct=35.0, is_multi_word=True, angle=-5.0),
    ]
    prompt = build_stage2_prompt(stage1_transcript="test text", strikethrough_regions=regions)
    # Strikethrough region injection is removed to prevent false strikethrough hallucinations on ruled/underlined text
    assert "CANDIDATE STRIKETHROUGH STROKES DETECTED BY PREPROCESSING:" not in prompt


def test_stage3_prompt_struck_rule():
    prompt = build_stage3_prompt(verified_transcript="student verified text")
    assert "STRUCK-THROUGH / CANCELLED TEXT:" in prompt
    assert "NEVER extract errors from crossed-out or struck-through words" in prompt



