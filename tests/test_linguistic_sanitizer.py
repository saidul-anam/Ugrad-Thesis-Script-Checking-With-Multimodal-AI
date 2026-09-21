"""
Unit tests for Deterministic Linguistic Sanitizer & Neuro-Symbolic Verification Gate.
"""

import pytest
from src.core.schemas import LinguisticErrorItem, AlignedAnswerItem
from src.utils.linguistic_sanitizer import (
    sanitize_transcript_for_linguistic_analysis,
    verify_and_filter_stage3_errors
)
from src.prompts.stage4_modular import is_objective_question, build_modular_question_prompt


def test_sanitize_transcript_headers_and_linewraps():
    raw_transcript = """
Ans: to the Q. No. 1
(A)
a) -> (iii) being displaced lives outside the country
Dans: The author become aware of the Gaza people.
I want to be a eingineer and admitted into a Dhaka Universi-
ty so that is important.
--- Page Break ---
Figure: Flow chart
Ans: to Q. No. 7
Artificial Indelligence is called AI.
"""
    cleaned = sanitize_transcript_for_linguistic_analysis(raw_transcript)
    
    # Check headers stripped
    assert "Ans: to the Q. No. 1" not in cleaned
    assert "Dans: The author" not in cleaned  # Stripped header prefix
    assert "(A)" not in cleaned
    assert "Figure: Flow chart" not in cleaned
    assert "--- Page Break ---" not in cleaned

    # Check hyphenated line wrap stitched
    assert "University" in cleaned
    assert "Universi-\nty" not in cleaned

    # Check student text preserved
    assert "Artificial Indelligence is called AI." in cleaned
    assert "eingineer" in cleaned


def test_verify_and_filter_stage3_errors():
    errors = [
        LinguisticErrorItem(
            error_type="spelling",
            erroneous_text="Dans",
            suggested_correction="Ans",
            context_sentence="Dans: The author become aware",
            explanation="Typographical error"
        ),
        LinguisticErrorItem(
            error_type="spelling",
            erroneous_text="really of Gaza",
            suggested_correction="reality of Gaza",
            context_sentence="back to the really of Gaza",
            explanation="Incorrect word form"
        ),
        LinguisticErrorItem(
            error_type="spelling",
            erroneous_text="Pesteun was an French scientists",
            suggested_correction="Pasteur was a French scientist",
            context_sentence="Pesteun was an French scientists he noticed",
            explanation="Multiple errors"
        ),
        LinguisticErrorItem(
            error_type="spelling",
            erroneous_text="healthrisk",
            suggested_correction="health risks",
            context_sentence="Taking healthrisk in pregnancy",
            explanation="Compound word"
        ),
        LinguisticErrorItem(
            error_type="spelling",
            erroneous_text="eingineer",
            suggested_correction="engineer",
            context_sentence="I want to be a eingineer",
            explanation="True spelling error"
        ),
        LinguisticErrorItem(
            error_type="spelling",
            erroneous_text="familyes",
            suggested_correction="families",
            context_sentence="take care of your familyes",
            explanation="True spelling error"
        ),
        LinguisticErrorItem(
            error_type="spelling",
            erroneous_text="Gaza",
            suggested_correction="Gaza",
            context_sentence="situation of Gaza",
            explanation="Proper noun"
        ),
        LinguisticErrorItem(
            error_type="punctuation",
            erroneous_text=",",
            suggested_correction=".",
            context_sentence="He went home , then slept",
            explanation="Missing comma or period"
        )
    ]

    question_vocab = {"gaza", "pasteur", "france"}
    validated = verify_and_filter_stage3_errors(errors, question_vocab=question_vocab, subject="English")

    err_map = {e.erroneous_text: e for e in validated}

    # 1. Header 'Dans' dropped
    assert "Dans" not in err_map

    # 2. 'really' reclassified to grammar because it's a real dictionary word
    assert "really of Gaza" in err_map
    assert err_map["really of Gaza"].error_type in ["grammar", "syntax"]

    # 3. Multi-word phrase 'Pesteun was an French scientists' reclassified from spelling
    assert "Pesteun was an French scientists" in err_map
    assert err_map["Pesteun was an French scientists"].error_type != "spelling"

    # 4. 'healthrisk' (valid compound note) suppressed
    assert "healthrisk" not in err_map

    # 5. 'Gaza' (question vocab) dropped
    assert "Gaza" not in err_map

    # 6. True spelling errors PRESERVED
    assert "eingineer" in err_map
    assert err_map["eingineer"].error_type == "spelling"
    assert "familyes" in err_map
    assert err_map["familyes"].error_type == "spelling"

    # 7. Punctuation errors completely dropped
    assert "," not in err_map


def test_is_spurious_reversion_and_flowchart_filtering():
    from src.pipeline.stage2_verifier import is_spurious_reversion
    assert is_spurious_reversion("work", "woork") is True
    assert is_spurious_reversion("gaza", "graza") is True
    assert is_spurious_reversion("which", "cohich") is True
    assert is_spurious_reversion("that", "thad") is True

    # Test flowchart cell filtering
    flowchart_errs = [
        LinguisticErrorItem(
            error_type="spelling",
            erroneous_text="woork",
            suggested_correction="work",
            context_sentence="| Dropping out of school | Begining full time woork in laws house |",
            explanation="Incorrect vowel duplication"
        ),
        LinguisticErrorItem(
            error_type="spelling",
            erroneous_text="Begining",
            suggested_correction="Beginning",
            context_sentence="| Dropping out of school | Begining full time woork in laws house |",
            explanation="Missing double consonant 'n'"
        )
    ]
    filtered = verify_and_filter_stage3_errors(flowchart_errs, subject="English")
    assert len(filtered) == 0


def test_objective_question_deduction_policy():
    # Test objective detection
    assert is_objective_question("1(A)", "MCQ", "Choose the best answer") is True
    assert is_objective_question("2", "Flow Chart", "Complete the flowchart") is True
    assert is_objective_question("4", "Cloze Test with Clues", "Fill in blanks") is True
    assert is_objective_question("6", "Rearranging", "Rearrange the sentences") is True

    # Test subjective continuous writing
    assert is_objective_question("7", "Paragraph", "Write a paragraph on AI") is False
    assert is_objective_question("9", "Story Completion", "Complete the story") is False
    assert is_objective_question("10", "Informal Letter", "Write a letter to friend") is False

    # Test prompt generation for objective question sets max penalty to 0.0
    ans_q2 = AlignedAnswerItem(
        q_no="2",
        q_name="Flow Chart",
        answer_text="1. Dropping out\n2. Work in laws house\n3. Taking healthrisk",
        page_numbers=[1],
        errors=[{"error_type": "spelling", "erroneous_text": "healthrisk", "suggested_correction": "health risks"}],
        word_count=12,
        character_count=60
    )
    prompt = build_modular_question_prompt(ans_q2, "Complete the flowchart", max_marks=10.0)
    assert "ZERO LINGUISTIC PENALTY" in prompt
    assert '"linguistic_penalty": 0.0' in prompt


def test_verify_and_filter_right_edge_truncations():
    """Verify that edge-truncated words at line ends are suppressed, while mid-line errors are kept."""
    transcript = (
        "we should increase the use of renewabl\n"
        "energy in our daily life.\n"
        "In 1980 the highest energy was pro\n"
        "duced from coal.\n"
        "They also lived in a small villa\n"
        "near the river.\n"
        "He will renewabl energy and see their familyes tomorrow.\n"
    )

    errors = [
        # Line-end spelling truncation
        LinguisticErrorItem(
            error_type="spelling",
            erroneous_text="renewabl",
            suggested_correction="renewable",
            context_sentence="we should increase the use of renewabl",
            explanation="Orthographic error; missing final vowel."
        ),
        # Line-end grammar truncation
        LinguisticErrorItem(
            error_type="grammar",
            erroneous_text="pro",
            suggested_correction="produced",
            context_sentence="In 1980 the highest energy was pro",
            explanation="Incomplete verb form"
        ),
        # Line-end spelling truncation
        LinguisticErrorItem(
            error_type="spelling",
            erroneous_text="villa",
            suggested_correction="village",
            context_sentence="They also lived in a small villa",
            explanation="Spelling error"
        ),
        # Mid-line genuine spelling error (same word 'renewabl', but mid-line!)
        LinguisticErrorItem(
            error_type="spelling",
            erroneous_text="renewabl",
            suggested_correction="renewable",
            context_sentence="He will renewabl energy and see their familyes tomorrow.",
            explanation="Spelling error"
        ),
        # End-of-line genuine spelling error (not a truncation prefix!)
        LinguisticErrorItem(
            error_type="spelling",
            erroneous_text="familyes",
            suggested_correction="families",
            context_sentence="He will renewabl energy and see their familyes tomorrow.",
            explanation="Spelling error"
        )
    ]

    filtered = verify_and_filter_stage3_errors(errors, transcript=transcript, subject="English")

    # Only genuine non-edge-truncated errors should remain:
    # 1. 'renewabl' when mid-line
    # 2. 'familyes' (wrong suffix, not prefix truncation)
    assert len(filtered) == 2
    err_texts = [e.erroneous_text for e in filtered]
    assert "familyes" in err_texts
    assert "renewabl" in err_texts  # Mid-line occurrence preserved


def test_se_11_q1_0001_renewabl_truncation_case():
    """Verify that the exact line 92 'renewabl' from SE_11_Q1_0001 is suppressed."""
    canonical_transcript_q8 = """
In the given pie-chart we can see that
the sources of the Use electricity in
1980.

In 1980 The main source of generation
electricity in Use was [struck: 24%] Natural
Coal which was the highest 46% of
gas

--- Page Break ---

the total sources . It was the
highest demand for generating electricity.
After that Natural gas the second
highest option for generating electricity
which was 24 %. Then it was 16 power which was 16% of total .
12% electricity a was generated from
oil . A Now the least used one was
Nuclear only 2%.

So, we can make a [struck: col] conclusion that
they [struck: used] used so much non-renewal
energy which are limited in nature. If
we want to save these natural resources
we should increase the use of renewabl
energy.
"""
    error = LinguisticErrorItem(
        error_type="spelling",
        erroneous_text="renewabl",
        suggested_correction="renewable",
        context_sentence="we should increase the use of renewabl energy",
        explanation="Orthographic error; missing final vowel."
    )

    filtered = verify_and_filter_stage3_errors([error], transcript=canonical_transcript_q8, subject="English")
    assert len(filtered) == 0  # Fully suppressed as right-edge truncation!
