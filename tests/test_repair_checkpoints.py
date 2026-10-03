import pytest
from scripts.repair_checkpoints import repair_text, repair_obj


def test_repair_text_latex_escapes():
    # Corrupted rightarrow from literal \r + ightarrow
    corrupted_cr = "step 1 \r ightarrow step 2"
    fixed, count = repair_text(corrupted_cr)
    assert count == 1
    assert "$\\rightarrow$" in fixed

    # Corrupted rightarrow missing backslash
    corrupted_missing = "step 1 $rightarrow$ step 2"
    fixed, count = repair_text(corrupted_missing)
    assert count == 1
    assert "$\\rightarrow$" in fixed

    # Valid string should remain untouched
    clean = "step 1 $\\rightarrow$ step 2"
    fixed, count = repair_text(clean)
    assert count == 0
    assert fixed == clean


def test_repair_obj_nested_structures():
    nested = {
        "page_no": 1,
        "transcript": "A \r ightarrow B",
        "details": [
            {"note": "$rightarrow$ arrow"},
            {"valid": "already clean"}
        ]
    }
    fixed, total_count = repair_obj(nested)
    assert total_count == 2
    assert fixed["transcript"] == "A $\\rightarrow$ B"
    assert fixed["details"][0]["note"] == "$\\rightarrow$ arrow"
    assert fixed["details"][1]["valid"] == "already clean"
