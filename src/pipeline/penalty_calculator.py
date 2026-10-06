"""
Tunable Penalty Calculator Engine.

Computes mathematically consistent deductions from Stage 3 linguistic error catalogs
based on per-class/level YAML configuration profiles.

Implements NCTB Board marking guidelines:
1. Zero penalty on objective / structured question types.
2. Deduplication of recurrent identical word errors.
3. Category-level and question-type deduction ceilings.
4. Transparent penalty audit reporting.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class PenaltyBreakdown:
    total_penalty: float = 0.0
    raw_penalty: float = 0.0
    is_capped: bool = False
    cap_applied: float = 0.0
    cap_reason: str = "None"
    category_counts: Dict[str, int] = field(default_factory=dict)
    category_deductions: Dict[str, float] = field(default_factory=dict)
    itemized_errors: List[Dict[str, Any]] = field(default_factory=list)


class PenaltyCalculator:
    """Calculates linguistic error deductions based on a tunable YAML configuration profile."""

    def __init__(self, config_path_or_profile: Optional[str] = None):
        self.config = self._load_config(config_path_or_profile)

    def _load_config(self, config_path_or_profile: Optional[str]) -> Dict[str, Any]:
        default_config = {
            "profile_name": "default",
            "deductions_per_error": {
                "spelling": 0.25,
                "grammar": 0.50,
                "syntax": 0.25,
                "punctuation_mechanics": 0.25,
                "illegible_omission": 0.50,
            },
            "question_type_caps": {
                "MCQ": 0.0,
                "Cloze_With_Clues": 0.0,
                "Cloze_Without_Clues": 0.0,
                "Table_Completion": 0.0,
                "Sentence_Matching": 0.0,
                "Rearrangement": 0.0,
                "Flow_Chart": 0.0,
                "Comprehension_Questions": 1.0,
                "Short_Answer": 1.0,
                "Summary": 1.5,
                "Paragraph": 2.0,
                "Story": 2.0,
                "Dialogue": 2.0,
                "default": 1.0
            },
            "overall_script_max_penalty": 6.0,
            "recurrent_error_policy": {
                "deduplicate_identical_words": True,
                "deduplicate_identical_rules": False,
                "ignore_allograph_ambiguities": True,
                "ignore_british_american_variants": True
            }
        }

        if not config_path_or_profile:
            return default_config

        p = Path(config_path_or_profile)
        if not p.exists():
            # Check in configs/penalties/
            alt = Path("configs/penalties") / f"{config_path_or_profile}.yaml"
            if alt.exists():
                p = alt

        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                    default_config.update(loaded)
            except Exception as e:
                print(f"[PenaltyCalculator] Warning: Failed loading config from {p}: {e}")

        return default_config

    def _normalize_category(self, raw_type: str) -> str:
        t = str(raw_type or "").lower().strip()
        if "spell" in t or "orthograph" in t:
            return "spelling"
        if "gramm" in t or "tense" in t or "agreement" in t:
            return "grammar"
        if "syntax" in t or "word_order" in t or "clause" in t:
            return "syntax"
        if "punct" in t or "capital" in t or "comma" in t:
            return "punctuation_mechanics"
        if "illegib" in t or "omiss" in t or "missing" in t:
            return "illegible_omission"
        return "grammar"

    def calculate_question_penalties(
        self,
        errors: List[Dict[str, Any]],
        task_type: str,
        max_marks: float
    ) -> PenaltyBreakdown:
        """
        Calculate error deductions for a single question answer block.
        """
        breakdown = PenaltyBreakdown()
        rates = self.config.get("deductions_per_error", {})
        caps = self.config.get("question_type_caps", {})
        policy = self.config.get("recurrent_error_policy", {})
        dedup_words = policy.get("deduplicate_identical_words", True)

        # 1. Zero penalty on objective / structured question types
        norm_task = str(task_type or "").replace(" ", "_").replace("-", "_")
        cap = float(caps.get(norm_task, caps.get("default", 1.0)))

        if cap <= 0.0 or not errors:
            breakdown.total_penalty = 0.0
            breakdown.raw_penalty = 0.0
            breakdown.cap_applied = cap
            if cap <= 0.0:
                breakdown.cap_reason = f"Zero linguistic penalty for objective task type '{task_type}'"
            return breakdown

        # 2. Process errors with deduplication of identical misspelled words
        seen_words = set()
        grouped_counts: Dict[str, int] = {"spelling": 0, "grammar": 0, "syntax": 0, "punctuation_mechanics": 0, "illegible_omission": 0}
        itemized: List[Dict[str, Any]] = []
        raw_total = 0.0

        for err in errors:
            cat = self._normalize_category(err.get("error_type", ""))
            word = str(err.get("erroneous_text", "")).lower().strip()

            if dedup_words and cat == "spelling" and word in seen_words:
                itemized.append({
                    "error": err,
                    "category": cat,
                    "rate": 0.0,
                    "deducted": 0.0,
                    "recurrent": True,
                    "note": f"Recurrent spelling error on '{word}' (penalized once per NCTB standard)"
                })
                continue

            if word:
                seen_words.add(word)

            rate = float(rates.get(cat, 0.25))
            raw_total += rate
            grouped_counts[cat] = grouped_counts.get(cat, 0) + 1

            itemized.append({
                "error": err,
                "category": cat,
                "rate": rate,
                "deducted": rate,
                "recurrent": False,
                "note": ""
            })

        # 3. Apply question ceiling
        effective_penalty = min(raw_total, cap)
        # Snap to nearest 0.25 increment
        snapped_penalty = round(round(effective_penalty * 4) / 4, 2)
        # Never exceed 20% of max_marks or make final marks negative
        snapped_penalty = min(snapped_penalty, round(max_marks * 0.2, 1))

        breakdown.raw_penalty = round(raw_total, 2)
        breakdown.total_penalty = snapped_penalty
        breakdown.category_counts = grouped_counts
        breakdown.itemized_errors = itemized
        breakdown.category_deductions = {k: round(v * rates.get(k, 0.25), 2) for k, v in grouped_counts.items()}

        if raw_total > snapped_penalty:
            breakdown.is_capped = True
            breakdown.cap_applied = snapped_penalty
            breakdown.cap_reason = f"Deduction capped at {snapped_penalty:.2f} marks for task type '{task_type}'"

        return breakdown
