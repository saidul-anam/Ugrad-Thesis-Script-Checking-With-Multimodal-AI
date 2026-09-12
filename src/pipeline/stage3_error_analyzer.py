import json
from typing import Optional, Dict, Any
from src.engine.base_engine import BaseVLMEngine
from src.core.schemas import Stage3ErrorResult, LinguisticErrorItem
from src.prompts.stage3_errors import build_stage3_prompt, STAGE3_SYSTEM_PROMPT
from src.pipeline.stage2_verifier import _extract_json_from_text
from src.utils.linguistic_sanitizer import (
    sanitize_transcript_for_linguistic_analysis,
    verify_and_filter_stage3_errors
)


class Stage3ErrorAnalyzer:
    """Stage 3: Linguistic Error Extraction (Text Only) with Neuro-Symbolic Verification Gate."""

    def __init__(self, engine: BaseVLMEngine):
        self.engine = engine

    def run(
        self,
        verified_transcript: str,
        question_vocab: Optional[Any] = None,
        subject: str = "English",
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 3072,
        thinking_mode: bool = False
    ) -> Stage3ErrorResult:
        # 1. Deterministic pre-sanitization: strip exam headers and stitch line-wraps
        clean_transcript = sanitize_transcript_for_linguistic_analysis(verified_transcript)
        prompt = build_stage3_prompt(verified_transcript=clean_transcript)

        response = self.engine.generate_text(
            prompt=prompt,
            system_prompt=STAGE3_SYSTEM_PROMPT,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            thinking_mode=thinking_mode
        )

        parsed_data = _extract_json_from_text(response)
        if parsed_data and "errors" in parsed_data:
            raw_errors = []
            for err in parsed_data.get("errors", []):
                etype = err.get("error_type", "spelling").lower()
                raw_errors.append(LinguisticErrorItem(
                    error_type=etype,
                    erroneous_text=err.get("erroneous_text", ""),
                    suggested_correction=err.get("suggested_correction", ""),
                    context_sentence=err.get("context_sentence", ""),
                    explanation=err.get("explanation", "")
                ))

            # 2. Deterministic post-validation: filter headers, reclassify real words, protect proper nouns
            q_vocab_set = set(question_vocab) if question_vocab else None
            validated_errors = verify_and_filter_stage3_errors(
                errors=raw_errors,
                question_vocab=q_vocab_set,
                subject=subject
            )

            spelling_cnt = sum(1 for e in validated_errors if "spell" in e.error_type.lower())
            grammar_cnt = sum(1 for e in validated_errors if "gram" in e.error_type.lower())
            syntax_cnt = sum(1 for e in validated_errors if "synt" in e.error_type.lower())

            return Stage3ErrorResult(
                errors=validated_errors,
                spelling_error_count=spelling_cnt,
                grammar_error_count=grammar_cnt,
                syntax_error_count=syntax_cnt,
                punctuation_error_count=0,
                total_error_count=len(validated_errors),
                linguistic_summary=parsed_data.get("linguistic_summary", "")
            )

        # Fallback if no errors identified or parsing raw string
        return Stage3ErrorResult(
            errors=[],
            spelling_error_count=0,
            grammar_error_count=0,
            syntax_error_count=0,
            punctuation_error_count=0,
            total_error_count=0,
            linguistic_summary="No structural errors cataloged in response."
        )
