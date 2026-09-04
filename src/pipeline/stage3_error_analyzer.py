import json
from typing import Optional, Dict, Any
from src.engine.base_engine import BaseVLMEngine
from src.core.schemas import Stage3ErrorResult, LinguisticErrorItem
from src.prompts.stage3_errors import build_stage3_prompt, STAGE3_SYSTEM_PROMPT
from src.pipeline.stage2_verifier import _extract_json_from_text


class Stage3ErrorAnalyzer:
    """Stage 3: Linguistic Error Extraction (Text Only)."""

    def __init__(self, engine: BaseVLMEngine):
        self.engine = engine

    def run(
        self,
        verified_transcript: str,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 3072,
        thinking_mode: bool = False
    ) -> Stage3ErrorResult:
        prompt = build_stage3_prompt(verified_transcript=verified_transcript)

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
            errors = []
            spelling_cnt = 0
            grammar_cnt = 0
            syntax_cnt = 0
            punct_cnt = 0

            for err in parsed_data.get("errors", []):
                etype = err.get("error_type", "spelling").lower()
                if "spell" in etype:
                    spelling_cnt += 1
                elif "gram" in etype:
                    grammar_cnt += 1
                elif "synt" in etype:
                    syntax_cnt += 1
                elif "punct" in etype:
                    punct_cnt += 1

                errors.append(LinguisticErrorItem(
                    error_type=etype,
                    erroneous_text=err.get("erroneous_text", ""),
                    suggested_correction=err.get("suggested_correction", ""),
                    context_sentence=err.get("context_sentence", ""),
                    explanation=err.get("explanation", "")
                ))

            return Stage3ErrorResult(
                errors=errors,
                spelling_error_count=parsed_data.get("spelling_error_count", spelling_cnt),
                grammar_error_count=parsed_data.get("grammar_error_count", grammar_cnt),
                syntax_error_count=parsed_data.get("syntax_error_count", syntax_cnt),
                punctuation_error_count=parsed_data.get("punctuation_error_count", punct_cnt),
                total_error_count=len(errors),
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
