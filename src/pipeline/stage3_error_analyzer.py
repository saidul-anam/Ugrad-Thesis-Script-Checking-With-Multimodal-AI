import re
import json
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
from src.engine.base_engine import BaseVLMEngine
from src.core.schemas import Stage3ErrorResult, LinguisticErrorItem
from src.prompts.stage3_errors import build_stage3_prompt, STAGE3_SYSTEM_PROMPT
from src.prompts.stage3_arbitration import build_stage3_arbitration_prompt, STAGE3_ARBITRATION_SYSTEM_PROMPT
from src.pipeline.stage2_verifier import _extract_json_from_text
from src.utils.linguistic_sanitizer import (
    sanitize_transcript_for_linguistic_analysis,
    verify_and_filter_stage3_errors
)


def _extract_json_data(text: str) -> Optional[Any]:
    """Extract JSON array or object from text response."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    first_brace = text.find("{")
    first_square = text.find("[")

    if first_square != -1 and (first_brace == -1 or first_square < first_brace):
        last_square = text.rfind("]")
        if last_square > first_square:
            try:
                return json.loads(text[first_square:last_square + 1])
            except Exception:
                pass
    elif first_brace != -1:
        last_brace = text.rfind("}")
        if last_brace > first_brace:
            try:
                return json.loads(text[first_brace:last_brace + 1])
            except Exception:
                pass

    return None


class Stage3ErrorAnalyzer:
    """Stage 3: Linguistic Error Extraction (Text Only) with Neuro-Symbolic Verification Gate."""

    def __init__(self, engine: BaseVLMEngine):
        self.engine = engine

    def arbitrate_visual_errors(
        self,
        image: Optional[Image.Image],
        errors: List[LinguisticErrorItem],
        temperature: float = 0.0,
        top_p: float = 0.1,
        thinking_mode: bool = False,
        gate: Optional[Any] = None,
        q_no: Optional[str] = None,
        answer_text: Optional[str] = None,
    ) -> Tuple[List[LinguisticErrorItem], List[Dict[str, Any]]]:
        """
        Stage 3b: Handwriting Ambiguity Arbitration.

        evidence mode (gate given): the EvidenceArbitrationGate scores every gate-able error
        (any error_type) from crop re-reads, forced choice, the learned writer profile and phonetics,
        plus non-dictionary tokens of `answer_text` that Stage 3 did not flag.
        legacy mode (gate None): one whole-page VLM call over the spelling candidates only.
        Returns (confirmed_errors, cleared_list).
        """
        if gate is not None:
            confirmed, cleared, _records = gate.arbitrate(errors, q_no, answer_text=answer_text)
            return confirmed, cleared
        return self._arbitrate_legacy(image, errors, temperature, top_p, thinking_mode)

    def _arbitrate_legacy(
        self,
        image: Optional[Image.Image],
        errors: List[LinguisticErrorItem],
        temperature: float = 0.0,
        top_p: float = 0.1,
        thinking_mode: bool = False
    ) -> Tuple[List[LinguisticErrorItem], List[Dict[str, Any]]]:
        """Legacy single-call whole-page arbitration (kept for ablation)."""
        spelling_candidates = [e for e in errors if "spell" in (e.error_type or "").lower()]
        if not spelling_candidates or image is None:
            return errors, []

        candidate_dicts = [
            {
                "erroneous_text": e.erroneous_text,
                "suggested_correction": e.suggested_correction,
                "context_sentence": e.context_sentence
            }
            for e in spelling_candidates
        ]

        prompt = build_stage3_arbitration_prompt(candidate_dicts)

        try:
            response = self.engine.generate_multimodal(
                image=image,
                prompt=prompt,
                system_prompt=STAGE3_ARBITRATION_SYSTEM_PROMPT,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=1024,
                thinking_mode=thinking_mode
            )
        except Exception as ex:
            print(f"[Stage 3b Arbitrator] Warning: Visual arbitration call failed ({ex}). Preserving errors.")
            return errors, []

        parsed = _extract_json_data(response)
        items = []
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            items = parsed.get("arbitrations") or parsed.get("results") or parsed.get("errors") or [parsed]

        ambiguities_cleared = []
        ambiguous_err_texts = set()

        for item in items:
            if not isinstance(item, dict):
                continue
            cand = str(item.get("candidate") or "").strip().lower()
            verdict = str(item.get("verdict") or "").upper()
            if "AMBIGU" in verdict or "BENEFIT" in verdict or "DOUBT" in verdict:
                ambiguous_err_texts.add(cand)
                ambiguities_cleared.append({
                    "candidate": item.get("candidate"),
                    "intended_word": item.get("intended_word"),
                    "verdict": "HANDWRITING_AMBIGUITY",
                    "reason": item.get("reason", "Handwriting stroke ambiguity awarded benefit of the doubt")
                })
                print(f"[Stage 3b Arbitrator] Granted Benefit of the Doubt: '{item.get('candidate')}' -> '{item.get('intended_word')}' ({item.get('reason')})")

        # Keep only errors that were NOT classified as handwriting stroke ambiguities
        confirmed_errors = []
        for e in errors:
            e_text = (e.erroneous_text or "").strip().lower()
            if e_text in ambiguous_err_texts and "spell" in (e.error_type or "").lower():
                continue
            confirmed_errors.append(e)

        return confirmed_errors, ambiguities_cleared

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
                subject=subject,
                transcript=verified_transcript
            )

            # Diagnostic logging if substantial answer produces zero errors
            word_count = len(clean_transcript.split())
            if word_count > 50 and len(validated_errors) == 0:
                snippet = (response[:200] + "...") if len(response) > 200 else response
                print(f"[Stage 3 Error Analyzer] WARNING: Transcript of {word_count} words yielded 0 errors. Model snippet: {snippet!r}")

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
        snippet = (response[:200] + "...") if len(response) > 200 else response
        print(f"[Stage 3 Error Analyzer] WARNING: Failed to extract valid JSON from Stage 3 response. Raw snippet: {snippet!r}")
        return Stage3ErrorResult(
            errors=[],
            spelling_error_count=0,
            grammar_error_count=0,
            syntax_error_count=0,
            punctuation_error_count=0,
            total_error_count=0,
            linguistic_summary="No structural errors cataloged in response."
        )
