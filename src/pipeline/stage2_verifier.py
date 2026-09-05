import json
import re
from typing import Optional, Dict, Any
from PIL import Image
from src.engine.base_engine import BaseVLMEngine
from src.core.schemas import Stage2VerificationResult, AutocorrectionDiffItem
from src.prompts.stage2_verification import build_stage2_prompt, STAGE2_SYSTEM_PROMPT


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from string even if surrounded by markdown code blocks."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except Exception:
            pass

    # Resilient fallback: attempt to repair truncated JSON by balancing braces
    if first_brace != -1:
        candidate = text[first_brace:].strip()
        # Strip incomplete trailing key-value or key fragment (e.g. , "linguistic or "linguistic": )
        candidate = re.sub(r',\s*"[^"]*"?\s*:\s*[^,}\]]*$', '', candidate)
        candidate = re.sub(r',\s*"[^"]*"?\s*$', '', candidate)
        candidate = re.sub(r',\s*$', '', candidate)
        open_curlies = candidate.count("{") - candidate.count("}")
        open_squares = candidate.count("[") - candidate.count("]")
        repaired = candidate + ("]" * max(0, open_squares)) + ("}" * max(0, open_curlies))
        try:
            return json.loads(repaired)
        except Exception:
            pass

    return None


class Stage2Verifier:
    """Stage 2: Autocorrection Verification (Image + Stage 1 Transcript -> Verified Transcript)."""

    def __init__(self, engine: BaseVLMEngine):
        self.engine = engine

    def run(
        self,
        image: Image.Image,
        stage1_transcript: str,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 3072,
        thinking_mode: bool = False
    ) -> Stage2VerificationResult:
        # Truncate prompt text if extremely long to avoid exceeding context window
        clipped_text = stage1_transcript[:3000] if len(stage1_transcript) > 3000 else stage1_transcript
        prompt = build_stage2_prompt(stage1_transcript=clipped_text)

        try:
            response = self.engine.generate_multimodal(
                image=image,
                prompt=prompt,
                system_prompt=STAGE2_SYSTEM_PROMPT,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=min(max_new_tokens, 1536),
                thinking_mode=thinking_mode
            )
        except Exception as e:
            print(f"[Stage2Verifier] Warning: Autocorrection verification failed ({e}). Preserving Stage 1 transcript.")
            return Stage2VerificationResult(
                verified_transcript=stage1_transcript,
                silent_corrections_fixed=[],
                total_corrections_count=0,
                verification_notes=f"Auto-verification preserved Stage 1 transcript (Error: {e})"
            )

        parsed_data = _extract_json_from_text(response)
        if parsed_data and "verified_transcript" in parsed_data:
            diffs = []
            for item in parsed_data.get("silent_corrections_fixed", []):
                diffs.append(AutocorrectionDiffItem(
                    stage1_output=item.get("stage1_output", ""),
                    actual_handwritten=item.get("actual_handwritten", ""),
                    reason=item.get("reason", ""),
                    context_snippet=item.get("context_snippet", "")
                ))
            
            return Stage2VerificationResult(
                verified_transcript=parsed_data["verified_transcript"],
                silent_corrections_fixed=diffs,
                total_corrections_count=len(diffs),
                verification_notes=parsed_data.get("verification_notes", "")
            )

        # Fallback if raw text returned without valid JSON structure
        return Stage2VerificationResult(
            verified_transcript=stage1_transcript,
            silent_corrections_fixed=[],
            total_corrections_count=0,
            verification_notes=f"Auto-verification preserved Stage 1 transcript. (Raw response: {response[:150]}...)"
        )
